from fastchat.conversation import get_conv_template
from loguru import logger
from tqdm import trange
import torch
from .base_model import Model

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from contextlib import nullcontext

import torch
import torch.nn as nn
import gc
from aisafetylab.attack.prompt_manager import Seq, MergedSeq, msg_to_seq
from aisafetylab.utils import (
    ReturnStruct,
    autocast_decorator,
    compute_perplexity,
    get_nonascii_toks,
    llm_loader,
    loss_seqs,
)


class LocalModel(Model):
    def __init__(self, model, tokenizer, model_name, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        # self.pos_to_token_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if 'llama3' in self.model_name.lower():
            self.eos_token_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.pad_token_id = self.tokenizer.pad_token_id
        
        try:
            _model_name = model_name
            if 'vicuna' in model_name:
                _model_name = 'vicuna_v1.1'
            self.conversation = get_conv_template(_model_name)
        except KeyError:
            logger.error(f'Invalid model_name: {model_name}. Refer to '
                          'https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py '
                          'for possible options and templates.')
            raise  # Continue raising the KeyError

        if model_name == 'llama-2':
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == 'zero_shot':
            self.conversation.roles = tuple(['### ' + r for r in self.conversation.template.roles])
            self.conversation.sep = '\n'

        if generation_config is None:
            generation_config = {}
        self.generation_config = generation_config
        self.device = next(self.model.parameters()).device

    def set_system_message(self, system_message: str):
        """
        Sets a system message for the conversation.
        :param str system_message: The system message to set.
        """
        self.conversation.system_message = system_message

    def get_response(self, prompts_list, max_n_tokens=None, no_template=False):
        """
        Get response from the model.
        :param prompts_list: List of prompts to get response for.
        :param max_n_tokens: Maximum number of tokens to generate.
        :param temperature: Temperature for sampling.
        :param no_template: Whether to use the template or not.
        :return: List of responses.
        """
        if not no_template:
            full_prompts = prompts_list
        else:
            full_prompts = []
            for prompt in prompts_list:
                full_prompt = self.apply_chat_template(prompt)
                full_prompts.append(full_prompt)

        full_prompts_list = full_prompts
        if 'llama' in self.model_name.lower():
            max_n_tokens += 1  # +1 to account for the first special token (id=29871) for llama2 models
        batch_size = len(full_prompts_list)
        vocab_size = len(self.tokenizer.get_vocab())
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True).to(self.device)
        input_ids = inputs["input_ids"]
        # Batch generation
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_n_tokens,  
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,  # added for Mistral
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_ids = output.sequences
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_ids.shape[1]:]  
        if 'llama2' in self.model_name.lower():
            output_ids = output_ids[:, 1:]  # ignore the first special token (id=29871)

        generated_texts = self.tokenizer.batch_decode(output_ids)
        # output.scores: n_output_tokens x batch_size x vocab_size (can be counter-intuitive that batch_size doesn't go first)
        logprobs_tokens = [torch.nn.functional.log_softmax(output.scores[i_out_token], dim=-1).cpu().numpy() 
                           for i_out_token in range(len(output.scores))]
        if 'llama2' in self.model_name.lower():
            logprobs_tokens = logprobs_tokens[1:]  # ignore the first special token (id=29871)

        logprob_dicts = [[{self.pos_to_token_dict[i_vocab]: logprobs_tokens[i_out_token][i_batch][i_vocab]
                         for i_vocab in range(vocab_size)} 
                         for i_out_token in range(len(logprobs_tokens))
                        ] for i_batch in range(batch_size)]

        outputs = [{'text': generated_texts[i_batch],
                    'logprobs': logprob_dicts[i_batch],
                    'n_input_tokens': len(input_ids[i_batch][input_ids[i_batch] != 0]),  # don't count zero-token padding
                    'n_output_tokens': len(output_ids[i_batch]),
                   } for i_batch in range(batch_size)
        ]

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs



    def generate(self, input_ids, gen_config=None, batch=False):
        input_ids = input_ids.to(self.model.device)
        if not batch:
            if gen_config is None:
                gen_config = self.model.generation_config
                gen_config.max_new_tokens = 64

            attn_masks = torch.ones_like(input_ids).to(self.model.device)
            output_ids = self.model.generate(input_ids,
                                        attention_mask=attn_masks,
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id)[0]
            return output_ids[input_ids.size(1):]

        else:
            raise NotImplementedError

    def apply_chat_template(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if self.model_name == 'llama-2':
            conversation = self.conversation.copy()
            if messages[0]['role'] == 'system':
                conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            for msg in messages:
                conversation.append_message(msg['role'], msg['content'])
            return conversation.get_prompt() + '[/INST]'
        elif 'vicuna' in self.model_name:
            conversation = self.conversation.copy()
            if messages[0]['role'] == 'system':
                conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            for msg in messages:
                conversation.append_message(msg['role'], msg['content'])
            prompt = conversation.get_prompt()
            return prompt.replace('user:', 'User:').strip() + ' ASSISTANT:'
        else:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def batch_chat(self, batch_messages, batch_size=8):
        prompts = []
        for messages in batch_messages:
            prompt = self.apply_chat_template(messages)
            prompts.append(prompt)

        responses = []
        for i in trange(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True).to(self.device)
            out = self.model.generate(**inputs, **self.generation_config)
            for j, input_ids in enumerate(inputs["input_ids"]):
                response = self.tokenizer.decode(out[j][len(input_ids):], skip_special_tokens=True)
                responses.append(response)

        return responses

    def chat(self, messages):
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        if self.model_name == 'llama-2':
            conversation = self.conversation.copy()
            if messages[0]['role'] == 'system':
                conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            for msg in messages:
                conversation.append_message(msg['role'], msg['content'])
            prompt = conversation.get_prompt() + '[/INST]'
            inputs = self.tokenizer([prompt], return_tensors='pt').to(self.device)
        elif 'vicuna' in self.model_name:
            prompt = self.apply_chat_template(messages)
            inputs = self.tokenizer([prompt], return_tensors='pt').to(self.device)
        else:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True).to(self.device)

        out = self.model.generate(**inputs, **self.generation_config)
        response = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return response


class LLM(nn.Module):
    def __init__(self, params, verbose=False) -> None:
        super().__init__()
        self.params = params
        self.verbose = verbose
        if (params.llm_params.lora_params is None):
            lora_checkpoint = None
            lora_config = None
        elif (params.llm_params.lora_params.warmstart == False):
            lora_checkpoint = None
            lora_config = params.llm_params.lora_params.lora_config
        else:
            lora_checkpoint = params.llm_params.lora_params.lora_checkpoint
            lora_config = params.llm_params.lora_params.lora_config

        self.model, self.tokenizer, self.embedding_matrix = llm_loader(
            checkpoint=params.llm_params.checkpoint, 
            device=params.llm_params.device, 
            dtype=params.llm_params.dtype, 
            freeze=params.llm_params.freeze, 
            lora_checkpoint=lora_checkpoint, 
            lora_config=lora_config, 
            verbose=verbose
        )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # TODO: This is a hack I added because Falcon-7b-isntruct doe snot have a pad token
                # We might run into trouble here because the Seq class will automatically treat any eos_token as a pad_token and set the padding mask to 0 for this token
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.params.llm_params.device
        if self.params.allow_non_ascii:
            self.disallowed_ids = None
        else:
            self.disallowed_ids = get_nonascii_toks(self.tokenizer, device=self.device)

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path, save_embedding_layers=True)

    def model_forward(self, query_seq, use_basemodel=False):
        # reorder such that all masked tokens are on the left
        mask = query_seq.mask
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)

        # print(self.model)
        toggle = False
        if use_basemodel:
            self.model.disable_adapters()
            toggle = True
        if query_seq.is_hard:
            ids = query_seq.ids
            sorted_ids = ids.gather(1, indices)
            shifted_sorted_pred_logits = self.model(
                input_ids=sorted_ids, attention_mask=sorted_mask
            ).logits
        else:
            embeds = query_seq.get_embed(self.embedding_matrix)
            indices_extended = indices[:, :, None].repeat(1, 1, embeds.shape[-1])
            sorted_embeds = embeds.gather(1, indices_extended)
            shifted_sorted_pred_logits = self.model(
                inputs_embeds=sorted_embeds, attention_mask=sorted_mask
            ).logits
        if toggle:
            self.model.enable_adapters()

        # reverse the sort to get the original order (also account for the shift)
        dummy_pred_logits = torch.zeros_like(shifted_sorted_pred_logits[:, :1, :])
        sorted_pred_logits = torch.cat(
            [dummy_pred_logits, shifted_sorted_pred_logits[:, :-1, :]], dim=1
        )
        reverse_indices = indices.argsort(dim=1)
        reverse_indices_extended = reverse_indices[:, :, None].repeat(
            1, 1, sorted_pred_logits.shape[-1]
        )
        shifted_pred_logits = sorted_pred_logits.gather(1, reverse_indices_extended)
        pred_logits = torch.cat(
            [shifted_pred_logits[:, 1:, :], shifted_sorted_pred_logits[:, -1:, :]],
            dim=1,
        )

        if self.disallowed_ids is not None:
            pred_logits[:, :, self.disallowed_ids] = -1e4
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            for i in range(pred_logits.shape[0]):
                if torch.isnan(pred_logits[i]).any():
                    print(i, "-th logits..........", pred_logits[i])
                    print("shifted_sorted_pred_logits", shifted_sorted_pred_logits[i])
                    print("ids........", ids[i])
                    print("sorted_masks.......", sorted_mask[i])
                    print("sorted_ids", sorted_ids[i])
            raise RuntimeError(f"NaN in pred_logits: {pred_logits}")
        new_mask = torch.ones_like(mask)
        new_mask[:, :-1] = mask[:, 1:]
        seq = Seq(
            logits=pred_logits,
            mask=new_mask,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        return seq

    @autocast_decorator
    def compute_pred_loss_teacher_forced(self, loss_params, label=None, **kwargs):
        gen_seqs = self.generate_teacher_forced(**kwargs)
        if label is None:
            label = gen_seqs.response_teacher
        loss_return = loss_seqs(gen_seqs.response_dist, label, **loss_params)

        pred_loss_return = ReturnStruct(
            loss=loss_return.loss,
            loss_masked=loss_return.loss_masked,
            loss_batch=loss_return.loss_batch,
            query=gen_seqs.query,
            response_teacher=gen_seqs.response_teacher,
            response_dist=gen_seqs.response_dist,
            label=label,
            perplexity=gen_seqs.perplexity,
            perplexity_per_token_masked=gen_seqs.perplexity_per_token_masked,
        )
        return pred_loss_return

    @autocast_decorator
    def generate_teacher_forced(
        self, key, detach_query=False, use_basemodel=False, **context
    ):
        query_seq, response_teacher_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        assert not response_teacher_seq.is_empty
        full_query_seq = MergedSeq([query_seq, response_teacher_seq])
        if detach_query:
            full_query_seq = full_query_seq.clone().detach()

        pred_full_query_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel
        )
        response_dist_seq = pred_full_query_seq[
            :, -response_teacher_seq.seq_len - 1 : -1
        ]
        perplexity, perplexity_per_token_masked = compute_perplexity(
            id_seq=response_teacher_seq, likelihood_seq=response_dist_seq
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_teacher=response_teacher_seq,
            response_dist=response_dist_seq,
            perplexity=perplexity,
            perplexity_per_token_masked=perplexity_per_token_masked,
        )
        return return_seqs

    def get_next_token(self, key, use_basemodel=False, **context):
        query_seq, key_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        full_query_seq = MergedSeq([query_seq, key_seq])

        pred_dist_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel
        )
        next_dist_seq = pred_dist_seq[:, -1:]

        return_seqs = ReturnStruct(query=full_query_seq, response_dist=next_dist_seq)
        return return_seqs

    def generate_autoregressive(
        self, key, use_basemodel=False, max_new_tokens=None, **context
    ):
        query_seq = self.prepare_prompt(context, up_to_key=key)

        mask = query_seq.mask
        ids = query_seq.ids
        sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)
        sorted_ids = ids.gather(1, indices)

        generation_config = self.model.generation_config
        if self.disallowed_ids is not None:
            generation_config.suppress_tokens = self.disallowed_ids.tolist()
        generation_config.renormalize_logits = True

        if max_new_tokens is None:
            max_new_tokens = self.params.gen_params.max_new_tokens

        gen_params = dict(self.params.gen_params)
        gen_params["max_new_tokens"] = max_new_tokens

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            output = self.model.generate(
                input_ids=sorted_ids,
                attention_mask=sorted_mask,
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                **gen_params,
            )

        output_ids = output.sequences[:, ids.shape[1] :]

        response_sample_seq = Seq(
            ids=output_ids, tokenizer=self.tokenizer, device=self.device
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_sample=response_sample_seq,
        )
        return return_seqs

    def prepare_prompt(self, context, up_to_key=None, return_key_seq=False):
        seqs = []
        for msg_dct in self.params.prompt_manager.prompt_template:
            if (
                up_to_key is not None
                and up_to_key == msg_dct.key
                and not return_key_seq
            ):
                break

            seq = msg_to_seq(
                msg=msg_dct.msg,
                tokenizer=self.tokenizer,
                device=self.device,
                context=context,
            )
            if up_to_key is not None and up_to_key == msg_dct.key and return_key_seq:
                break
            seqs.append(seq)

        merged_prompt_seq = MergedSeq(seqs)
        if return_key_seq:
            return merged_prompt_seq, seq
        else:
            return merged_prompt_seq