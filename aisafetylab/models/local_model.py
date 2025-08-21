from fastchat.conversation import get_conv_template
from loguru import logger
from tqdm import trange
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from vllm import LLM as VLLM
import numpy as np
from .base_model import Model
from .sequence import *
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the original repo by Facebook.
from contextlib import nullcontext

import torch
import torch.nn as nn
import gc
# from aisafetylab.attack.prompt_manager.sequence import Seq, MergedSeq, msg_to_seq
from aisafetylab.utils import (
    ReturnStruct,
    autocast_decorator,
    compute_perplexity,
    get_nonascii_toks,
    llm_loader,
    loss_seqs,
)


class LocalModel(Model):
    def __init__(self, model=None, tokenizer=None, model_name=None, generation_config=None, model_path=None, tokenizer_path=None, device=None, vllm_mode=False):
        if model is not None:
            self.model = model
        else:
            assert model_path is not None, "model_path must be provided if model is not given"
            if vllm_mode:
                if device is None:
                    self.model = VLLM(model_path)
                else:
                    self.model = VLLM(model_path, device=device)
            else:
                if device is None:
                    self.model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda').eval()
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        
        if tokenizer is not None: 
            self.tokenizer = tokenizer
        else:
            if tokenizer_path is None:
                tokenizer_path = model_path
                logger.warning("tokenizer_path is not provided, using model_path as tokenizer_path")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', trust_remote_code=True)
        
        self.vllm_mode = vllm_mode
        logger.info(f'Using VLLM mode: {self.vllm_mode}')
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            logger.warning("tokenizer's pad_token is None, setting pad_token to eos_token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        # self.pos_to_token_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id
        
        try:
            _model_name = model_name
            if 'vicuna' in model_name:
                _model_name = 'vicuna_v1.1'
            self.conversation = get_conv_template(_model_name)
        except:
            logger.warning("using default conversation template")

        if model_name == 'llama-2':
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == 'zero_shot':
            self.conversation.roles = tuple(['### ' + r for r in self.conversation.template.roles])
            self.conversation.sep = '\n'

        if not vllm_mode:
            if generation_config is None:
                generation_config = {}
            self.generation_config = generation_config

        else:
            if generation_config is None:
                self.generation_config = SamplingParams()
            else:
                if isinstance(generation_config, SamplingParams):
                    self.generation_config = generation_config
                else:
                    # handle incorrect keys
                    # removed keys
                    generation_config = self.transfer_generation_config_to_vllm(generation_config)
                    self.generation_config = SamplingParams(**generation_config)
        
        # check this for multi gpu scenario
        # self.device = next(self.model.parameters()).device
        if device:
            self.device = device
        else:
            if not vllm_mode:
                self.device = next(self.model.parameters()).device
            else:
                self.device = self.model.llm_engine.device_config.device
            
    def transfer_generation_config_to_vllm(self, generation_config):
        removed_keys = set(['do_sample'])
        change_name_keys = {'max_new_tokens': 'max_tokens'}
        new_generation_config = {}
        for k, v in generation_config.items():
            if k == 'top_k' and v == 0:
                # logger.info('Change top_k from 0 to -1 for VLLM')
                new_generation_config['top_k'] = -1
            elif k in removed_keys:
                # logger.info(f'Remove sampling parameter "{k}" as it is not used in VLLM')
                continue
            elif k in change_name_keys:
                # logger.info(f'Change sampling parameter "{k}" to "{change_name_keys[k]}" for VLLM')
                new_generation_config[change_name_keys[k]] = v
            else:
                new_generation_config[k] = v
        
        return new_generation_config

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

            attn_masks = torch.ones_like(input_ids).to(self.model.device)
            output_ids = self.model.generate(input_ids,
                                        attention_mask=attn_masks,
                                        generation_config=gen_config,
                                        pad_token_id=self.tokenizer.pad_token_id)[0]
            return output_ids[input_ids.size(1):]

        else:
            raise NotImplementedError

    def apply_chat_template(self, messages, force_prefill=False):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        prefill = False
        if messages[-1]['role'] == 'assistant' or force_prefill:
            prefill = True
        
        try:
            # first try the model's own tokenizer
            if prefill:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        except Exception as e:
            conversation = self.conversation.copy()
            if messages[-1]['role'] != 'assistant':
                messages.append({"role": "assistant", "content": None})
        
            if messages[0]['role'] == 'system':
                conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            for msg in messages:
                conversation.append_message(msg['role'], msg['content'])
            
            prompt = conversation.get_prompt()
            if conversation.name == 'vicuna_v1.1':
                prompt = prompt.replace('user:', 'User:').replace('assistant:', 'ASSISTANT:')
            
            
        if self.tokenizer.bos_token and prompt.startswith(self.tokenizer.bos_token):
            # if there are two bos tokens, remove one
            # prompt = prompt.replace(self.tokenizer.bos_token, '', 1).lstrip()
            prompt = prompt.replace(self.tokenizer.bos_token, '', 1)
        
        if self.tokenizer.bos_token and not prompt.startswith(self.tokenizer.bos_token):
            prompt = self.tokenizer.bos_token + prompt
        
        logger.info(f'prompt before check prefill: {prompt}')
        if prefill:
            if self.tokenizer.eos_token and prompt.strip().endswith(self.tokenizer.eos_token):
                idx = prompt.rindex(self.tokenizer.eos_token)
                prompt = prompt[:idx].rstrip()
                logger.info(f'prompt after check prefill: {prompt}')

        return prompt
    
    def get_ppl(self, batch_messages):
        ppls = []
        device = self.model.device
        for messages in batch_messages:
            input_text = self.apply_chat_template(messages[:-1])
            output_text = self.apply_chat_template(messages).replace(input_text, '', 1).lstrip()
            # logger.debug(f"input_text: {input_text}\noutput_text: {output_text}")
            inputs = self.tokenizer([input_text], return_tensors="pt", truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            outputs = self.tokenizer([output_text], return_tensors="pt", truncation=True, padding=True, add_special_tokens=False)
            output_ids = outputs["input_ids"].to(device)
            output_attention_mask = outputs['attention_mask'].to(device)
            
            concat_input_ids = torch.cat([input_ids, output_ids], dim=1)
            concat_attention_mask = torch.cat([attention_mask, output_attention_mask], dim=1)
            
            labels = concat_input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            labels[:, :input_ids.shape[1]] = -100
            
            outputs = self.model(concat_input_ids, attention_mask=concat_attention_mask)
            logits = outputs.logits
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            # print(logits.shape, labels.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # logits = logits.view(logits.size(0), logits.size(2), logits.size(1))
            shift_logits = shift_logits.permute(0, 2, 1)
            loss = criterion(shift_logits, shift_labels)
            # print(loss)
            # ppl = torch.exp((loss[:, -output_ids.size(1):]).mean(-1)).item()
            ipt_len = input_ids.size(1)
            loss = loss[0, ipt_len-1:]
            ppl = np.exp(loss.mean().item())
            # logger.debug(f'loss: {loss.mean()}, ppl: {ppl}')

            ppls.append(ppl)
        
        return ppls
            

    def batch_chat(self, batch_messages, batch_size=8, skip_special_tokens=True, use_tqdm=True, force_prefill=False, **kwargs):
        prompts = []
        for messages in batch_messages:
            prompt = self.apply_chat_template(messages, force_prefill=force_prefill)
            prompts.append(prompt)

        responses = []
        
        if self.vllm_mode:
            if "sampling_params" in kwargs:
                temp_generation_config = kwargs["sampling_params"]
            else:
                temp_generation_config = self.generation_config.clone()
                kwargs = self.transfer_generation_config_to_vllm(kwargs)
                for k in kwargs:
                    if k in self.generation_config.__annotations__.keys():
                        setattr(temp_generation_config, k, kwargs[k])
                        
            # logger.debug(f'input_texts: {input_texts}')

            outputs = self.model.generate(prompts, temp_generation_config, use_tqdm=use_tqdm)
            responses = [output.outputs[0].text for output in outputs]
            
        else:
        
            temp_generation_config = self.generation_config.copy()
            
            if "generation_config" in kwargs:
                temp_generation_config = kwargs["generation_config"]
            else:
                temp_generation_config = self.generation_config.copy()
                for k in kwargs:
                    if k in self.generation_config.keys():
                        setattr(temp_generation_config, k, kwargs[k])
                        
            if use_tqdm:
                bar = trange(0, len(prompts), batch_size)
            else:
                bar = range(0, len(prompts), batch_size)
            
            for i in bar:
                batch_prompts = prompts[i:i + batch_size]
                # logger.debug(f'batch_prompts: {batch_prompts}')
                inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
                out = self.model.generate(**inputs, **temp_generation_config)
                for j, input_ids in enumerate(inputs["input_ids"]):
                    # logger.debug(f'complete gen: {self.tokenizer.decode(out[j], skip_special_tokens=True)}')
                    response = self.tokenizer.decode(out[j][len(input_ids):], skip_special_tokens=skip_special_tokens)
                    responses.append(response)

        return responses

    def chat(self, messages, use_chat_template=True, **kwargs):
        if isinstance(messages, str) and use_chat_template == True:
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        # logger.debug(f'prompt: {prompt}')
        if use_chat_template == True:
            prompt = self.apply_chat_template(messages)
        else:
            prompt = messages
        
        if self.vllm_mode:
            if "sampling_params" in kwargs:
                temp_generation_config = kwargs["sampling_params"]
            else:
                temp_generation_config = self.generation_config.clone()
                # resolve name conflicts in vllm and transformers
                kwargs = self.transfer_generation_config_to_vllm(kwargs)
                for k in kwargs:
                    if k in self.generation_config.__annotations__.keys():
                        setattr(temp_generation_config, k, kwargs[k])

            outputs = self.model.generate([prompt], temp_generation_config)
            response = outputs[0].outputs[0].text
            logger.debug(f'In LocalModel chat, using vllm_mode. Prompt: {prompt}\nResponse: {response}')
            
        else:
            inputs = self.tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to(self.device)

            temp_generation_config = self.generation_config.copy()
            
            if "generation_config" in kwargs:
                temp_generation_config = kwargs["generation_config"]
            else:
                temp_generation_config = self.generation_config.copy()
                temp_generation_config.update(kwargs)
                        
            # logger.debug(f'Generation config: {temp_generation_config}')
            
            with torch.no_grad():
                out = self.model.generate(**inputs, **temp_generation_config)
                
            response = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        # response = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=False)

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
