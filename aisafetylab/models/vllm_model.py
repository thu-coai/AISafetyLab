from loguru import logger
from .base_model import Model
from .sequence import *
from fastchat.conversation import get_conv_template
from vllm import LLM, SamplingParams
class VLLMModel(Model):
    def __init__(self, model, tokenizer, model_name, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.pad_token_id = self.tokenizer.pad_token_id
        try:
            _model_name = model_name
            if 'vicuna' in model_name:
                _model_name = 'vicuna_v1.1'
            self.conversation = get_conv_template(_model_name)
        except KeyError:
            logger.warning("using default conversation template")

        if model_name == 'llama-2':
            self.conversation.sep2 = self.conversation.sep2.strip()

        if model_name == 'zero_shot':
            self.conversation.roles = tuple(['### ' + r for r in self.conversation.template.roles])
            self.conversation.sep = '\n'
        
        if generation_config is None:
            self.generation_config = SamplingParams()
        else:
            if isinstance(generation_config, SamplingParams):
                self.generation_config = generation_config
            else:
                # handle incorrect keys
                # removed keys
                removed_keys = set(['do_sample'])
                change_name_keys = {'max_new_tokens': 'max_tokens'}
                new_generation_config = {}
                for k, v in generation_config.items():
                    if k == 'top_k' and v == 0:
                        logger.info('Change top_k from 0 to -1 for VLLM')
                        new_generation_config['top_k'] = -1
                    elif k in removed_keys:
                        logger.info(f'Remove sampling parameter "{k}" as it is not used in VLLM')
                        continue
                    elif k in change_name_keys:
                        logger.info(f'Change sampling parameter "{k}" to "{change_name_keys[k]}" for VLLM')
                        new_generation_config[change_name_keys[k]] = v
                    else:
                        new_generation_config[k] = v
                generation_config = new_generation_config
                self.generation_config = SamplingParams(**generation_config)


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
    
    def chat(self, messages, use_tqdm=True, **kwargs):
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]
        input_text = self.apply_chat_template(messages)
        
        if "sampling_params" in kwargs:
            temp_generation_config = kwargs["sampling_params"]
        else:
            temp_generation_config = self.generation_config.clone()
            for k in kwargs:
                if k in self.generation_config.__annotations__.keys():
                    setattr(temp_generation_config, k, kwargs[k])

        outputs = self.model.generate([input_text], temp_generation_config, use_tqdm=use_tqdm)
        generated_text = outputs[0].outputs[0].text
        return generated_text
        
    def batch_chat(self, batch_messages, use_tqdm=True, force_prefill=False, **kwargs):
        # print(f'get into batch_chat')
        input_texts = []
        for messages in batch_messages:
            if isinstance(messages, str):
                messages = [
                    {
                        "role": "user",
                        "content": messages
                    }
                ]
            input_text = self.apply_chat_template(messages, force_prefill=force_prefill)
            input_texts.append(input_text)
            
        if "sampling_params" in kwargs:
            temp_generation_config = kwargs["sampling_params"]
        else:
            temp_generation_config = self.generation_config.clone()
            for k in kwargs:
                if k in self.generation_config.__annotations__.keys():
                    setattr(temp_generation_config, k, kwargs[k])
                    
        # logger.debug(f'input_texts: {input_texts}')

        outputs = self.model.generate(input_texts, temp_generation_config, use_tqdm=use_tqdm)
        generated_texts = [output.outputs[0].text for output in outputs]
        # print(f'input_text: {input_texts[0]}\ngenerated_text: {self.tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)}')

        return generated_texts