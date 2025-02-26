from aisafetylab.defense.inference_defense.base_defender import IntraprocessDefender
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors import safe_open
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from typing import Union
import os, json
from loguru import logger

class DRODefender(IntraprocessDefender):
    """
    Defender that adds a safety reminder to the input prompt.
    """
    def __init__(self, model, tokenizer, model_name, system_prompt_type="default"):
        self.model = model
        self.model_name = model_name
        self.system_prompt_type = system_prompt_type
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(script_dir, f'dro/trained_prompts/{self.model_name}/type.all_length.{self.system_prompt_type}.safetensors')
        if tokenizer.chat_template is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            tokenizer.chat_template = open(f'{script_dir}/../../../../datasets/chat_templates/vicuna.jinja').read().replace('    ', '').replace('\n', '')
        with safe_open(prompt_path, framework='pt') as f:
            self.soft_prompt = f.get_tensor('soft_prompt')
        self.toker, self.new_input_embeddings = self.process_soft_prompt_as_word_embedding(model, tokenizer, self.soft_prompt)
        logger.info(f"DRODefender initialized")

    def prepend_sys_prompt(self, messages, prompt_length):
        messages = [{'role': 'system', 'content': ''.join([f'<soft_prompt_{i}>' for i in range(prompt_length)])}] + messages
        return messages
    
    def process_soft_prompt_as_word_embedding(
        self,
        model: PreTrainedModel,
        toker: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        soft_prompt: torch.nn.Parameter
    ) -> nn.Module:
        # We embed soft prompt into input word embedding and safe it
        # When loaded later, simply call model.set_input_embeddings()
        config = model.config
        padding_idx = config.pad_token_id

        old_toker_size = len(toker)
        toker.add_tokens([f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))], special_tokens=True)
        new_toker_size = len(toker)

        old_input_embeddings = model.get_input_embeddings()
        embedding_dim = old_input_embeddings.embedding_dim
        old_num_embeddings = old_input_embeddings.num_embeddings
        new_num_embeddings = max(new_toker_size, old_num_embeddings)

        new_input_embeddings = nn.Embedding(new_num_embeddings, embedding_dim, padding_idx)
        new_input_embeddings.weight.data[:old_toker_size] = old_input_embeddings.weight.data[:old_toker_size]
        new_input_embeddings.weight.data[old_toker_size:new_toker_size] = soft_prompt.data.to('cpu')
        return toker, new_input_embeddings
    

    def defend(self, model, messages):
        model = model.model
        generation_config = model.generation_config
        model.set_input_embeddings(self.new_input_embeddings.to(device=model.device, dtype=model.dtype))
        
        if isinstance(messages, str):
            messages = [{
                'role': 'user',
                'content': messages
                }
            ]
            
        messages = self.prepend_sys_prompt(messages, self.soft_prompt.size(0))
        input_text = self.toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = torch.tensor(
            self.toker.convert_tokens_to_ids(self.toker.tokenize(input_text)),
            dtype=torch.long,
        ).unsqueeze(0).to(model.device)
        # Tokenize the single input message

        # Generate outputs using the model
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=1.0,
            do_sample=False
        )

        # Decode the generated output IDs to text
        generated_texts = self.toker.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)

        # Return the generated texts
        return generated_texts