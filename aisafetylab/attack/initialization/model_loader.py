from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# from aisafetylab.utils import find_seperators
from types import SimpleNamespace
class ModelLoader:
    def init_model(self, model_path, template_path = None, find_seperators=False):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if template_path is not None:
            template_str = open(template_path).read().replace('    ', '').replace('\n', '')
            tokenizer.chat_template = template_str

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='cuda:1', trust_remote_code=True)
        if find_seperators:
            prefix_seperator, suffix_seperator, adv_token = find_seperators(tokenizer)

        embedding_matrix = model.get_input_embeddings().weight
        world = SimpleNamespace()
        world.model = model
        world.tokenizer = tokenizer
        world.embedding_matrix = embedding_matrix
        if find_seperators:
            # world.prefix_seperator_id = tokenizer.convert_tokens_to_ids([prefix_seperator])[0]
            # world.suffix_seperator_id = tokenizer.convert_tokens_to_ids([suffix_seperator])[0]
            world.prefix_seperator_id = tokenizer.encode([prefix_seperator], add_special_tokens = False)[0]
            world.suffix_seperator_id = tokenizer.encode([suffix_seperator], add_special_tokens = False)[0]
            world.prefix_seperator = prefix_seperator
            world.suffix_seperator = suffix_seperator
            world.adv_token = adv_token
        return world