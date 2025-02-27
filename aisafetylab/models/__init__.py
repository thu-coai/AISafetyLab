from .local_model import *
from .openai_model import OpenAIModel
from .vllm_model import VLLMModel

def load_model(model=None, tokenizer=None, model_name=None, generation_config=None, base_url=None, api_key=None, vllm_mode=False):
    if api_key is not None:
        return OpenAIModel(model_name, base_url, api_key, generation_config)
    elif vllm_mode:
        return VLLMModel(model, tokenizer, model_name, generation_config)
    else:
        return LocalModel(model, tokenizer, model_name, generation_config)