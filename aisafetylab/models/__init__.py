from .local_model import *
from .openai_model import OpenAIModel
from .vllm_model import VLLMModel

def load_model(model=None, tokenizer=None, model_name=None, generation_config=None, model_path=None, tokenizer_path=None, device=None, base_url=None, api_key=None, vllm_mode=False):
    if base_url is not None or api_key is not None:
        return OpenAIModel(model_name, base_url, api_key, generation_config)
    else:
        return LocalModel(model, tokenizer, model_name, generation_config, model_path, tokenizer_path, device, vllm_mode)
    