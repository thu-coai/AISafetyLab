from .local_model import *
from .openai_model import OpenAIModel


def load_model(model=None, tokenizer=None, model_name=None, generation_config=None, base_url=None, api_key=None):
    if api_key is None:
        return LocalModel(model, tokenizer, model_name, generation_config)
    return OpenAIModel(model_name, base_url, api_key, generation_config)
