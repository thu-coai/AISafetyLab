import json
from tqdm import tqdm
import numpy as np
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from loguru import logger

if __name__ == "__main__":
    # vllm only supports CUDA_VISIBLE_DEVICES for GPU allocation
    model_name = "gpt-3.5-turbo"
    base_url = None # replace with request url
    api_key = None # replace with request api key
    model = load_model(model_name = model_name, base_url=base_url, api_key=api_key)
    
    query = "Hello?"
    response = model.chat(query)
    print(response)
    
    queries = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = model.batch_chat(queries)
    print(responses)
    