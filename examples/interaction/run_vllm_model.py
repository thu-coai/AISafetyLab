import json
from tqdm import tqdm
import numpy as np
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from vllm import LLM, SamplingParams
from loguru import logger

if __name__ == "__main__":
    # vllm only supports CUDA_VISIBLE_DEVICES for GPU allocation
    model_path = "lmsys/vicuna-7b-v1.5"
    model = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_config = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=4096)
    model = load_model(model=model, tokenizer=tokenizer, model_name='Llama-3-8B-Instruct', generation_config=generation_config, vllm_mode=True)
    
    query = "Hello?"
    response = model.chat(query, generation_config=generation_config)
    print(response)
    
    queries = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = model.batch_chat(queries)
    print(responses)
    