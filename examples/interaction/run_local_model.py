import json
from tqdm import tqdm
import numpy as np
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from loguru import logger

if __name__ == "__main__":
    model_path = "lmsys/vicuna-7b-v1.5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    generation_config = {
        "do_sample": False,
        "max_new_tokens": 512
    }
    model = load_model(model=model, tokenizer=tokenizer, model_name='Llama-3-8B-Instruct', generation_config=generation_config)
    
    query = "Hello?"
    response = model.chat(query, generation_config=generation_config)
    print(response)
    
    queries = ["What is the capital of France?", "What is the capital of Germany?"]
    responses = model.batch_chat(queries)
    print(responses)
    