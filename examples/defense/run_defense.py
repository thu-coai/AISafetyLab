import json
from tqdm import tqdm
import numpy as np
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from aisafetylab.evaluation.scorers import LlamaGuard3Scorer
from aisafetylab.models import LocalModel
from aisafetylab.defense.inference_defense import create_defender_from_yaml
from aisafetylab.defense.inference_defense import batch_chat

def load_file(path):
    if path.endswith('.jsonl'):
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    
    raise ValueError(f'Unsupported file format: {path}')

def defend_chat(path, model, defenders=None, batch_size=8, save_path=None):
    data = load_file(path)
    all_scores = []
    queries = [d['final_query'] for d in data]
    responses = batch_chat(model, queries, defenders, batch_size=batch_size)
    try:
        raw_queries = [d['query'] for d in data]
    except:
        raw_queries = [d['goals'] for d in data]
    for i in range(len(data)):
        data[i]['final_response'] = responses[i]
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
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
    model = load_model(model=model, tokenizer=tokenizer, model_name='vicuna-7b-v1.5', generation_config=generation_config)
    
    defenders = []
    
    # defender_name="aligner"
    # defender_name="dro"
    # defender_name="erase_and_check"
    # defender_name="goal_prioritization"
    # defender_name="icd"
    # defender_name="paraphrase"
    # defender_name="parden"
    # defender_name="prompt_guard"
    # defender_name="robust_aligned"
    # defender_name="safedecoding"
    # defender_name="self_eval"
    # defender_name="smoothllm"
    defender_name="self_reminder"
    
    defender_path = f"../defense/configs/{defender_name}.yaml"
    defenders.append(create_defender_from_yaml(defender_path))

    # Define the attack type
    attack_type = "gcg"
    path = '../attack/results/gcg_harmbench_50_vicuna-7b-v1.5.jsonl'
    save_path = f"./defend_results/{defender_name}/gcg_vicuna-7b_LlamaGuard3.json"

    defend_chat(
        path,
        model,
        defenders,
        batch_size=8,
        save_path=save_path,
    )
