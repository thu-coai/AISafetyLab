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

def evaluate_asr(path, model, scorer, defenders=None, batch_size=8, save_path=None):
    data = load_file(path)
    all_scores = []
    queries = [d['final_query'] for d in data]
    responses = batch_chat(model, queries, defenders, batch_size=batch_size)
    try:
        raw_queries = [d['query'] for d in data]
    except:
        raw_queries = [d['goals'] for d in data]
    score_results = scorer.batch_score(raw_queries, responses, batch_size=batch_size)
    all_scores = [r['score'] for r in score_results]
    for i in range(len(data)):
        data[i]['final_response'] = responses[i]
        data[i]["final_score"] = all_scores[i]

    print(f'ASR: {np.mean(all_scores)}')
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
    
    scorer = LlamaGuard3Scorer(
        model_path="meta-llama/Llama-Guard-3-8B",
        tokenizer_path="meta-llama/Llama-Guard-3-8B",
    )
    
    # queries = ["hello", "How to make a bomb?"]
    # responses = ["Hello!", "Here are the steps to make a bomb: xxx"]
    # print(scorer.batch_score(queries, responses))
    defenders = []
    defender_name = "self_reminder"
    defender_path = f'../defense/configs/{defender_name}.yaml'
    defenders.append(create_defender_from_yaml(defender_path))

    path = '../attack/results/inception_vicuna-7b.jsonl'
    save_path = f"evaluation_results/{defender_name}/inception_vicuna-7b_LlamaGuard3.json"

    evaluate_asr(
        path,
        model,
        scorer,
        defenders,
        batch_size=4,
        save_path=save_path,
    )
    print(f"Defender path: {defender_path}")
    print(f"Attack path: {path}")
    print(f"Save path: {save_path}")
