import json
from tqdm import tqdm
import numpy as np
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from aisafetylab.evaluation.scorers import OverRefusalScorer
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
    queries = [d['prompt'] for d in data]
    responses = batch_chat(model, queries, defenders, batch_size=batch_size)
    score_results = [scorer.score(q, r) for q, r in tqdm(zip(queries, responses))]
    all_scores = [r['score'] for r in score_results]
    for i in range(len(data)):
        data[i]['response'] = responses[i]
        data[i]["score"] = all_scores[i]

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
    scorer = OverRefusalScorer(
        model_name="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="YOUR_KEY",
    )
    defenders = []
    defender_path = "../defense/configs/self_reminder.yaml"
    defender_name = "self_reminder"
    defenders.append(create_defender_from_yaml(defender_path))

    # XSTEST dataset
    path = "thu-coai/AISafetyLab_Datasets/xstest_safe"
    save_path = f"evaluation_results/overrefusal/{defender_name}_vicuna-7b-v1.5_test.json"

    evaluate_asr(
        path,
        model,
        scorer,
        defenders,
        batch_size=8,
        save_path=save_path,
    )
