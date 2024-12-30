import json
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model
from aisafetylab.evaluation.scorers import LlamaGuard3Scorer
from aisafetylab.models import LocalModel

def load_file(path):
    if path.endswith('.jsonl'):
        with open(path) as f:
            return [json.loads(line) for line in f]
    
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    
    raise ValueError(f'Unsupported file format: {path}')

def evaluate_asr(path, model, scorer, batch_size=8, save_path=None):
    data = load_file(path)
    all_scores = []
    # for d in tqdm(data):
    #     query = d['final_query']
    #     response = model.chat(query)
    #     d['final_response'] = response
    #     score_dict = scorer.score(d['query'], response)
    #     score = score_dict['score'] # 0 for safe, 1 for unsafe
    #     d['final_score'] = score
    #     all_scores.append(score)
    queries = [d['final_query'] for d in data]
    responses = model.batch_chat(queries, batch_size=batch_size)
    # responses = [d["response"] for d in data]
    raw_queries = [d['query'] for d in data]
    score_results = scorer.batch_score(raw_queries, responses, batch_size=batch_size)
    all_scores = [r['score'] for r in score_results]
    for i in range(len(data)):
        data[i]['final_response'] = responses[i]
        data[i]["final_score"] = all_scores[i]

    print(f'ASR: {np.mean(all_scores)}')
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

    # path = '../attack/results/autodan_vicuna-7b.jsonl'
    path = '../attack/results/advprompter_harmbench_50_vicuna-7b-v1.5.jsonl'
    # path = '../logs/vicuna7b0v1.5_Harmbench_10shot_patternScore.jsonl'
    evaluate_asr(
        path,
        model,
        scorer,
        batch_size=8,
        # save_path="evaluation_results/ica_vicuna-7b_10shot_LlamaGuard3.json",
        save_path="evaluation_results/advprompter_vicuna-7b_LlamaGuard3.json",
    )
