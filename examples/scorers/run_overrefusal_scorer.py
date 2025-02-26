from aisafetylab.evaluation.scorers.overrefusal_scorer import OverRefusalScorer

## for api-based models

model_name = 'gpt-4o-mini'
api_key = '<Your OpenAI api-key here>'

overrefusal_scorer = OverRefusalScorer(model_name=model_name, api_key=api_key)

## for local models

# model_path = 'meta-llama/Llama-3.1-8B-Instruct'
# device = 'cuda:0'
# model_name = 'llama-3'

# print('loading model...')
# if torch.cuda.is_available():
#     device = torch.device(device)
# else:
#     device = torch.device('cpu')
# tokenizer_path = model_path
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
# model = model.to(device).eval()
# model = LocalModel(model, tokenizer, model_name)
# print('finish loading')

# overrefusal_scorer = OverRefusalScorer(model=model)

query = 'Hello'
response = 'Sorry, but I can not assist it.'
res = overrefusal_scorer.score(query, response)
print(res['score'])
