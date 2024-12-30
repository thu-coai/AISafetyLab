from aisafetylab.evaluation.scorers.shieldlm_scorer import ShieldLMScorer

model_path = "thu-coai/ShieldLM-14B-qwen"
model_base = 'qwen'
shieldlm_scorer = ShieldLMScorer(model_path=model_path, model_base=model_base)
print('finish loading model')

query = 'hello'
response = 'hello'
res = shieldlm_scorer.score(query, response)
print(res['score'])
print(res['output'])
