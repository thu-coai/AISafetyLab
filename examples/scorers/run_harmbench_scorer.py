from aisafetylab.evaluation.scorers.harmbench_scorer import HarmBenchScorer

model_path = "cais/HarmBench-Llama-2-13b-cls"
harmbench_scorer = HarmBenchScorer(model_path=model_path)
print('finish loading model')

query = 'Hello'
response = 'Hello'
res = harmbench_scorer.score(query, response)
print(f'Query: {query}\nResponse: {response}\nScore: {res["score"]}\n')
# print(res['score'])
