from aisafetylab.evaluation.scorers.classification_scorer import ClassficationScorer

model_path = 'hubert233/GPTFuzz'
classfication_scorer = ClassficationScorer()
classfication_scorer.set_model(model_path=model_path)

print('finish loading model')

query = 'Hello'
response = 'hello'
res = classfication_scorer.score(query, response)
print(res['score'])
