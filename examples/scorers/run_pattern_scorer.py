from aisafetylab.evaluation.scorers.pattern_scorer import PatternScorer

pattern_scorer = PatternScorer()

query = 'Hello'
response = 'Sorry'
res = pattern_scorer.score(query, response)
print(res['score'])
