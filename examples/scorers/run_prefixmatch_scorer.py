from aisafetylab.evaluation.scorers.prefixmatch_scorer import PrefixMatchScorer

prefixmatch_scorer = PrefixMatchScorer(targets=['Sure', 'Here'])

query = 'Hello'
response = 'hello'
res = prefixmatch_scorer.score(query, response)
print(res['score'])
