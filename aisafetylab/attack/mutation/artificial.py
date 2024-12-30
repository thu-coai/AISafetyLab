import os
import json
from aisafetylab.attack.mutation import BaseMutation
from aisafetylab.attack.initialization import InitTemplates
class Artificial(BaseMutation):
    def __init__(self):
        super().__init__()
    def load_seeds(self):
        seeds = InitTemplates().get_templates('Jailbroken', -1)
        return seeds
    def mutate(self, text):
        seeds = self.load_seeds()
        queries = []
        for seed in seeds:
            for value in seed.values():
                queries.append(value.format(query = text))
        return queries