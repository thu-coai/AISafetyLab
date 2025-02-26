from .base_scorer import BaseScorer

class PrefixMatchScorer(BaseScorer):
    def __init__(self, targets=[]):
        super().__init__()
        self.targets = targets
        
    def score(self, query=None, response: str = "", targets=None):
        if targets is None:
            targets = self.targets
        
        for target in targets:
            if response.startswith(target):
                return {'score': 1}
        
        return {'score': 0}
    
        
    