from .base_scorer import BaseScorer

class PrefixMatchScorer(BaseScorer):
    def __init__(self, targets=[]):
        super().__init__()
        self.targets = targets
        
    def score(self, text, targets=None):
        if targets is None:
            targets = self.targets
        
        for target in targets:
            if text.startswith(target):
                return {'score': 1}
        
        return {'score': 0}
    
        
    