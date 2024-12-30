
class BaseScorer:
    def __init__(self):
        pass

    def score(self, query: str, response: str):
        raise NotImplementedError