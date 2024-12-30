from aisafetylab.attack.mutation import BaseMutation

class Disemvowel(BaseMutation):
    def __init__(self):
        super().__init__()

    def mutate(self, text):
        text = ''.join([char for char in text if char not in 'aeiouAEIOU'])
        return text