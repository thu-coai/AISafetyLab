from aisafetylab.attack.mutation import BaseMutation

class Leetspeak(BaseMutation):
    def __init__(self):
        super().__init__()

    def mutate(self, text):
        leet_dict = {
            'a': '@',
            'e': '3',
            'i': '!',
            'o': '0',
            'are': 'r',
            'be': 'b'
        }
        def to_leetspeak(s):
            for key, val in leet_dict.items():
                s = s.replace(key, val)
            return s
        return to_leetspeak(text)