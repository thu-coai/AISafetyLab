import base64
from aisafetylab.attack.mutation import BaseMutation

class Base64_raw(BaseMutation):
    def __init__(self):
        super().__init__()
        
    def mutate(self, text):
        # Base64-encode the input text
        encoded = base64.b64encode(text.encode()).decode()
        text = f"""{encoded}"""
        return text
