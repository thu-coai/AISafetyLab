"""
PPL filter Method
============================================
This Class use the model below.

Model link: https://huggingface.co/meta-llama/Prompt-Guard-86M
Source repository: https://huggingface.co/meta-llama/Prompt-Guard-86M
"""
import torch
from aisafetylab.defense.inference_defense.base_defender import PreprocessDefender
from loguru import logger

class PromptGuardDefender(PreprocessDefender):
    """
    Defender that adds a safety reminder to the input prompt.
    """
    def __init__(self, model, tokenizer):
        """
        Initialize the SelfReminder Defender.

        Args:
            reminder_text (str): The safety reminder to be added.
        """
        self.model = model
        self.tokenizer = tokenizer
        
    def guard(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        if predicted_class_id == "BENIGN":
            return False
        else:
            return True
        
    def defend(self, messages):
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]
        for message in messages:
            if message["role"] == "user":
                if self.guard(message["content"]):
                    return messages, True
        return messages, False