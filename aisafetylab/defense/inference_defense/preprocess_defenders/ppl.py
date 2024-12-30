"""
PPL filter Method
============================================
This Class achieves a defense method describe in the paper below.

Paper title: Detecting language model attacks with perplexity
Paper link: https://openreview.net/pdf?id=lNLVvdHyAw
Source repository: None
"""
from aisafetylab.defense.inference_defense.base_defender import PreprocessDefender
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import LocalModel, OpenAIModel
import json

class PPLDefender(PreprocessDefender):
    """
    Defender that adds a safety reminder to the input prompt.
    """
    def __init__(self, model_name, model_path, threshold: float = 300.0, windows: int = 10):
        """
        Initialize the PPLDefender Defender.

        Args:
            threshold (float): PPL threshold, below which text will be filtered.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = self.load_model(model_name, model_path, self.device)
        self.threshold = threshold
        self.windows = windows
        logger.debug(f"PPLDefenderDefender initialized with threshold: {threshold}")

    def calculate_ppl(self, text):
        inputs = self.pipeline.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        num_tokens = input_ids.size(1)

        if num_tokens <= self.windows:
            outputs = self.pipeline.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
            ppl = torch.exp(torch.tensor(loss)).item()
        else:
            losses = []
            num_full_chunks = num_tokens // self.windows

            for i in range(num_full_chunks):
                chunk_ids = input_ids[:, i*self.windows : (i+1)*self.windows]
                outputs = self.pipeline.model(input_ids=chunk_ids, labels=chunk_ids)
                losses.append(outputs.loss.item())

            remaining = num_tokens % self.windows
            if remaining > 0:
                last_chunk_ids = input_ids[:, -self.windows:]
                outputs = self.pipeline.model(input_ids=last_chunk_ids, labels=last_chunk_ids)
                losses.append(outputs.loss.item())

            max_loss = max(losses)

            ppl = torch.exp(torch.tensor(max_loss)).item()

        return ppl
        
    def is_safe(self, text: str) -> bool:
        """
        Determine whether the text is filtered through PPL.

        Args:
            text (str): The input text.

        Returns:
            bool: Whether it is filtered or not.
        """

        ppl = self.calculate_ppl(text)
        
        return ppl < self.threshold
        
    def defend(self, messages):
        """
        Defend by adding a PPL filter to the input text.

        Args:
            messages (str / list): The original input messages.

        Returns:
            list: The original input messages.
            bool: The input message is harmful or not.
        """
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        if self.is_safe(messages[-1]["content"]):
            return messages, False

        else:
            return messages, True
    
    def load_model(self, model_name, model_path, device=None, api_key=None, base_url=None):
        if model_name == 'openai':
            return OpenAIModel(
                model_name=model_path,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            model=AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,).eval().to(device)
            tokenizer=AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            return LocalModel(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
            )