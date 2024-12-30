"""
Robust-Align Defense Method
============================================
This Class achieves a defense method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM
Arxiv link: https://arxiv.org/abs/2309.14348
Source repository: https://github.com/AAAAAAsuka/llm_defends
"""

from aisafetylab.defense.inference_defense.base_defender import IntraprocessDefender
from aisafetylab.models import LocalModel
from aisafetylab.evaluation.scorers import PatternScorer
from loguru import logger
import random

class RobustAlignDefender(IntraprocessDefender):
    """
    Defense method for alignment-breaking attacks based on robustness evaluation of the LLM response.

    This class uses a scoring mechanism to assess whether the model's response is safe and 
    performs Monte Carlo sampling to perturb the input tokens and evaluate the robustness 
    of the model’s response to such perturbations.

    Args:
        scorer (PatternScorer, optional): Scoring function to evaluate model response safety. Defaults to PatternScorer.
        drop_rate (float, optional): The rate of token deletion during the perturbation. Defaults to 0.3.
        sampling_num (int, optional): The number of Monte Carlo samples for robustness evaluation. Defaults to 20.
        threshold (float, optional): The threshold of false samples to classify a query as unsafe. Defaults to 0.2.
    """
    def __init__(self, scorer=PatternScorer(), drop_rate=0.3, sampling_num=10, threshold=0.2, verbose=False):
        self.scorer = scorer
        self.drop_rate = drop_rate
        self.sampling_num = sampling_num
        self.threshold = threshold
        self.verbose = verbose
        logger.info(f"RobustAlignDefender initialized with judge scorer: \n{type(self.scorer).__name__}")
        
    def delete_random_tokens(self, tokens):
        """
        Delete a random subset of tokens from the input based on a specified probability.

        Args:
            tokens (list): The list of input tokens to perturb.
            p (float): The probability of deleting each token.

        Returns:
            list: The perturbed list of tokens with some randomly deleted.
        """
        token_num = len(tokens)
        drop_count = int(token_num * self.drop_rate)
        drop_indices = random.sample(range(token_num), drop_count)
        return [token for idx, token in enumerate(tokens) if idx not in drop_indices]
    
    def robustness_check(self, model, messages):
        if not isinstance(model, LocalModel):
            logger.error(f"RobustAlignDefender only support Local Model for the need of tokenizer!")
            
        # Monte Carlo sampling
        prompt = model.apply_chat_template(messages)
        tokens = model.tokenizer(prompt, add_special_tokens=False).input_ids
        false_num = 0

        for sample in range(self.sampling_num):
            perturbed_tokens = self.delete_random_tokens(tokens)
            perturbed_text = model.tokenizer.decode(perturbed_tokens)
            messages[-1]["content"] = perturbed_text
            dropped_response = model.chat(messages)  # Using model.chat() to get the response
            dropped_response_score = self.scorer.score(dropped_response)['score']
            # logger.debug(f"Response: \n{dropped_response}")

            if dropped_response_score == 0:
                false_num += 1
                # logger.debug(f"false samples = {false_num}/{sample+1}")
            false_rate = false_num / self.sampling_num
            if false_rate >= self.threshold:
                if self.verbose:
                    logger.debug(f"### RA-LLM ###: query is not safe, false samples = {false_num}/{sample + 1}")
                return True

            # Check if remaining samples are unlikely to meet the threshold
            if self.sampling_num - sample - 1 + false_num < (self.threshold * self.sampling_num):
                if self.verbose:
                    logger.debug(f"### RA-LLM ###: query is safe, false samples = {false_num}/{sample + 1}")
                return False

        if self.verbose:
            logger.debug(f"### RA-LLM ###: query is safe, false rate: {false_rate}")
        return False

    def defend(self, model, messages):
        """
        Defend against a potentially unsafe query by evaluating the model's response and performing robustness checks.

        Args:
            model (LocalModel): The model used to generate responses.
            messages (str or list): The input messages to be passed into the model. If a string is provided, it is 
                                    wrapped in a list format.

        Returns:
            str: The model's response if safe, otherwise returns a predefined apology message indicating defense triggered.
        """
        sorry_response = "I'm sorry, but I cannot provide assistance or information on that topic."
        
        # Check original response state
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]
            
        response = model.chat(messages)
        response_score = self.scorer.score(response)

        if self.verbose:
            logger.debug(f"Query: \n{messages}")
            logger.debug(f"Response: \n{response}")

        if response_score == 0:
            # Original query is not safe!
            return sorry_response
        
        is_unsafe = self.robustness_check(model, messages)
        
        if is_unsafe:
            return sorry_response
        else:
            return response
        