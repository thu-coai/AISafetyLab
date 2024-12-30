"""
Erase-and-Check Defense Method
============================================
This Class achieves a defense method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: Certifying LLM Safety against Adversarial Prompting
Arxiv link: https://arxiv.org/pdf/2309.02705
Source repository: https://github.com/aounon/certified-llm-safety
"""

from aisafetylab.defense.inference_defense.base_defender import IntraprocessDefender
from aisafetylab.models import LocalModel
from aisafetylab.evaluation.scorers import PatternScorer
from aisafetylab.defense.inference_defense.defender_texts import SORRY_RESPONSE
from loguru import logger
import random

class EraseCheckDefender(IntraprocessDefender):
    """
    Implements the erase-and-check defense method by progressively truncating the user's input
    from the end and checking if any of the truncated inputs produce harmful responses.
    
    Mention, we now only support the first mode of the original paper: adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt.

    Args:
        scorer (PatternScorer): Scorer to evaluate if a response is harmful.
        max_erase (int): Maximum number of tokens to erase from the end of the user's input.
        randomized (bool): Whether to randomly sample truncated inputs.
        prompt_sampling_ratio (float): Ratio of truncated inputs to sample if randomized is True.
    """
    def __init__(self, scorer=PatternScorer(), max_erase=20, randomized=False, prompt_sampling_ratio=0.1, verbose=False):
        self.scorer = scorer
        self.max_erase = max_erase
        self.randomized = randomized
        self.prompt_sampling_ratio = prompt_sampling_ratio
        self.verbose = verbose
        logger.info(f"EraseCheckDefender initialized with scorer: {type(self.scorer).__name__}")

    def erase_and_check_suffix(self, model, messages):
        """
        Erase the user's input one token at a time from the end and check if any of the truncated inputs produce harmful responses.

        Args:
            model (LocalModel): The model used to generate responses.
            messages (list): The input messages to be passed into the model.

        Returns:
            bool: True if all truncated inputs are safe, False if any produce a harmful response.
        """
        user_prompt = messages[-1]["content"]
        tokens = model.tokenizer(user_prompt, add_special_tokens=False)['input_ids']
        prompt_length = len(tokens)

        prompt_list = [user_prompt]

        max_erase = min(self.max_erase, prompt_length)

        for i in range(1, max_erase + 1):
            # Erase i tokens from the end
            truncated_tokens = tokens[:-i]
            if not truncated_tokens:
                # No tokens left, skip
                continue
            truncated_prompt = model.tokenizer.decode(truncated_tokens)
            prompt_list.append(truncated_prompt)

        # If randomized, sample a subset of prompts
        if self.randomized:
            num_prompts_to_test = int(self.prompt_sampling_ratio * (len(prompt_list) - 1))
            if num_prompts_to_test > 0:
                prompts_to_test = [prompt_list[0]] + random.sample(prompt_list[1:], num_prompts_to_test)
            else:
                prompts_to_test = prompt_list
        else:
            prompts_to_test = prompt_list

        # Now, for each prompt, generate the response and check if it's harmful
        for idx, truncated_prompt in enumerate(prompts_to_test):
            # Prepare messages with truncated prompt
            truncated_messages = messages[:-1] + [{
                "role": messages[-1]["role"],
                "content": truncated_prompt
            }]
            response = model.chat(truncated_messages)
            response_score = self.scorer.score(query=messages[-1]["content"], response=response)['score']
            if self.verbose:
                logger.info(f"Truncated Prompt {idx}: {truncated_prompt}")
                logger.info(f"Response: {response}")
                logger.info(f"Score: {response_score}")
            if response_score == 0:
                # Response is harmful
                if self.verbose:
                    logger.info(f"Harmful response detected with truncated prompt.")
                return True
        # All responses are safe
        return False

    def defend(self, model, messages):
        """
        Defend against a potentially unsafe query by evaluating the model's response and performing the erase-and-check defense.

        Args:
            model (LocalModel): The model used to generate responses.
            messages (str or list): The input messages to be passed into the model. If a string is provided, it is 
                                    wrapped in a list format.

        Returns:
            str: The model's response if safe, otherwise returns a predefined apology message indicating defense triggered.
        """

        # Ensure messages is in the correct format
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]

        # Generate the original response
        response = model.chat(messages)
        response_score = self.scorer.score(query=messages[-1]['content'], response=response)

        if self.verbose:
            logger.debug(f"Original Query: {messages[-1]['content']}")
            logger.debug(f"Original Response: {response}")

        if response_score == 0:
            # Original response is harmful
            return SORRY_RESPONSE

        # Apply erase-and-check defense
        old_max_new_tokens = model.generation_config["max_new_tokens"]
        model.generation_config["max_new_tokens"] = 32
        is_unsafe = self.erase_and_check_suffix(model, messages)
        model.generation_config["max_new_tokens"] = old_max_new_tokens

        if is_unsafe:
            return SORRY_RESPONSE
        else:
            return response