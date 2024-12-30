"""
Goal Prioritization Defense Method
============================================
This Class achieves a defense method describe in the paper below.

Paper title: Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization
Paper link: https://aclanthology.org/2024.acl-long.481/
Source repository: https://github.com/thu-coai/JailbreakDefense_GoalPriority
"""

from aisafetylab.defense.inference_defense.base_defender import PreprocessDefender
from loguru import logger
from aisafetylab.defense.inference_defense.defender_texts import (
    GOAL_PRIORITIZATION_DEFAULT_TEXT,
)


class GoalPrioritizationDefender(PreprocessDefender):

    def __init__(self, defend_text=GOAL_PRIORITIZATION_DEFAULT_TEXT):
        """

        Args:
            defend_text (str): The defend text to be added.
        """
        if not isinstance(defend_text, str):
            logger.error("Defend text must be a string.")
            raise TypeError("Defend text must be a string.")
        self.defend_text = defend_text
        logger.debug(
            f"GoalPrioritizationDefender initialized with defend text: \n{defend_text}"
        )

    def defend(self, messages):
        """
        Defend by adding a prompt emphasizing the goal priority to the input text.

        Args:
            messages (str / list): The original input messages.

        Returns:
            str: The defended input text.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        messages[-1]["content"] = self.defend_text.replace(
            "{input_text}", messages[-1]["content"]
        )
        return messages, False
