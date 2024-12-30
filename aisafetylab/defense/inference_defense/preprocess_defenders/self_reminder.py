"""
Self-Reminder Defense Method
============================================
This Class achieves a defense method describe in the paper below.

Paper title: Defending ChatGPT against jailbreak attack via self-reminders
Paper link: https://www.nature.com/articles/s42256-023-00765-8
Source repository: None
"""

from aisafetylab.defense.inference_defense.base_defender import PreprocessDefender
from loguru import logger
from aisafetylab.defense.inference_defense.defender_texts import SELF_REMINDER_DEFAULT_TEXT

class SelfReminderDefender(PreprocessDefender):
    """
    Defender that adds a safety reminder to the input prompt.
    """
    def __init__(self, reminder_text=SELF_REMINDER_DEFAULT_TEXT):
        """
        Initialize the SelfReminder Defender.

        Args:
            reminder_text (str): The safety reminder to be added.
        """
        if not isinstance(reminder_text, str):
            logger.error("Reminder text must be a string.")
            raise TypeError("Reminder text must be a string.")
        self.reminder_text = reminder_text
        logger.debug(f"SelfReminderDefender initialized with reminder text: \n{reminder_text}")

    def defend(self, messages):
        """
        Defend by adding a safety reminder to the input text.

        Args:
            messages (str / list): The original input messages.

        Returns:
            str: The defended input text with safety reminder.
        """
        if isinstance(messages, str):
            messages = [
                {
                    "role": "user",
                    "content": messages
                }
            ]
        
        messages[-1]["content"] = self.reminder_text.replace("{input_text}", messages[-1]["content"])
        return messages, False