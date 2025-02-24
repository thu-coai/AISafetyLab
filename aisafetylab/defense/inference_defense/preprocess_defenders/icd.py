"""
In Context Learning Defense Method
============================================
This Class achieves a defense method describe in the paper below.

Paper title: Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations
arXiv link: https://arxiv.org/pdf/2310.06387
Source repository: None
"""

from aisafetylab.defense.inference_defense.base_defender import PreprocessDefender
from aisafetylab.models.base_model import Model
from aisafetylab.defense.inference_defense.defender_texts import SELF_EXAM_DEFAULT_TEXT
from loguru import logger

import random, os, json

class ICDefender(PreprocessDefender):
    """
    Defense method via a set of safe demonstrations to guard the model not to generate anything
 harmful (In-Context Defense, ICD).

    Args:
        num_of_demos (int, required): the number of safe demonstrations for guarding
        lang (str, optional): the language of input message
    """

    def __init__(self, num_of_demos=5, lang="en", demo_path="icl"):
        self.num_of_demos = num_of_demos
        self.lang = lang
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.demo_path = os.path.join(script_dir, demo_path)

    def load_demos(self):
        """
        Load the given demonstrations candidates.

        Returns:
            list: The list of given demos.
        """

        
        if self.lang == 'en':
            with open(f"{self.demo_path}/defense_demo.json", "r", encoding='utf-8') as f:
                demos_ = json.load(f)
        elif self.lang == 'zh':
            with open(f"{self.demo_path}/defense_demo_zh.json", "r", encoding='utf-8') as f:
                demos_ = json.load(f)
        demos = [(item["query"], item["refusal_response"]) for item in demos_]
        return demos

    def get_demos(self, demos):
        """
        Get specific demonstrations for each input

        Args:
            demos (list): All the given demo candidates.

        Returns:
            list: The selected demos for defend.
        """

        samples_ = random.sample(demos, self.num_of_demos)
        samples = []
        for item in samples_:
            samples.append(item)
        return samples
    
    def defend(self, messages):
        """
        Defend by adding demonstrations of refusing harmful questions.

        Args:
            messages (str): The original input messages.

        Returns:
            list: The defended messages with demonstrations in a multi-turn dialogue manner.
            bool: The input message is harmful or not
        """
        
        # 1. Collect some harmful requests {x1,x2,··· ,xk} (can be reused)
        # 2. Collect their corresponding safe responses {y1,y2,··· ,yk} of each xi (i = 1,2,··· ,k)
        demos = self.load_demos()

        # 3. Gather the safe demonstrations {(xi,yi)} and x as the safe prompt with the requests and responses Psafe = [x1,y1,··· ,x_k,y_k,x]
        samples = self.get_demos(demos)

        # 4. format the input messages
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        prompt = ""
        for (query, resp) in samples:
            prompt += f"User:{query}\n\nAssistant:{resp}\n\n"
        prompt += f"User:{messages[-1]['content']}"
        messages[-1]["content"] = prompt
        
        logger.info(f"ICD prompt: {prompt}")
        
        return messages, False
