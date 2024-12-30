import numpy as np
from abc import ABC, abstractmethod
from aisafetylab.dataset import AttackDataset
from aisafetylab.attack.selection.base_selection import BaseSelection



class MctsSelection(BaseSelection):
    def __init__(self, dataset, inital_prompt_pool, questions,ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__()
        self.Datasets = dataset
        self.inital_prompt_pool = inital_prompt_pool
        self.questions = questions
        self.step = 0
        self.mctc_select_path = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio 
        self.alpha = alpha
        self.beta = beta


    def select(self) -> AttackDataset:
        self.step += 1
        if len(self.Datasets) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.Datasets) - len(self.rewards))])
        self.mctc_select_path = []

        cur = max(
            self.inital_prompt_pool,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        
        self.mctc_select_path.append(cur)

        while len(cur.children) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.children,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return AttackDataset([cur])

    def update(self, prompt_nodes: AttackDataset):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        last_choice_node = self.Datasets[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.questions)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * max(self.beta, (1 - 0.1 * last_choice_node.level))