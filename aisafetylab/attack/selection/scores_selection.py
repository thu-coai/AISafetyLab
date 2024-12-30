r"""
'ScoresSelectio', select those examples whose scores are high(scores are on the extent of jailbreaking),
detail information can be found in the following paper.

Paper title: Tree of Attacks: Jailbreaking Black-Box LLMs Automatically
arXiv link: https://arxiv.org/abs/2312.02119
Source repository: https://github.com/RICommunity/TAP
"""
import numpy as np
from typing import List

from .base_selection import BaseSelection
from aisafetylab.dataset import AttackDataset

class ScoresSelection(BaseSelection):
    def __init__(self, Dataset: AttackDataset, tree_width):
        r"""
        Initialize the selector with a dataset and a tree width.

        :param ~AttackDataset Dataset: The dataset from which examples are to be selected.
        :param int tree_width: The maximum number of examples to select.
        """
        super().__init__()
        self.tree_width = tree_width
        
    def select(self, dataset:AttackDataset) -> AttackDataset:
        r"""
        Selects a subset of examples from the dataset based on their scores.

        :param ~AttackDataset dataset: The dataset from which examples are to be selected.
        :return List[example]: A list of selected examples with high evaluation scores.
        """
        list_dataset = [example for example in dataset.data]
        # Ensures that elements with the same score are randomly permuted
        np.random.shuffle(list_dataset)
        list_dataset.sort(key=lambda x:x.eval_results[-1],reverse=True)

        # truncate/select based on judge_scores/example.eval_results[-1]
        width = min(self.tree_width, len(list_dataset))
        truncated_list = [list_dataset[i] for i in range(width) if list_dataset[i].eval_results[-1] > 0]
        # Ensure that the truncated list has at least two elements
        if len(truncated_list) == 0:
            truncated_list = [list_dataset[0],list_dataset[1]]

        return AttackDataset(truncated_list)