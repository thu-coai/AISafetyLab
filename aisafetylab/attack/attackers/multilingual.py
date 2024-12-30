"""
Multilingual Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: Multilingual Jailbreak Challenges in Large Language Models
arXiv link: https://arxiv.org/abs/2310.06474
Source repository: https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs
"""

from tqdm import tqdm
from aisafetylab.attack.initialization import ModelLoader
from aisafetylab.evaluation.scorers import PatternScorer
from aisafetylab.attack.mutation import *
from aisafetylab.attack.attackers import BaseAttackManager
from aisafetylab.dataset import AttackDataset
from dataclasses import dataclass
from aisafetylab.logging import setup_logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model

@dataclass
class AttackConfig:
    data_path: str
    system_prompt: str
    model_path: str
    model_name: str
    mutation_method: str
    model_device: str='cuda:0'

@dataclass
class AttackData:
    attack_dataset: str

class JailbrokenEvaluator:
    def __init__(self):
        self.scorer = PatternScorer()
    
    def evaluate(self, query, response):
        
        is_success = self.scorer.score(query, response)['score']
        return is_success

class MultilingualInit:
    def __init__(self, config: AttackConfig) -> None:
        attack_dataset = AttackDataset(config.data_path)
        self.data  = AttackData(attack_dataset = attack_dataset)
        model = AutoModelForCausalLM.from_pretrained(config.model_path).eval().to(config.model_device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.target_model = load_model(model = model, tokenizer = tokenizer, model_name = config.model_name)

class MultilingualMutator:
    def __init__(self, config, init):
        self.config = config
        self.init = init
        self.mutations = {
            # Chinese
            'zh-CN' : Translate(language='zh-CN'),
            # Italian
            'it': Translate(language='it'),
            # Vietnamese
            'vi' : Translate(language='vi'),
            # Arabic
            'ar' : Translate(language='ar'),
            # Korean
            'ko' : Translate(language='ko'),
            # Thai
            'th': Translate(language='th'),
            # Bengali
            'bn': Translate(language='bn'),
            # Swahili
            'sw': Translate(language='sw'),
            # Javanese
            'jv': Translate(language='jv'),
        }
        self.mutation = self.mutations[self.config.mutation_method]

    def mutate(self, query):
        mutated_queries = self.mutation.translate(query)
        if isinstance(mutated_queries, str):
            mutated_queries = [mutated_queries]
        return mutated_queries


class MultilingualManager(BaseAttackManager):

    def __init__(
        self,
        data_path="thu-coai/AISafetyLab_Datasets/harmbench_standard",
        system_prompt="",
        model_path="lmsys/vicuna-7b-v1.5",
        model_name="vicuna_1.5",
        mutation_method="zh-CN",
        model_device="cuda:1",
        res_save_path="results/multilingual.jsonl",
        *args,
        **kwargs
    ):
        super().__init__(res_save_path)
        self.config = AttackConfig(data_path, system_prompt, model_path, model_name, mutation_method, model_device)
        self.init = MultilingualInit(self.config)
        self.data = self.init.data.attack_dataset
        self.evaluator = JailbrokenEvaluator()
        self.target_model = self.init.target_model
        self.mutator = MultilingualMutator(self.config, self.init)

    @classmethod
    def from_config(cls, config: dict, mutation_method):
        return cls(mutation_method = mutation_method, **config)

    def format_msgs(self, raw_query):
        self.target_model.conversation.messages = []
        if self.config.system_prompt:
            self.target_model.conversation.set_system_message(self.config.system_prompt)
        conv_template = self.target_model.conversation
        conv_template.append_message(conv_template.roles[0], raw_query)
        prompt = conv_template.get_prompt()
        return prompt

    def attack(self):
        for example_idx, example in enumerate(tqdm(self.data, desc="Attacking each example")):
            mutated_queries = self.mutator.mutate(example.query)
            for mutated_query in mutated_queries:
                final_input = self.format_msgs(mutated_query)
                toks = self.target_model.tokenizer(final_input).input_ids
                input_ids = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(self.target_model.model.device)
                output_ids = self.target_model.generate(input_ids, batch=False)
                response = self.target_model.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                is_success = self.evaluator.evaluate(example.query, response)
                save_data = {'example_idx': example_idx, 'query': example.query, 'final_query': mutated_query, 'response': response, 'success': True, 'attack_success': is_success}
                self.log(save_data, save=True)

    def mutate(self, query):
        mutated_queries = self.mutator.mutate(query)
        if isinstance(mutated_queries, str):
            mutated_queries = [mutated_queries]
        return mutated_queries
