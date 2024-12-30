"""
Jailbroken Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.

Paper title: Jailbroken: How Does LLM Safety Training Fail?
arXiv link: https://arxiv.org/pdf/2307.02483.pdf
Source repository: None
"""

from tqdm import tqdm
from aisafetylab.attack.initialization import ModelLoader
from aisafetylab.evaluation.scorers import PatternScorer
from aisafetylab.attack.mutation import *
from aisafetylab.attack.attackers import BaseAttackManager
from aisafetylab.dataset import AttackDataset
from dataclasses import dataclass
from aisafetylab.logging import setup_logger
from aisafetylab.models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@dataclass
class AttackConfig:
    data_path: str
    system_prompt: str
    attack_model_path: str
    attack_model_name: str
    model_path: str
    model_name: str
    mutation_method: str='combination_3'
    attack_model_device: str='cuda:0'
    model_device: str='cuda:1'

@dataclass
class AttackData:
    attack_dataset: str

class JailbrokenEvaluator:
    def __init__(self):
        self.scorer = PatternScorer()
    
    def evaluate(self, query, response):
        
        is_success = self.scorer.score(query, response)['score']
        return is_success

class JailbrokenInit:
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
        attack_model = AutoModelForCausalLM.from_pretrained(config.attack_model_path).eval().to(config.attack_model_device)
        attack_tokenizer = AutoTokenizer.from_pretrained(config.attack_model_path)
        if attack_tokenizer.pad_token is None:
            attack_tokenizer.pad_token = attack_tokenizer.eos_token
            attack_tokenizer.pad_token_id = attack_tokenizer.eos_token_id
        self.attack_model = load_model(model = attack_model, tokenizer = attack_tokenizer, model_name = config.attack_model_name)
class JailbrokenMutator:
    def __init__(self, config, init):
        self.config = config
        self.mutaitions = {
            'artificial':Artificial(),
            'base64':Base64(),
            'base64_input_only':Base64_input_only(),
            'base64_raw':Base64_raw(),
            'combination_1':Combination_1(),
            'combination_2':Combination_2(),
            'combination_3':Combination_3(),
            'disemvowel': Disemvowel(),
            'leetspeak': Leetspeak(),
            'rot13':Rot13(),
            'auto_payload_splitting':Auto_payload_splitting(init.attack_model),
            'auto_obfuscation':Auto_obfuscation(init.attack_model),
        }
        self.mutation = self.mutaitions[self.config.mutation_method]

    def mutate(self, query):
        mutated_queries = self.mutation.mutate(query)
        if isinstance(mutated_queries, str):
            mutated_queries = [mutated_queries]
        return mutated_queries

class JailbrokenManager(BaseAttackManager):

    def __init__(
        self,
        data_path="thu-coai/AISafetyLab_Datasets/harmbench_standard",
        system_prompt="",
        attack_model_path="lmsys/vicuna-7b-v1.5",
        attack_model_name="vicuna_1.5",
        model_path="lmsys/vicuna-7b-v1.5",
        model_name="vicuna_1.5",
        mutation_method="combination_3",
        attack_model_device="cuda:0",
        model_device="cuda:1",
        res_save_path="results/jailbroken.jsonl",
        *args,
        **kwargs
    ):
        """
        A class used to manage and execute jailbreaking attacks on a target model.

        Parameters
        ----------
        data_path: str
            The path to load the jailbreaking prompts (default is 'thu-coai/AISafetyLab_Datasets/harmbench_standard').
        system_prompt: str
            The system prompt to the target_model (default is an empty string).
        attack_model_path: str
            The path to the model used for the attack (default is 'lmsys/vicuna-7b-v1.5').
        attack_model_name: str
            The name of the attack model (default is 'vicuna_1.5').
        model_path: str
            The path to the target model to attack (default is 'lmsys/vicuna-7b-v1.5').
        model_name: str
            The name of the target model (default is 'vicuna_1.5').
        mutation_method: str
            The method used for generating mutations in the jailbreaking attack (default is 'combination_3').
        attack_model_device: str
            The device on which the attack model is loaded (default is 'cuda:0').
        model_device: str
            The device on which the target model is loaded (default is 'cuda:1').
        res_save_path: str
            The path to save the results of the jailbreaking attack (default is 'results/jailbroken.jsonl').

        Methods
        -------
        mutate(): Mutate a single input string.
        attack(): Attack with defenders.
        """

        super().__init__(res_save_path)
        self.config = AttackConfig(data_path, system_prompt, attack_model_path, attack_model_name, model_path, model_name, mutation_method, attack_model_device, model_device)
        self.init = JailbrokenInit(self.config)
        self.data = self.init.data.attack_dataset
        self.evaluator = JailbrokenEvaluator()
        self.attack_model = self.init.attack_model
        self.target_model = self.init.target_model
        self.mutator = JailbrokenMutator(self.config, self.init)

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
