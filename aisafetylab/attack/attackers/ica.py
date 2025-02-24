"""
In-Context-Learning Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.

Paper title: Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations
arXiv link: https://arxiv.org/pdf/2310.06387
Source repository: None
"""

import random
import torch
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass
from typing import Optional
from aisafetylab.dataset import AttackDataset
from aisafetylab.attack.attackers import BaseAttackManager
from aisafetylab.evaluation.scorers import PatternScorer, HarmBenchScorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from aisafetylab.models import load_model, LocalModel
from aisafetylab.logging import setup_logger

import os
import json

@dataclass
class AttackConfig:

    attack_data_path: str
    target_model_path: str
    target_model_name: str
    demo_num: int
    lang: str
    system_prompt: str
    device: str
    res_save_path: Optional[str] = None
    log_path: Optional[str] = None
    template_path: Optional[str] = None
    


@dataclass
class AttackData:
    attack_dataset: str
    lang: str

class ICAInit:

    def __init__(self, config) -> None:
        
        logger.info("Config: ", config)

        self.config = config

        attack_dataset = AttackDataset(self.config.attack_data_path)
        self.data  = AttackData(attack_dataset=attack_dataset, lang=self.config.lang)

        self.device = torch.device(self.config.device)
        
        target_model = AutoModelForCausalLM.from_pretrained(self.config.target_model_path).eval().to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.target_model_path)
        self.target_model = load_model(model=target_model, tokenizer=tokenizer, model_name=self.config.target_model_name)
        self.tokenizer = self.target_model.tokenizer

        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./data/icl/")
        print("full_path: ", full_path)
        if self.config.lang == 'en':
            with open(f"{full_path}/demo.json", "r", encoding='utf-8') as f:
                self.demos = json.load(f)
        elif self.config.lang == 'zh':
            with open(f"{full_path}/data_zh.json", "r", encoding='utf-8') as f:
                self.demos = json.load(f)    

    def get_demos(self):

        samples_ = random.sample(self.demos, self.config.demo_num)
        samples = []
        for item in samples_:
            samples.append(item)

        return samples

    
class ICAFeedback:
    def __init__(self):
        self.scorer = PatternScorer()
    
    def evaluate(self, query, response):
        
        is_success = self.scorer.score(query, response)
        return is_success



class ICAEvaluator:
    def __init__(self, evaluator_type, evaluator_path):

        self.evaluator_type = evaluator_type
        if evaluator_type == "pattern":
            self.scorer = PatternScorer()
        else:
            self.scorer = HarmBenchScorer(model_path=evaluator_path, tokenizer_path=evaluator_path)
    
    def evaluate(self, query, response):
        
        is_success = self.scorer.score(query, response)['score']
        return is_success

        
class ICAManager(BaseAttackManager):
    """
    A class used to manage and execute ICA attacks.

    Parameters
    ----------

    attack_data_path: str
        The path to load the jailbreaking prompts
    target_model_path: str
        The path to load the targe model to attack 
    target_model_name: str
        The name of the targe model to attack (default is 'vicuna7bv1.5')
    demo_num: int
        The number of demonstrations to attack (default is 1)
    lang: str
        The language type of jailbreaking prompts (default is 'en')
    evaluator_type: str
        The evaluator model used for jailbreaking evaluation (default is 'pattern')
    evaluator_path: str
        The path to load the evaluator model
    system_prompt: str
        The system prompt for attack
    res_save_path: str
        The path to save the results of ICA-Attack
    log_path: str
        The path to save the logs of ICA-Attack
    template_path: str
        The path to load the template prompts
    
    Methods:
        mutate(): Mutate a single input string.
        attack(): Attack with defenders.
    """

    def __init__(
        self,        
        attack_data_path,
        target_model_path,
        target_model_name,
        demo_num=1,
        lang='en',
        evaluator_type='pattern',
        evaluator_path='cais/HarmBench-Llama-2-13b-cls',
        system_prompt='',
        device='cuda:0',
        res_save_path=f'../results/ica.jsonl',
        log_path=f'../logs/',
        template_path=""):
        super().__init__(res_save_path)

        self.config = AttackConfig(attack_data_path, target_model_path, target_model_name, demo_num,  lang,  system_prompt,  device, res_save_path, log_path, template_path)

        self.init = ICAInit(self.config)
        # self.feedbacker = ICAFeedback()
        self.evaluator = ICAEvaluator(evaluator_type, evaluator_path)
        self.data = self.init.data.attack_dataset

        self.target_model = self.init.target_model
        self.tokenizer = self.init.tokenizer
        self.target_model_name = target_model_name
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.sys_prompt = self.config.system_prompt
        self.device = torch.device(self.config.device)

        # self.local_model = LocalModel(self.target_model, self.tokenizer, self.model_name)
        

        self.log_path = log_path
        setup_logger(log_file_path=self.log_path)
    
    def format_msgs(self, demos, raw_query):

        prompt = ""
        for item in demos:
            q,a = item["query"], item["unsafe_response"]
            prompt += f"User:{q}\n\nAssistant:{a}\n\n"
        prompt += f"User:{raw_query}"

        return prompt
    
    def get_input_ids(self, raw_query):
        
        demos = self.init.get_demos()
        prompt = self.format_msgs(demos, raw_query)
        
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(self.device)

        return input_ids, prompt

    def mutate(self, query):

        input_ids, input_prompt = self.get_input_ids(query)
        return input_prompt


    def attack(self):
        
        for example_idx, example in enumerate(tqdm(self.data, desc="Attacking each example")):

            # 1. Collect some other harmful prompts {x1, x2, · · · , xk} (may be irrelevant to x; can be reused)
            # 2. Collect the corresponding harmful response {y1, y2, · · · , yk} of each xi(i = 1, 2, · · · , k)
            # 3. Gather the demonstrations {(xi, yi)} and x as the adversarial prompt
            # 4. format data
            raw_query =  example.query
            input_ids, input_prompt = self.get_input_ids(raw_query)

            # 5. get outputs
            output_ids = self.target_model.generate(input_ids, batch=False)
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # 6. Evaluate jailbreak success or not
            is_success = self.evaluator.evaluate(example.query, response)
            
            save_data = {'example_idx': example_idx, 'query': raw_query, 'final_query': input_prompt, 'response': response, 'success': True, 'attack_success': is_success}
            self.log(save_data, save=True)
