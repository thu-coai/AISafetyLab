"""
DRA Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction
arXiv link: https://arxiv.org/abs/2402.18104
Source repository: https://github.com/LLM-DRA/DRA
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Any
from aisafetylab.attack.attackers import BaseAttackManager
from aisafetylab.attack.initialization import InitTemplates, PopulationInitializer
from aisafetylab.attack.mutation.dra_mutation import *
from aisafetylab.evaluation.scorers import PatternScorer, HarmBenchScorer, LlamaGuard3Scorer
from aisafetylab.models import load_model
import torch
from aisafetylab.utils import Timer
import random
from detoxify import Detoxify


@dataclass
class AttackConfig:
    attack_data_path: str
    target_model_name: str
    res_save_path: str
    detoxify_model_path: str
    detoxify_config_path: str
    target_model_path: str
    target_tokenizer_path: str
    api_key: str
    base_url: str
    evaluator_type: str
    evaluator_model_path: str
    device: str
    mode: str
    max_new_tokens: int = 2048


@dataclass
class AttackData:
    toxic_trunc: float
    benign_trunc: float
    target_model: Any = None
    tokenizer: Any = None
    population: List[dict] = field(default_factory=list)
    moderation_map: dict = field(default_factory=dict)
    detoxify_model: Any = None


@dataclass
class AttackStatus:
    current_idx: int = 0
    current_example: str = ''
    attack_prompt: str = ''
    attack_response: str = ''
    result_message: dict = field(default_factory=dict)
    attack_success_number: int = 0
    current_attack_success: bool = False
    current_prefix: str = ''
    current_suffix: str = ''
    total_attack_number: int = 0


class DRAInit:
    def __init__(self):
        self.population_initer = PopulationInitializer()
        self.template_initer = InitTemplates()

    def init_manager(self, config: AttackConfig, data: AttackData):
        # data.population = self.population_initer.init_population(data_path=config.data_path)[:50]
        data.population = self.population_initer.init_population(data_path=config.attack_data_path)
        self.init_manager_for_mutate(config, data)
        if config.target_model_path:
            target_model = AutoModelForCausalLM.from_pretrained(config.target_model_path).eval().to(config.device)
            if not config.target_tokenizer_path:
                config.target_tokenizer_path = config.target_model_path
            tokenizer = AutoTokenizer.from_pretrained(config.target_tokenizer_path)
            data.target_model = load_model(model=target_model, tokenizer=tokenizer, model_name=config.target_model_name, generation_config={'max_length': config.max_new_tokens})
            data.tokenizer = tokenizer
        else:
            raise NotImplementedError
            # data.target_model = load_model(model_name=config.model_name, base_url=config.base_url, api_key=config.api_key)


    def init_manager_for_mutate(self, config: AttackConfig, data: AttackData):
        templates = self.template_initer.get_templates('dra', 1)[0]
        data.moderation_map = templates['moderation_map']
        data.detoxify_model = Detoxify(model_type='original', checkpoint=config.detoxify_model_path, huggingface_config_path=config.detoxify_config_path)
        # data.detoxify_model = Detoxify('original', checkpoint=config.detoxify_model_path)



class DRASelector:
    def __init__(self):
        pass

    def select_finished(self, data: AttackData, status: AttackStatus):
        return status.current_idx == len(data.population)

    def select_example(self, data: AttackData, status: AttackStatus):
        status.current_example = data.population[status.current_idx]['query']
        status.current_idx += 1


class DRAMutator:
    def __init__(self):
        pass

    def check_moderation(self, query, data: AttackData):
        result = data.detoxify_model.predict(query)
        max_score = max(result.values())
        if max_score >= 0.01:
            return True
        else:
            return False


    def prefix_gen(self, status: AttackStatus):
        question = status.current_example.lower()
        res = ''
        for letter in question:
            # Skip or preserve
            if letter not in LETTER_MAP.keys():
                LETTER_MAP[letter] = [letter]
            token = random.choice(LETTER_MAP[letter])
            idxs = []
            for i in range(len(token)):
                if token[i] == letter:
                    idxs.append(i)
            mutate = random.choice(idxs)
            token = list(token)
            token[mutate] = f'({token[mutate]})'
            mutate_token = ''.join(token)
            res += (mutate_token + "\n")

        status.current_prefix = res


    def suffix_gen(self, config: AttackConfig, data: AttackData, status: AttackStatus):
        # [0] load cache
        moderation_cache = data.moderation_map
        tokens = status.current_example.split()
        
        # [1] word guide, or extract keyword? only select the keywords?
        word_guide = ""
        for token in tokens:
            if token not in moderation_cache.keys():
                judger = self.check_moderation(token, data)
                moderation_cache[token] = judger
            else:
                judger = moderation_cache[token]
            
            if judger:
                ######### short words ############
                if len(token) <= 2:
                    word_guide += f"`{token}` "
                    continue
                if len(token) <= 4:
                    if token[-1].isalpha(): # ensure the last char is a-z
                        idx = 1
                    else:
                        idx = 2
                    word_guide += f"`{token[:-idx]}` "
                    continue
                ######### GPT-4: model with strong capability ##########
                if config.target_model_name in ["gpt-4", "gpt-4o"]:
                    # cutoff gpt4
                    toxic_trunc_start = max(1, int(len(token) * data.toxic_trunc))
                    idx = random.randint(toxic_trunc_start, len(token)-1)
                    random_cutoff = random.randint(0, 2)
                    if random_cutoff != 0:
                        word_guide += f"`{token[:idx]}` "
                        if random.randint(0, 1):
                            word_guide += f"`{token[idx:]} `"
                    else:
                        word_guide += f"`{token[idx:]}` "
                ######### other model #########
                else:
                    ### adjust the cutoff prob automatically based on the attack result
                    toxic_trunc_start = max(2, int(len(token) * data.toxic_trunc))
                    idx = random.randint(toxic_trunc_start, len(token)-1)
                    word_guide += f"`{token[:idx]}` "
                    # randomly pad the word to preserve the imaginary
                    if random.randint(0, 1):
                        word_guide += f"`{token[idx:]} `"
            else:
                if len(token) <= 3:
                    word_guide += f"`{token}` "
                    continue
                # randomly cutoff
                # cutoff prob
                ub = 3 if config.target_model_name in ["gpt-4", "gpt-4o"] else 2
                if random.randint(0, ub) == 0:
                    word_guide += f"`{token}` "
                else:
                    # gpt4: cutoff from head or tail
                    if config.target_model_name in ["gpt-4", "gpt-4o"]:
                        benign_trunc_start = int(len(token) * data.benign_trunc)
                        idx = random.randint(1, benign_trunc_start)
                        if random.randint(0, 1):
                            word_guide += f"`{token[idx:]}` "
                        else:
                            word_guide += f"`{token[:idx]}` "
                    # llama: cutoff from head
                    # llama: randomly pad the word to preserve the imaginary
                    else:
                        ### version 1
                        # idx = random.randint(len(token)*2//3, len(token)-1)
                        benign_trunc_start = min(len(token)*2//3, int(len(token) * data.benign_trunc))
                        idx = random.randint(benign_trunc_start, len(token)-1)
                        word_guide += f"`{token[:idx]}` "
                        # TODO
                        if random.randint(0, 1):
                            word_guide += f"`{token[idx:]}` "
        # [2] tool guide
        # naively choose some 
        tools = random.sample(TOOL_LIST, 2)
        tools = ' and '.join(tools)
        # [3] fill it
        if "gpt" in config.target_model_name:
            if 'gpt-4' in config.target_model_name and 'gpt-4o' not in config.target_model_name:
                alpha = 0.7
                S = 'GPT' if random.uniform(0, 1) < alpha else 'LLAMA'
                if random.randint(0, 1):
                    suffix = eval(f'{S}_SUFFIX_TOOL_DEV').format(word_guide, tools)
                else:
                    suffix = eval(f'{S}_SUFFIX_DEV').format(word_guide)
            else:        
                if random.randint(0, 1):
                    suffix = GPT_SUFFIX_TOOL_DEV.format(word_guide, tools)
                else:
                    suffix = GPT_SUFFIX_DEV.format(word_guide)
        else:
            # randomly use tools.
            if 'lyrics of' not in status.current_example and 'long passage' not in status.current_example:
                if random.randint(0, 2):
                    suffix = LLAMA_SUFFIX_TOOL_DEV.format(word_guide, tools)
                    suffix = suffix.replace('`', '"')
                else:
                    suffix = LLAMA_SUFFIX_DEV.format(word_guide)
                    suffix = suffix.replace('`', '"')
            else:
                if 'lyrics of' in status.current_example:
                    tools = 'lyrics'
                elif 'long passage' in status.current_example:
                    tools = 'long passage'
                suffix = LLAMA_SUFFIX_TOOL.format(word_guide, tools)
                suffix = suffix.replace('`', '"')
        
        status.current_suffix = suffix


    def mutate_example(self, config: AttackConfig, data: AttackData, status: AttackStatus):
        self.prefix_gen(status)
        self.suffix_gen(config, data, status)
        status.attack_prompt = status.current_prefix + status.current_suffix


class DRAEvaluator:
    def __init__(self, evaluator_type, evaluator_model_path, mode):
        self.evaluator_type = evaluator_type
        if mode == 'mutate': return
        if evaluator_type is None: return
        if evaluator_type == "pattern":
            self.evaluator = PatternScorer()
        elif evaluator_type == "llamaguard3":
            self.evaluator = LlamaGuard3Scorer(model_path=evaluator_model_path, tokenizer_path=evaluator_model_path)
        elif evaluator_type == "harmbench":
            self.evaluator = HarmBenchScorer(model_path=evaluator_model_path)
        else:
            raise ValueError(f"Invalid evaluator type: {evaluator_type}")

    def evaluate(self, data: AttackData, status: AttackStatus):
        success = 0
        response = data.target_model.chat(status.attack_prompt)
        result_message = {}
        result_message['example_idx'] = status.current_idx 
        if self.evaluator_type is not None:
            judge = self.evaluator.score(status.current_example, response).get('score', 0)
            result_message.update({'success': judge, 'final_query': status.attack_prompt, 'query': status.current_example, 'response': response})
            if judge:
                success = 1
        else:
            result_message.update({'final_query': status.attack_prompt, 'query': status.current_example, 'response': response})
        status.result_message = result_message

        if self.evaluator_type:
            status.attack_success_number += success
            status.current_attack_success = success


class DRAManager(BaseAttackManager):
    """
    A class used to manage and execute DRA attacks.

    Parameters
    ----------

    attack_data_path: str
        The path to load the jailbreaking prompts
    target_model_name: str
        The name of the targe model to attack (default is None)
    res_save_path: str
        The path to save the results of DRA-Attack
    detoxify_model_path: str
        The path to the detoxify model
    detoxify_config_path:
        The path to the directory containing config.json of detoxify model
    target_model_path: str
        The path to load the targe model to attack (necessary for a local model)
    target_tokenizer_path: str
        The path to load the targe model tokenizer to attack (default is None)
    api_key: str
        The api key for calling the target model through api (default is None)
    base_url: str
        The base url for calling the target model through api (default is None)
    evaluator_type: str
        The evaluator model used for jailbreaking evaluation (default is None)
    evaluator_path: str
        The path to load the evaluator model (necessary for a local evaluator model)
    device: str
        The GPU device id (default is None)
    mode: str
        The mode of DRA-attack (can be 'attack' or 'mutate', default is 'attack')
    toxic_trunc: float
        The truncation ratio for toxic tokens (default is 0.5).
    benign_trunc: float
        The truncation ratio for toxic tokens (default is 0.5).
    
    Methods:
        mutate(): Mutate a single input string.
        attack(): Attack with defenders.
    """

    def __init__(
        self,
        attack_data_path: str,
        target_model_name: str,
        res_save_path: str,
        detoxify_model_path: str,
        detoxify_config_path: str,
        target_model_path: str = None,
        target_tokenizer_path: str = None,
        api_key: str = None,
        base_url: str = None,
        evaluator_type: str = None,
        evaluator_model_path: str = None,
        device: str = None,
        mode: str = 'attack',
        toxic_trunc: float = 0.5,
        benign_trunc: float = 0.5,
        **kwargs
    ):
        super().__init__(res_save_path)
        self.config = AttackConfig(attack_data_path, target_model_name, res_save_path, detoxify_model_path, detoxify_config_path, target_model_path, target_tokenizer_path, api_key, base_url, evaluator_type, evaluator_model_path, device, mode)
        self.data = AttackData(toxic_trunc=toxic_trunc, benign_trunc=benign_trunc)
        self.status = AttackStatus()
        self.init = DRAInit()
        self.selector = DRASelector()
        self.mutator = DRAMutator()
        self.evaluator = DRAEvaluator(evaluator_type, evaluator_model_path, mode)
        if mode == 'mutate':
            self.log('Create for mutate only. Do not call attack in this mode.')

    def attack(self):
        self.init.init_manager(self.config, self.data)
        timer = Timer.start()
        self.log('Attack started')
        while (not self.selector.select_finished(self.data, self.status)):
            self.selector.select_example(self.data, self.status)
            self.mutator.mutate_example(self.config, self.data, self.status)
            self.evaluator.evaluate(self.data, self.status)
            self.update_res(self.status.result_message)
            if self.config.evaluator_type is not None:
                self.log(f'Attack on sample {self.status.current_idx} success: {self.status.current_attack_success}')
            else:
                self.log(f'Attack on sample {self.status.current_idx} finished.')


        if self.config.evaluator_type is not None:
            self.log(f'ASR: {round(self.status.attack_success_number / len(self.status.total_attack_number) * 100, 2)}%')
            self.log(f'Time cost: {timer.end()}s.')

    def mutate(self, prompt: str):
        self.init.init_manager_for_mutate(self.data)
        self.status.current_example = prompt
        self.mutator.mutate_example(self.config, self.data, self.status)
        return self.status.attack_prompt