"""
GPTFUZZER Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: GPTFUZZER : Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts
arXiv link: https://arxiv.org/pdf/2309.10253.pdf
Source repository: https://github.com/sherdencooper/GPTFuzz.git
"""

from loguru import logger
import random
from tqdm import tqdm
import json

import copy
from typing import List

from dataclasses import dataclass, field, fields
from typing import Any, Optional
from aisafetylab.utils import Timer

from aisafetylab.attack.attackers.base_attacker import BaseAttackManager
from aisafetylab.attack.initialization.seed_template import SeedTemplate
from aisafetylab.dataset.example import Example
from aisafetylab.dataset import AttackDataset
from aisafetylab.evaluation.scorers.classification_scorer import ClassficationScorer
from aisafetylab.attack.selection.mcts_selection import MctsSelection

from aisafetylab.attack.mutation.crossover import GPTFuzzerCrossOver
from aisafetylab.attack.mutation import Expand, GenerateSimilar, Shorten, Rephrase
from aisafetylab.models.local_model import LocalModel
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class AttackConfig:
    attack_model_path: str
    target_model_path: str
    attack_model_name: str
    target_model_name: str
    attack_model_generation_config: dict
    target_model_generation_config: dict
    eval_model_path: str
    dataset_path: str
    seeds_num: int
    energy: int
    max_query: int
    max_jailbreak: int
    max_reject: int
    max_iteration: int
    template_file: Optional[str] = None
    res_save_path: Optional[str] = None


@dataclass
class AttackData:
    attack_results: Optional[AttackDataset] = AttackDataset([])
    query_to_idx: Optional[dict] = field(default_factory=dict)
    questions: Optional[AttackDataset] = AttackDataset([])
    prompt_nodes: Optional[AttackDataset] = AttackDataset([])
    initial_prompts_nodes: Optional[AttackDataset] = AttackDataset([])
    query: str = None
    jailbreak_prompt: str = None
    reference_responses: List[str] = field(default_factory=list)
    target_responses: List[str] = field(default_factory=list)
    eval_results: list = field(default_factory=list)
    attack_attrs: dict = field(default_factory=lambda: {'Mutation': None, 'query_class': None})
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    index: int = None
    visited_num: int = 0
    level: int = 0
    current_query: int = 0
    current_jailbreak: int = 0
    current_reject: int = 0
    total_query: int = 0
    total_jailbreak: int = 0
    total_reject: int = 0
    current_iteration: int = 0
    _num_query: int = 0
    _num_jailbreak: int = 0
    _num_reject: int = 0

    def copy(self):
        return copy.deepcopy(self)

    @property
    def num_query(self):
        return len(self.target_responses)

    @property
    def num_jailbreak(self):
        return sum(i for i in self.eval_results)

    @property
    def num_reject(self):
        return len(self.eval_results) - sum(i for i in self.eval_results)
    
    def copy_for_eval(self):
        """
        Lightweight copy used during evaluation to avoid deepcopy overhead.
        """
        return AttackData(
            query=self.query,
            jailbreak_prompt=self.jailbreak_prompt,
            attack_attrs=self.attack_attrs.copy(),
            index=self.index,
            level=self.level
        )


class GPTFuzzerInit(object):
    """
    Initializes the attack data structure and seeds.
    """
    def __init__(self, config: AttackConfig, jailbreak_datasets: Optional[AttackDataset] = None):
        self.config = config
        
        self.data = AttackData(questions=jailbreak_datasets)
        self.data.questions_length = len(self.data.questions) if jailbreak_datasets else 0

        # Initialize seeds and prompts
        self.data.initial_prompt_seed = SeedTemplate().new_seeds(
            seeds_num=config.seeds_num, prompt_usage='attack',
            method_list=['Gptfuzzer'], template_file=config.template_file
        )
        
        self.data.prompt_nodes = AttackDataset(
            [AttackData(jailbreak_prompt=prompt) for prompt in self.data.initial_prompt_seed]
        )
        for i, instance in enumerate(self.data.prompt_nodes):
            instance.index = i
            instance.visited_num = 0
            instance.level = 0
        for i, instance in enumerate(self.data.questions):
            instance.index = i
        self.data.initial_prompts_nodes = AttackDataset([instance for instance in self.data.prompt_nodes])


class GPTFuzzerMutation(object):
    """
    Handles mutations for generating new prompts during the attack.
    """
    def __init__(self, attack_model, initial_prompts_nodes):

        self.mutations = [
            GPTFuzzerCrossOver(attack_model, seed_pool=initial_prompts_nodes),
            Expand(attack_model),
            GenerateSimilar(attack_model),
            Shorten(attack_model),
            Rephrase(attack_model)
        ]
        
        self.mutator = None
        
    def mutate(self, instance):
        self.mutator = random.choice(self.mutations)
        return self.mutator._get_mutated_instance(instance)


class GPTFuzzerEvaluator(object):
    """
    Evaluates the results of the generated prompts.
    """
    def __init__(self, eval_model=None, eval_model_path=None):
        assert eval_model is not None or eval_model_path is not None, "At least one of eval_model or eval_model_path must be provided"
        self.evaluator = ClassficationScorer(eval_model)
        self.evaluator.set_model(eval_model_path)


class GPTFuzzerManager(BaseAttackManager):

    def __init__(
        self,
        attack_model_path,
        target_model_path,
        attack_model_name,
        target_model_name,
        attack_model_generation_config=dict(
            max_new_tokens=512, do_sample=True, temperature=0.8
        ),
        target_model_generation_config=dict(
            max_new_tokens=512, do_sample=True, temperature=0.8
        ),
        eval_model_path="hubert233/GPTFuzz",
        dataset_path="thu-coai/AISafetyLab_Datasets/harmbench_standard",
        seeds_num=76,
        energy=5,
        max_query=100000,
        max_jailbreak=100000,
        max_reject=100000,
        max_iteration=100,
        template_file="aisafetylab/attack/initialization/init_templates.json",
        device="cuda:0",
        res_save_path="./results/gptfuzzer_results.jsonl",
        subset_slice=None
    ):

        super().__init__(res_save_path=res_save_path)

        _fields = fields(AttackConfig)
        local_vars = locals()
        # logger.debug(f'local_vars: {local_vars}')
        _kwargs = {field.name: local_vars[field.name] for field in _fields}
        self.config = AttackConfig(**_kwargs)
        logger.info(f"Attack configuration: {self.config}")

        self.attack_model = LocalModel(
            model=AutoModelForCausalLM.from_pretrained(self.config.attack_model_path, torch_dtype="auto").to(device).eval(),
            tokenizer=AutoTokenizer.from_pretrained(self.config.attack_model_path),
            model_name=self.config.attack_model_name,
            generation_config=self.config.attack_model_generation_config
        )

        if self.config.attack_model_path == self.config.target_model_path:
            self.target_model = self.attack_model
        else:
            self.target_model = LocalModel(
                model=AutoModelForCausalLM.from_pretrained(self.config.target_model_path, torch_dtype="auto").to(device).eval(),
                tokenizer=AutoTokenizer.from_pretrained(self.config.target_model_path),
                model_name=self.config.target_model_name,
                generation_config=self.config.target_model_generation_config
            )

        self.jailbreak_datasets = AttackDataset(self.config.dataset_path, subset_slice=subset_slice)

        self.gptfuzzer_init = GPTFuzzerInit(self.config, self.jailbreak_datasets)
        self.data = self.gptfuzzer_init.data

        self.select_policy =  MctsSelection(self.data.prompt_nodes, self.data.initial_prompts_nodes, self.data.questions)

        self.mutator = GPTFuzzerMutation(
            self.attack_model,
            self.data.initial_prompts_nodes
        )

        self.evaluator = GPTFuzzerEvaluator(eval_model_path=self.config.eval_model_path).evaluator

    def mutate(self, query, target="Sure, here's"):
        # Wrap the query into an Example instance for mutation
        example_instance = AttackData(query=query, jailbreak_prompt=target)

        # Apply a mutation operation using the mutator
        mutated_instance = self.mutator.mutate(example_instance)[0]

        # Generate the final query by applying the jailbreak prompt
        if '{query}' in mutated_instance.jailbreak_prompt:
            final_query = mutated_instance.jailbreak_prompt.format(query=mutated_instance.query)
        else:
            final_query = mutated_instance.jailbreak_prompt + mutated_instance.query
        return final_query

    def attack(self):
        """
        Main loop for the fuzzing process, repeatedly selecting, mutating, evaluating, and updating.
        """
        logger.info("Fuzzing started!")
        example_idx_to_results = {}
        try:
            while not self.is_stop():
                seed_instance = self.select_policy.select()[0]
                logger.debug('begin to mutate')
                mutated_results = self.single_attack(seed_instance)
                for instance in mutated_results:
                    instance.parents = [seed_instance]
                    instance.children = []
                    seed_instance.children.append(instance)
                    instance.index = len(self.data.prompt_nodes)
                    self.data.prompt_nodes.data.append(instance)
                logger.debug(f'mutated results length: {len(mutated_results)}')
                for mutator_instance in tqdm(mutated_results):
                    self.temp_results = AttackDataset([])
                    logger.info('Begin to get model responses')
                    input_seeds = []
                    for query_instance in tqdm(self.data.questions):
                        if '{query}' in mutator_instance.jailbreak_prompt:
                            input_seed = mutator_instance.jailbreak_prompt.replace('{query}', query_instance.query)
                        else:
                            input_seed = mutator_instance.jailbreak_prompt + query_instance.query
                        input_seeds.append(input_seed)
                    
                    all_responses = self.target_model.batch_chat(input_seeds, batch_size=10)
                            
                    for idx, query_instance in enumerate(tqdm(self.data.questions)):
                        # logger.debug(f'mutator_instance: {mutator_instance}')
                        # t = Timer.start()
                        temp_instance = mutator_instance.copy_for_eval()
                        # print(f'Copy costs {t.end()} seconds')
                        temp_instance.target_responses = []
                        temp_instance.eval_results = []
                        temp_instance.query = query_instance.query

                        if temp_instance.query not in self.data.query_to_idx:
                            self.data.query_to_idx[temp_instance.query] = len(self.data.query_to_idx)
                        # query_idx = self.data.query_to_idx[temp_instance.query]

                        # if '{query}' in temp_instance.jailbreak_prompt:
                        #     input_seed = temp_instance.jailbreak_prompt.replace('{query}', temp_instance.query)
                        # else:
                        #     input_seed = temp_instance.jailbreak_prompt + temp_instance.query
                        
                        # # logger.debug(f'input_seed: {input_seed}, target_model device: {self.target_model.device}')
                        # response = self.target_model.chat(input_seed)
                        response = all_responses[idx]
                        temp_instance.target_responses.append(response)
                        self.temp_results.data.append(temp_instance)

                    logger.info(f'Begin to evaluate the model responses')
                    # logger.debug(f'evaluator device: {self.evaluator.model.device}')
                    self.evaluator(self.temp_results)
                    logger.debug(f'Finish evaluating the model responses')

                    mutator_instance.level = seed_instance.level + 1
                    mutator_instance.visited_num = 0

                    self.update(self.temp_results)
                    for instance in self.temp_results:
                        # self.data.attack_results.data.append(instance.copy())
                        example_idx = self.data.query_to_idx[instance.query]
                        if '{query}' in instance.jailbreak_prompt:
                            final_query = instance.jailbreak_prompt.replace('{query}', instance.query)
                        else:
                            final_query = instance.jailbreak_prompt + instance.query
                        logger.debug(f'final_query: {final_query}\nresponse: {instance.target_responses[0]}\nattack_success: {instance.eval_results[0]}')

                        result = {
                            "example_idx": self.data.query_to_idx[instance.query],
                            "query": instance.query,
                            "final_query": final_query,
                            "response": instance.target_responses[0] if instance.target_responses else "",
                            "success": any(instance.eval_results),
                            "attack_success": instance.eval_results[0] if instance.eval_results else 0
                        }
                        if example_idx not in example_idx_to_results:
                            example_idx_to_results[example_idx] = result
                        else:
                            if result['attack_success']:
                                example_idx_to_results[example_idx] = result
                            elif not example_idx_to_results[example_idx]['attack_success']:
                                example_idx_to_results[example_idx] = result
                        # self.save(result)
                logger.info(f"Current iteration: {self.data.current_iteration}")
                
                with open(self.res_save_path, 'w') as f:
                    keys = sorted(example_idx_to_results.keys())
                    for key in keys:
                        f.write(json.dumps(example_idx_to_results[key], ensure_ascii=False) + '\n')
                

        except KeyboardInterrupt:
            logger.info("Fuzzing interrupted by user!")
        logger.info("Fuzzing finished!")


    def single_attack(self, instance: Example):
        """
        Perform an attack using a single query.
        """
        assert instance.jailbreak_prompt is not None, 'A jailbreak prompt must be provided'
        instance = instance.copy()
        instance.parents = []
        instance.children = []

        return_dataset = AttackDataset([])
        for i in range(self.config.energy):

            attack_dataset = AttackDataset([instance])
            new_dataset = []
            for sample in attack_dataset:
                mutated_instance_list = self.mutator.mutate(sample)
                new_dataset.extend(mutated_instance_list)
            instance = new_dataset[0]

            if instance.query is not None:
                if '{query}' in instance.jailbreak_prompt:
                    input_seed = instance.jailbreak_prompt.format(query=instance.query)
                else:
                    input_seed = instance.jailbreak_prompt + instance.query
                response = self.target_model.chat(input_seed)
                instance.target_responses.append(response)
            instance.parents = []
            instance.children = []
            return_dataset.data.append(instance)
        return return_dataset

    def is_stop(self):
        checks = [
            ('max_query', 'total_query'),
            ('max_jailbreak', 'total_jailbreak'),
            ('max_reject', 'total_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        logger.debug(f'in is_stop check-> max_iteration: {self.config.max_iteration}, current_iteration: {self.data.current_iteration}')
        return any(getattr(self.config, max_attr) != -1 and getattr(self.data, curr_attr) >= getattr(self.config, max_attr) for
                   max_attr, curr_attr in checks)

    def update(self, dataset: AttackDataset):
        self.data.current_iteration += 1

        current_jailbreak = 0
        current_query = 0
        current_reject = 0
        for instance in dataset:
            current_jailbreak += instance.num_jailbreak
            current_query += instance.num_query
            current_reject += instance.num_reject

            self.data.total_jailbreak += instance.num_jailbreak
            self.data.total_query += instance.num_query
            self.data.total_reject += instance.num_reject

        self.data.current_jailbreak = current_jailbreak
        self.data.current_query = current_query
        self.data.current_reject = current_reject

        self.select_policy.update(dataset)

if __name__ == "__main__":
    from aisafetylab.attack.attackers.gptfuzzer import GPTFuzzerManager
    from aisafetylab.utils import parse_arguments
    from aisafetylab.utils.config_utils import ConfigManager

    args = parse_arguments()

    if args.config_path is None:
        args.config_path = 'examples/attack/configs/gptfuzzer.yaml'
    config_manager = ConfigManager(config_path=args.config_path)
    gptfuzzer_manager = GPTFuzzerManager.from_config(config_manager.config)
    result = gptfuzzer_manager.mutate("What is the capital of France?", "Sure, here's")
    print(result)
