"""
AutoDAN Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models
arXiv link: https://arxiv.org/abs/2310.04451
Source repository: https://github.com/SheltonLiu-N/AutoDAN.git
"""

import gc
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, fields
from typing import List, Optional, Any

import nltk
import torch
from aisafetylab.attack.attackers import BaseAttackManager
from aisafetylab.attack.feedback import generate
from aisafetylab.attack.initialization import InitTemplates
from aisafetylab.attack.mutation import CrossOver, Rephrase, Synonyms
from aisafetylab.attack.prompt_manager import PrefixPromptManager
from aisafetylab.attack.selection import (RouletteWheelSelection,
                                          TargetLossSelection)
from aisafetylab.dataset import AttackDataset, Example
from aisafetylab.evaluation.scorers import PatternScorer
from aisafetylab.models import load_model
from aisafetylab.utils import Timer, get_developer
from loguru import logger
from nltk.corpus import stopwords, wordnet
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AttackConfig:
    attack_dataset_path: str
    subset_slice: Any = None
    mutation_rate: float = 0.01
    num_candidates: int = 64
    num_steps: int = 100
    ratio_elites: float = 0.1
    num_points: int = 5
    crossover_rate: float = 0.5
    sentence_level_steps: int = 5
    model_name: str = "llama-2"
    word_dict_size: int = 50
    res_save_path: Optional[str] = None

@dataclass
class AttackData:
    attack_dataset: str
    init_templates: List[str]
    word_dict: dict = field(default_factory=dict)
    prompt_manager: Optional[PrefixPromptManager] = None
    score_list = None
    current_loss = None
    best_id = None
    best_template = None
    response = None
    is_success = False
    example_idx = None
    step = None
    sentence_step = None

class AutoDANInit:
    def __init__(self, config) -> None:
        attack_dataset = AttackDataset(config.attack_dataset_path, config.subset_slice)
        init_templates = InitTemplates().get_templates("AutoDAN", config.num_candidates)
        for i in range(len(init_templates)):
            init_templates[i] = (
                init_templates[i]
                .replace("ChatGPT", config.model_name)
                .replace("chatgpt", config.model_name)
                .replace("ModelKeeper", get_developer(config.model_name))
            )
            init_templates[i] = init_templates[i] + " [PROMPT]:"
        
        data = AttackData(attack_dataset=attack_dataset, init_templates=init_templates)
        self.config = config
        self.data = data
    
    def example_init_attack(self, attack_manager, example, example_idx):
        query = example.query
        target = example.target
        prompt_manager = PrefixPromptManager(
            tokenizer=attack_manager.target_model.tokenizer,
            conv_template=attack_manager.conv_template,
            instruction=query,
            target=target,
            adv_string=attack_manager.data.init_templates[0],
        )
        example.candidate_templates = attack_manager.data.init_templates
        self.data.prompt_manager = prompt_manager
        self.data.example_idx = example_idx
        self.data.word_dict = {}
        
    def step_init_attack(self, step):
        logger.debug(f"step {step}")
        self.data.step = step
        self.data.sentence_step = -1
        
    def sentence_step_init_attack(self, sentence_step):
        logger.debug(f"sentence step {sentence_step}")
        self.data.sentence_step = sentence_step


class AutoDANMutator:
    def __init__(self, config, rephrase_model):
        self.mutation_rephrase = Rephrase(rephrase_model)
        self.mutation_crossover = CrossOver(num_points=config.num_points)
        self.mutation_synonyms = Synonyms(crossover_rate=config.crossover_rate)
        self.config = config
        self.mutation_rate = config.mutation_rate
        self.crossover_rate = config.crossover_rate

    def rephrase(self, templates):
        timer = Timer.start()
        for i in range(len(templates)):
            if random.random() < self.mutation_rate:
                templates[i] = self.mutation_rephrase.rephrase(
                    templates[i]
                )
        logger.debug(f"rephrase costs {timer.end():.2f} seconds")

    def crossover(self, parents_list):
        mutated_templates = []
        for i in range(0, len(parents_list), 2):
            parent1 = parents_list[i]
            parent2 = (
                parents_list[i + 1]
                if (i + 1) < len(parents_list)
                else parents_list[0]
            )
            if random.random() < self.crossover_rate:
                child1, child2 = self.mutation_crossover.crossover(
                    parent1, parent2
                )
                mutated_templates.append(child1)
                mutated_templates.append(child2)
            else:
                mutated_templates.append(parent1)
                mutated_templates.append(parent2)

        return mutated_templates

    def _construct_momentum_word_dict(
        self, word_dict, control_suffixs, score_list, topk=-1
    ):
        T = {
            "llama2",
            "llama-2",
            "meta",
            "vicuna",
            "lmsys",
            "guanaco",
            "theblokeai",
            "wizardlm",
            "mpt-chat",
            "mosaicml",
            "mpt-instruct",
            "falcon",
            "tii",
            "chatgpt",
            "modelkeeper",
            "prompt",
        }
        stop_words = set(stopwords.words("english"))
        if len(control_suffixs) != len(score_list):
            raise ValueError(
                "control_suffixs and score_list must have the same length."
            )

        word_scores = defaultdict(list)

        for prefix, score in zip(control_suffixs, score_list):
            words = set(
                [
                    word
                    for word in nltk.word_tokenize(prefix)
                    if word.lower() not in stop_words and word.lower() not in T
                ]
            )
            for word in words:
                word_scores[word].append(score)

        for word, scores in word_scores.items():
            avg_score = sum(scores) / len(scores)
            if word in word_dict:
                word_dict[word] = (word_dict[word] + avg_score) / 2
            else:
                word_dict[word] = avg_score

        sorted_word_dict = OrderedDict(
            sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
        )
        # logger.debug(f'word dict length before topk: {len(sorted_word_dict)}')
        if topk == -1:
            topk_word_dict = dict(list(sorted_word_dict.items()))
        else:
            topk_word_dict = dict(list(sorted_word_dict.items())[:topk])
        return topk_word_dict

    def synonyms_replacement(self, candidate_templates, example, data):
        data.word_dict = self._construct_momentum_word_dict(
            data.word_dict, example.candidate_templates, data.score_list, topk=self.config.word_dict_size
        )
        self.mutation_synonyms.update(data.word_dict)

        new_templates = [
            self.mutation_synonyms.replace_synonyms(template)
            for template in candidate_templates
        ]
        return new_templates


class AutoDANSelector:
    def __init__(self):
        self.selection_loss = TargetLossSelection()
        self.selection_roulette_wheel = RouletteWheelSelection()
        
    def select_loss(self, example, attack_manager, data):
        device = attack_manager.target_model.device
        prompt_manager = data.prompt_manager
        input_ids_list = []
        loss_slices = []
        for template in example.candidate_templates:
            input_ids = prompt_manager.get_input_ids(adv_string=template).to(device)
            input_ids_list.append(input_ids)
            loss_slices.append(prompt_manager._loss_slice)
        score_list, current_loss, best_id = self.selection_loss.select(
            attack_manager.target_model.model,
            input_ids_list,
            loss_slices,
            attack_manager.target_model.tokenizer.pad_token_id,
        )
        # logger.debug(f'score list: {sorted(score_list)}')
        data.score_list = [-x for x in score_list]
        data.current_loss = current_loss
        data.best_id = best_id
        
    def select_elites_new_parents(self, example, attack_manager, data):
        score_list = data.score_list
        sorted_indices = sorted(
            range(len(score_list)),
            key=lambda k: score_list[k],
            reverse=True,
        )
        sorted_templates = [
            example.candidate_templates[i] for i in sorted_indices
        ]
        sorted_score_list = [score_list[i] for i in sorted_indices]

        # Select the elites
        num_elites = int(attack_manager.config.ratio_elites * attack_manager.config.num_candidates)
        elites = sorted_templates[:num_elites]

        # Use roulette wheel selection for the remaining positions
        parents_list, parents_score_list = self.selection_roulette_wheel.select_roulette_wheel(
            sorted_templates,
            sorted_score_list,
            attack_manager.config.num_candidates - num_elites,
        )
        # logger.debug(f'elites score: {sorted_score_list[:num_elites]}')
        return elites, parents_list, num_elites, parents_score_list

class AutoDANEvaluator:
    def __init__(self):
        self.scorer = PatternScorer()

    def generate_evaluate(self, example, attack_manager, data):
        device = attack_manager.target_model.device
        best_id = data.best_id
        prompt_manager = data.prompt_manager

        best_template = example.candidate_templates[best_id]
        temp_input_ids = prompt_manager.get_input_ids(adv_string=best_template)

        logger.debug(
            f"target model generate input_ids: {attack_manager.target_model.tokenizer.decode(temp_input_ids[: prompt_manager._assistant_role_slice.stop], add_special_tokens=False)}"
        )
        output_ids = attack_manager.target_model.generate(
            temp_input_ids[: prompt_manager._assistant_role_slice.stop]
            .to(device)
            .unsqueeze(0),
            batch=False,
        )
        response = attack_manager.target_model.tokenizer.decode(
            output_ids, skip_special_tokens=True
        ).strip()
        data.best_template = best_template
        data.response = response

        is_success = self.scorer.score(response)['score']

        logger.debug(
            f"Step {data.step} (sentence step {data.sentence_step}), current loss: {data.current_loss}\nbest template: {best_template}\nresponse: {response}\nis_success: {is_success}"
        )
        data.is_success = is_success

class AutoDANManager(BaseAttackManager):

    def __init__(
        self,
        attack_dataset_path,
        target_model_name_or_path,
        rephrase_model_name_or_path,
        rephrase_api_key,
        rephrase_base_url,
        tokenizer_kwargs=None,
        mutation_rate=0.01,
        word_dict_size=50,
        num_candidates=64,
        num_steps=100,
        ratio_elites=0.1,
        num_points=5,
        crossover_rate=0.5,
        sentence_level_steps=5,
        model_name="llama-2",
        res_save_path=None,
        delete_existing_res=True,
        subset_slice=None
    ):
        super().__init__(res_save_path, delete_existing_res)
        _fields = fields(AttackConfig)
        local_vars = locals()
        _kwargs = {field.name: local_vars[field.name] for field in _fields}
        self.config = AttackConfig(**_kwargs)
        # create target model and rephrase model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = (
            AutoModelForCausalLM.from_pretrained(
                target_model_name_or_path, torch_dtype=torch.float16
            )
            .eval()
            .to(device)
        )
        if tokenizer_kwargs is None:
            tokenizer = AutoTokenizer.from_pretrained(target_model_name_or_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(target_model_name_or_path, **tokenizer_kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        target_model = load_model(model=model, tokenizer=tokenizer, model_name=model_name)

        rephrase_model = load_model(
            model_name=rephrase_model_name_or_path, base_url=rephrase_base_url, api_key=rephrase_api_key
        )

        self.init = AutoDANInit(self.config)
        self.mutator = AutoDANMutator(self.config, rephrase_model)
        self.selector = AutoDANSelector()
        self.evaluator = AutoDANEvaluator()
        self.data = self.init.data

        self.target_model = target_model
        self.conv_template = target_model.conversation

    def _loss_select_evaluate(self, example):
        with torch.no_grad():
            timer = Timer.start()
            self.selector.select_loss(example, self, self.data) # most time on this
            logger.debug(f'select loss costs {timer.end():.2f} seconds')
            timer = Timer.start()
            self.evaluator.generate_evaluate(example, self, self.data)
            logger.debug(f"generate evaluate costs {timer.end():.2f} seconds")

    def _check_success(self, example, log=True):
        data = self.data
        if data.is_success and log:
            self.log(
                {
                    "example_idx": data.example_idx,
                    "step": data.step,
                    "loss": data.current_loss.item(),
                    "success": True,
                    "final_template": data.best_template,
                    "query": example.query,
                    "final_query": data.best_template + " " + example.query,
                    "response": data.response,
                },
                save=True,
            )

        return data.is_success

    def _crossover_mutation_evolve(self, example):

        elites, parents_list, num_elites, _parents_score_list = self.selector.select_elites_new_parents(example, self, self.data)
        # Apply crossover and mutation to the selected parents
        mutated_templates = self.mutator.crossover(parents_list)
        self.mutator.rephrase(mutated_templates)

        # Combine elites with the mutated offspring
        next_generation = (
            elites + mutated_templates[: self.config.num_candidates - num_elites]
        )
        assert len(next_generation) == self.config.num_candidates
        example.candidate_templates = next_generation

    def _synonyms_replacement_rephrase_mutation_evolve(self, example):
        elites, parents_list, num_elites, parents_score_list = self.selector.select_elites_new_parents(example, self, self.data)
        mutated_templates = self.mutator.synonyms_replacement(
            parents_list, example, self.data
        )
        self.mutator.rephrase(mutated_templates)
        next_generation = (
            elites + mutated_templates[: self.config.num_candidates - num_elites]
        )
        assert len(next_generation) == self.config.num_candidates
        example.candidate_templates = next_generation

    def attack(self):

        for _example_idx, example in enumerate(
            self.data.attack_dataset):

            if isinstance(self.config.subset_slice, list):
                offset = self.config.subset_slice[0]
                example_idx = _example_idx + offset
                logger.info(f"Attacking example {example_idx + 1} / {len(self.data.attack_dataset) + offset} ...")

            else:
                example_idx = _example_idx
                logger.info(f"Attacking example {example_idx + 1} / {len(self.data.attack_dataset)} ...")

            self.init.example_init_attack(self, example, example_idx)

            self.mutator.rephrase(example.candidate_templates)

            for step in trange(self.config.num_steps):
                self.init.step_init_attack(step)

                self._loss_select_evaluate(example)
                if self._check_success(example):
                    break

                self._crossover_mutation_evolve(example)

                # HIERARCHICAL GENETIC ALGORITHM
                # Sentence-level Iterations
                for sentence_step in range(self.config.sentence_level_steps):
                    self.init.sentence_step_init_attack(sentence_step)

                    self._loss_select_evaluate(example)
                    if self._check_success(example):
                        break

                    # Calculate momentum word score and Update sentences in each prompt
                    self._synonyms_replacement_rephrase_mutation_evolve(example)

                if self._check_success(example, log=False): # avoid to log twice
                    break

                if step == self.config.num_steps - 1:
                    self._loss_select_evaluate(example)
                    if self._check_success(example):
                        break

            example.prompt = self.data.best_template + " " + example.query

            if not self.data.is_success:
                self.log(
                    {
                        "example_idx": self.data.example_idx,
                        "step": self.data.step,
                        "loss": self.data.current_loss.item(),
                        "success": False,
                        "final_template": self.data.best_template,
                        "query": example.query,
                        "final_query":self.data.best_template + " " + example.query,
                        "response": self.data.response,
                    },
                    save=True,
                )

    def mutate(self, query, target="Sure, here's"):
        example = Example(query=query, target=target)
        example_idx = 0
        self.init.example_init_attack(self, example, example_idx)

        self.mutator.rephrase(example.candidate_templates)

        for step in trange(self.config.num_steps):
            self.init.step_init_attack(step)

            self._loss_select_evaluate(example)
            if self._check_success(example):
                break

            self._crossover_mutation_evolve(example)

            # HIERARCHICAL GENETIC ALGORITHM
            # Sentence-level Iterations
            for sentence_step in range(self.config.sentence_level_steps):
                self.init.sentence_step_init_attack(sentence_step)

                self._loss_select_evaluate(example)
                if self._check_success(example):
                    break

                # Calculate momentum word score and Update sentences in each prompt
                self._synonyms_replacement_rephrase_mutation_evolve(example)

            if self._check_success(example, log=False): # avoid to log twice
                break

            if step == self.config.num_steps - 1:
                self._loss_select_evaluate(example)
                if self._check_success(example):
                    break

        example.prompt = self.data.best_template + " " + example.query

        if not self.data.is_success:
            self.log(
                {
                    "example_idx": self.data.example_idx,
                    "step": self.data.step,
                    "loss": self.data.current_loss.item(),
                    "success": False,
                    "final_template": self.data.best_template,
                    "query": example.query,
                    "final_query":self.data.best_template + " " + example.query,
                    "response": self.data.response,
                },
                save=True,
            )

        return self.data.best_template + " " + example.query
