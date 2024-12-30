"""
DeepInception Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.
In addition to the template combination, 
we support searching with parameters like layer number and scene type to find a specific successful attack template.

Paper title: DeepInception: Hypnotize Large Language Model to Be Jailbreaker
arXiv link: https://arxiv.org/pdf/2311.03191
Source repository: https://github.com/tmlr-group/DeepInception
"""
from tqdm import tqdm, trange
from loguru import logger
from dataclasses import dataclass, field, fields
from typing import List, Optional
from aisafetylab.attack.attackers import BaseAttackManager
from aisafetylab.dataset import AttackDataset
from aisafetylab.attack.initialization import InitTemplates
from aisafetylab.attack.feedback import generate
from aisafetylab.evaluation.scorers import PatternScorer, HarmBenchScorer, LlamaGuard3Scorer
from aisafetylab.attack.mutation import DeepInception
from aisafetylab.models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class AttackConfig:
    # Configuration for the attack
    attack_data_path: str
    target_model_path: str
    target_model_name: str
    device: str
    lang: str
    layer_num: str
    character_num: str
    scene: str
    res_save_path: str
    mode: str
    
@dataclass
class AttackData:
    # Data structure for attack data
    attack_dataset: str
    lang: str
    
class DeepInceptionInit:
    """
    Initialize the attack with the given configuration.
    """
    def __init__(self, config) -> None:
        # Load attack dataset and model
        self.config = config
        self.config.template = InitTemplates().get_templates('DeepInception', 1)[0]
        print(config.mode)
        if config.mode == 'mutate':
            return
        attack_dataset = AttackDataset(config.attack_data_path)
        self.data = AttackData(attack_dataset=attack_dataset, lang=config.lang)
        model = AutoModelForCausalLM.from_pretrained(config.target_model_path).eval().to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.target_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.target_model = load_model(model=model, tokenizer=tokenizer, model_name=config.target_model_name)
        
class DeepInceptionMutator():
    def __init__(self, config):
        # Initialize the mutator with attack parameters
        self.mutator = DeepInception(
            config.layer_num,
            config.character_num,
            config.scene,
            config.template,
            config.lang
        )
    
    def mutate(self, query: str):
        # Apply mutation to the query
        return self.mutator.apply_template(query)

class DeepInceptionEvaluator:
    def __init__(self,evaluator_type, evaluator_model_path=None):
        # Initialize the evaluator based on the type
        self.evaluator_type = evaluator_type
        if evaluator_type == "pattern":
            self.scorer = PatternScorer()
        elif evaluator_type == "LlamaGuard3Scorer":
            self.scorer = LlamaGuard3Scorer(model_path=evaluator_model_path, tokenizer_path=evaluator_model_path)
        else:
            self.scorer = HarmBenchScorer(model_path=evaluator_model_path, tokenizer_path=evaluator_model_path)
    
    def evaluate(self, query, response):
        # Evaluate the success of the attack
        is_success = self.scorer.score(query, response)['score']
        return is_success

class DeepInceptionManager(BaseAttackManager):
    def __init__(
        self, 
        attack_data_path='harmbench.json',
        target_model_path='lmsys/vicuna-7b-v1.5',
        target_model_name='vicuna_v1.1',
        lang='en',
        evaluator_type='harmbench',
        template_path=None,
        device='cuda:0',
        evaluator_model_path="cais/HarmBench-Llama-2-13b-cls",
        res_save_path='./log_deepinception.jsonl', 
        layer_num=5, 
        character_num=5, 
        scene="science fiction",
        mode='attack'
        ):
        
        """Initialize the DeepInception class.

        Args:
            attack_data_path (str): Path to the dataset containing queries designed for the attack (harmful queries).
            target_model_path (str): Path to the target model that will be attacked.
            target_model_name (str): The name of the target model for proper logging and configuration handling.
            lang (str): The language in which the attack should be performed (e.g., 'en' for English).
            evaluator_type (str): The type of evaluator to assess the success of the attack. It can be one of 'pattern', 'LlamaGuard3Scorer', or 'harmbench'.
            template_path (Optional[str]): Path to custom templates to use for the attack. If None, default templates are used.
            device (str): The device on which to run the model, typically 'cuda:0' for GPU or 'cpu' for CPU.
            evaluator_model_path (str): Path to the model used by the evaluator for scoring the attack success.
            res_save_path (str): Path to save the results of the attack, typically in JSON Lines format.
            layer_num (int): The number of layers for the generated story, influencing attack complexity.
            character_num (int): The number of characters per layer in the generated story, also controlling attack depth.
            scene (str): The thematic scene of the story, which can influence the narrative of the generated attack query.
            mode (str): The mode of the attack, can be 'attack' or 'mutate'.

        This constructor initializes the attack configuration, the mutator for generating queries,
        and the evaluator for assessing the attack's success.
        """
        super().__init__(res_save_path)
        self.config = AttackConfig(
            attack_data_path, 
            target_model_path,
            target_model_name,
            device,
            lang,
            layer_num, 
            character_num, 
            scene,
            res_save_path,
            mode)
        self.init = DeepInceptionInit(self.config)
        self.mutator = DeepInceptionMutator(self.config)
        if self.config.mode == 'mutate':
            return
        self.evaluator = DeepInceptionEvaluator(evaluator_type, evaluator_model_path)
        self.data = self.init.data.attack_dataset
        self.target_model = self.init.target_model
    
    def attack(self):
        # Perform the attack on each example in the dataset
        for example_idx, example in enumerate(tqdm(self.data, desc="Attacking each example")):
            query = example.query
            final_query = self.mutator.mutate(query)
            response = self.target_model.chat(final_query)
            judge = self.evaluator.evaluate(query, response)
            self.log({'example_idx': example_idx, 'success': judge, 'final_query': final_query, 'query': query, 'response': response}, save=True)
    
    def mutate(self, query):
        # Mutate a given query
        final_query = self.mutator.mutate(query)
        return final_query