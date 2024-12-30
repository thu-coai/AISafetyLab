import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
sys.path.append(os.getcwd())
from aisafetylab.attack.attackers.deep_inception import DeepInceptionManager
from aisafetylab.logging import setup_logger
from aisafetylab.dataset import AttackDataset
from aisafetylab.models import LocalModel, OpenAIModel
from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments
from aisafetylab.attack.attackers.ica import ICAManager

args = parse_arguments()
config_manager = ConfigManager(config_path=args.config_path)

attack_manager = DeepInceptionManager.from_config(config=config_manager.config)
attack_manager.attack()