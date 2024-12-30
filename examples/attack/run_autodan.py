# refer to https://github.com/pytorch/pytorch/issues/97207
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=UserWarning, message='The torchvision.datapoints')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from aisafetylab.attack.attackers.autodan import AutoDANManager
from aisafetylab.logging import setup_logger
from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments

sys.path.append(os.getcwd())

setup_logger(log_file_path='logs/autodan_vicuna_{time}.log', stderr_level='DEBUG')

args = parse_arguments()
config_manager = ConfigManager(config_path=args.config_path)
attacker = AutoDANManager.from_config(config_manager.config)
attacker.attack()
