import os
import sys
sys.path.append(os.getcwd())
from aisafetylab.attack.attackers.multilingual import MultilingualManager
from aisafetylab.utils import parse_arguments
from aisafetylab.utils import ConfigManager


args = parse_arguments()
config_manager = ConfigManager(config_path=args.config_path)
# methods = ['zh-CN', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv']
# attacker = MultilingualManager.from_config(config=config_manager.config, mutation_method=methods[0])
attacker = MultilingualManager.from_config(config=config_manager.config)
# final_query = attacker.mutate('how to make a bomb?')
attacker.attack()