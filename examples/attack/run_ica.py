import os
import sys
sys.path.append(os.getcwd())

from aisafetylab import utils
from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments
from aisafetylab.attack.attackers.ica import ICAManager


args = parse_arguments()
if args.config_path is None:
    args.config_path = 'attack/configs/ica.yaml'
config_manager = ConfigManager(config_path=args.config_path)

attack_manager = ICAManager.from_config(config=config_manager.config)
attack_manager.attack()

# attack_pt = attack_manager.mutate("How to make a bomb?")
# print(attack_pt)