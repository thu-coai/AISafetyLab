import os
import sys
sys.path.append(os.getcwd())
from aisafetylab.attack.attackers.dra import DRAManager
from aisafetylab.logging import setup_logger
from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments

args = parse_arguments()
config_manager = ConfigManager(config_path=args.config_path)
# setup_logger(log_file_path=config_manager.config.log_file_path, stderr_level='INFO')

dra_manager = DRAManager.from_config(config_manager.config)
dra_manager.attack()
