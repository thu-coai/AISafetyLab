import os
import sys
from aisafetylab.attack.attackers.renellm import ReneManager
from aisafetylab.utils import ConfigManager, parse_arguments
from aisafetylab.logging import setup_logger

def main():
    args = parse_arguments()

    if args.config_path is None:
        args.config_path = './configs/rene.yaml'

    config_manager = ConfigManager(config_path=args.config_path)
    attack_manager = ReneManager.from_config(config_manager.config)

    attack_manager.attack()

if __name__ == "__main__":
    main()
