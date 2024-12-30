from aisafetylab.attack.attackers.laa import LAAManager
from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments

if __name__ == "__main__":
    args = parse_arguments()


    if args.config_path is None:
        args.config_path = './configs/laa.yaml'


    config_manager = ConfigManager(config_path=args.config_path)

    attack_manager = LAAManager.from_config(config_manager.config)

    attack_manager.attack()
