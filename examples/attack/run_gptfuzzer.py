#TODO
from aisafetylab.attack.attackers.gptfuzzer import GPTFuzzerManager
from aisafetylab.utils import parse_arguments
from aisafetylab.utils.config_utils import ConfigManager

args = parse_arguments()

if args.config_path is None:
    args.config_path = 'examples/attack/configs/gptfuzzer.yaml'


config_manager = ConfigManager(config_path=args.config_path)
 
gptfuzzer_manager = GPTFuzzerManager.from_config(config_manager.config)
gptfuzzer_manager.attack()
