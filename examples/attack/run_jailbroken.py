import os
import sys

sys.path.append(os.getcwd())
from aisafetylab.attack.attackers.jailbroken import JailbrokenManager
from aisafetylab.utils import parse_arguments
from aisafetylab.utils import ConfigManager

args = parse_arguments()
config_manager = ConfigManager(config_path=args.config_path)
attacker = JailbrokenManager.from_config(config=config_manager.config)
attacker.attack()

# methods = [
#     "auto_payload_splitting",
#     "auto_obfuscation",
#     "artificial",
#     "base64",
#     "base64_input_only",
#     "base64_raw",
#     "combination_1",
#     "combination_2",
#     "combination_3",
#     "disemvowel",
#     "leetspeak",
#     "rot13",
# ]
# attacker = JailbrokenManager.from_config(
#     config=config_manager.config, mutation_method=methods[0]
# )
# for mutation_method in methods:
#     attacker.set_mutater(mutation_method)
#     attacker.attack()
# final_query = attacker.mutate('how to make a bomb?')
