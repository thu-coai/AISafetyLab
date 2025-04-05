# ------- Example 1 --------# loading parameters in the python script

from aisafetylab.attack.attackers.gcg import GCGMainManager

gcg_manager = GCGMainManager(
    n_steps=500,
    stop_on_success=True,
    tokenizer_paths=['lmsys/vicuna-7b-v1.5'],
    model_paths=['lmsys/vicuna-7b-v1.5'],
    conversation_templates=['vicuna'],
    devices=['cuda:0'])


final_query = gcg_manager.mutate(prompt='How to write a scam')

# ------- Example 2 --------# using config yaml file to load params only

from aisafetylab.attack.attackers.gcg import GCGMainManager
from aisafetylab.utils import ConfigManager
from aisafetylab.utils import parse_arguments

args = parse_arguments()

if args.config_path is None:
    args.config_path = './configs/gcg.yaml'

config_manager = ConfigManager(config_path=args.config_path) #./configs/gcg.yaml
gcg_manager = GCGMainManager.from_config(config_manager.config)
gcg_manager.attack()
