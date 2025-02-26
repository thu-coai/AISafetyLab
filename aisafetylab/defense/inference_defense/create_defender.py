from aisafetylab.defense.inference_defense.preprocess_defenders import (
    SelfReminderDefender,
    PromptGuardDefender,
    ICDefender,
    PPLDefender,
    ParaphraseDefender,
    GoalPrioritizationDefender,
)
from aisafetylab.defense.inference_defense.intraprocess_defenders import (
    RobustAlignDefender,
    EraseCheckDefender,
    SmoothLLMDefender,
    DRODefender,
    SafeDecodingDefender,
)
from aisafetylab.defense.inference_defense.postprocess_defenders import (
    SelfExamDefender,
    AlignerDefender,
    PardenDefender,
)
import yaml
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
import torch

def create_defender(defender_type, **kwargs):
    """
    Factory function to create defender instances.

    Args:
        defender_type (str): The type of defender to create.
        **kwargs: Additional keyword arguments for defender initialization.

    Returns:
        Defender: An instance of a defender subclass.
    """
    defender_classes = {
        # Preprocess defenders
        'SelfReminder': SelfReminderDefender,
        'GoalPrioritization': GoalPrioritizationDefender,
        'PromptGuard': PromptGuardDefender,
        'ICD': ICDefender,
        'PPL': PPLDefender,
        'Paraphrase': ParaphraseDefender,

        # IntraProcess defenders
        'RobustAlign': RobustAlignDefender,
        'EraseCheck': EraseCheckDefender,
        'SmoothLLM': SmoothLLMDefender,
        'SafeDecoding': SafeDecodingDefender,
        'DRO': DRODefender,

        # Postprocess defenders
        'SelfExam': SelfExamDefender,
        'Aligner': AlignerDefender,
        'PARDEN': PardenDefender,
        

        # Additional defenders can be added here
    }

    try:
        defender_class = defender_classes[defender_type]
        return defender_class(**kwargs)
    except KeyError:
        raise ValueError(f"Unknown defender type: {defender_type}, Select From: {defender_classes.keys()}")
    except TypeError as e:
        raise ValueError(f"Error creating defender: {e}")
    
def create_defender_from_yaml(yaml_path):
    """
    Creates a defender instance based on the configuration provided in a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        Defender: An instance of a defender subclass.
    """
    # Load YAML configuration
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    # Extract defender type
    defender_type = config.get('defender_type')
    if not defender_type:
        raise ValueError("YAML configuration must include 'defender_type'.")

    defender_params = {k: v for k, v in config.items() if k != 'defender_type'}

    # Initialize HuggingFace model and tokenizer if 'model_name' is provided
    if 'model' in defender_params:
        model_name = defender_params['model']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            if defender_type == "PromptGuard":
                model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
            else: 
                model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        except Exception as e:
            raise ValueError(f"Error loading model '{model_name}': {e}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise ValueError(f"Error loading tokenizer for model '{model_name}': {e}")
        
        defender_params['model'] = model
        defender_params['tokenizer'] = tokenizer
        
    return create_defender(defender_type, **defender_params)