from aisafetylab.defense.training_defense.utils.args import ModelArguments, TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    """
    Load and configure the model and tokenizer based on provided arguments.
    
    Args:
        model_args (ModelArguments): Arguments related to model configuration
        training_args (TrainingArguments): Arguments related to training setup
    
    Returns:
        tuple: (model, tokenizer) pair
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # Ensure the model is in the correct precision format
    if training_args.fp16:
        model = model.half()
    
    return model, tokenizer