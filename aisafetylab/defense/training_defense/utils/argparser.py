from typing import Any, Dict, Optional, Tuple
from transformers import HfArgumentParser
import sys
import os
from aisafetylab.defense.training_defense.utils.args import ModelArguments, DataArguments, TrainingArguments, GeneratingArguments
from loguru import logger

def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    """Parse arguments from dictionary, yaml/json file, or command line."""
    if args is not None:
        return parser.parse_dict(args)

    # Parse from yaml/json file if provided
    if sys.argv[-1].endswith(".yaml") or sys.argv[-1].endswith(".yml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[-1]))
    if sys.argv[-1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[-1]))

    # Parse from command line arguments
    return parser.parse_args_into_dataclasses()

def get_train_args(args: Optional[Dict[str, Any]] = None) -> Tuple[ModelArguments, DataArguments, TrainingArguments, GeneratingArguments]:
    """Parse and validate training arguments."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GeneratingArguments))
    model_args, data_args, training_args, generating_args = _parse_args(parser, args)
    
    # Validate required arguments
    if model_args.model_name_or_path is None:
        raise ValueError("Please provide `model_name_or_path`.")
    if data_args.dataset is None:
        raise ValueError("Please provide `dataset`.")
    if training_args.output_dir is None:
        raise ValueError("Please provide `output_dir`.")

    # Validate training method
    if training_args.method not in ["safe_tuning", "safe_unlearning", "rlhf"]:
        raise ValueError(f"Unknown training method: {training_args.method}")

    return model_args, data_args, training_args, generating_args
