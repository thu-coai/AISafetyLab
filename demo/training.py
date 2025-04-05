from typing import Any, Dict, Optional
from aisafetylab.defense.training_defense.train.safe_tuning.workflow import run_safe_tuning
from aisafetylab.defense.training_defense.train.safe_unlearning.workflow import run_safe_unlearning
from aisafetylab.defense.training_defense.train.rlhf.workflow import run_rlhf
from aisafetylab.defense.training_defense.utils.argparser import get_train_args

def run(args: Optional[Dict[str, Any]] = None) -> None:
    """Execute the training workflow based on provided arguments.
    
    Args:
        args (Optional[Dict[str, Any]], optional): Dictionary containing training configuration.
            If None, arguments will be parsed from command line. Defaults to None.
            
    Raises:
        ValueError: If training_args.method is not supported.
    """
    model_args, data_args, training_args, generating_args = get_train_args(args)

    if training_args.method == "safe_tuning":
        run_safe_tuning(model_args, data_args, training_args, generating_args)
    if training_args.method == "safe_unlearning":
        run_safe_unlearning(model_args, data_args, training_args, generating_args)
    elif training_args.method == "rlhf":
        run_rlhf()
    else:
        raise ValueError(f"Unknown Training Method: {training_args.method}.")

def main():
    """Entry point of the script."""
    run()

if __name__ == "__main__":
    main()