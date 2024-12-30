import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from aisafetylab.defense.training_defense.train.rlhf.trainer import SafeRLHFTrainer
from datasets import load_dataset


def load_config(config_path="AISafetyLab/aisafetylab/defense/training_defense/examples/safe_rlhf/default_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def prepare_datasets(train_path, eval_path):
    train_dataset = load_dataset("json", data_files={"train": train_path})["train"]
    eval_dataset = load_dataset("json", data_files={"eval": eval_path})["eval"]
    return train_dataset, eval_dataset


def run_rlhf(config, task_type):
    print(f"Running task: {task_type}")
    model_class = AutoModelForCausalLM  # Adjust if using specific models for Reward/Cost
    model = model_class.from_pretrained(config[f"{task_type}_model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(config[f"{task_type}_model_name_or_path"])

    train_dataset, eval_dataset = prepare_datasets(
        config[f"{task_type}_train_dataset"], config[f"{task_type}_eval_dataset"]
    )

    training_args = TrainingArguments(
        output_dir=config[f"{task_type}_output_dir"],
        overwrite_output_dir=config["overwrite_output_dir"],
        num_train_epochs=config[f"{task_type}_num_train_epochs"],
        per_device_train_batch_size=config[f"{task_type}_per_device_train_batch_size"],
        gradient_accumulation_steps=config[f"{task_type}_gradient_accumulation_steps"],
        learning_rate=config[f"{task_type}_learning_rate"],
        evaluation_strategy=config.get(f"{task_type}_eval_strategy", "epoch"),
        save_steps=config.get(f"{task_type}_save_steps", 1000),
        logging_steps=config.get(f"{task_type}_logging_steps", 10),
        bf16=config.get("bf16", False),
        tf32=config.get("tf32", False),
        report_to="none",
    )

    trainer = SafeRLHFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        method=task_type,
    )

    trainer.train()


def main(config_path="AISafetyLab/aisafetylab/defense/training_defense/examples/safe_rlhf/default_config.yaml"):
    config = load_config(config_path)
    for task_type in ["sft", "reward", "cost", "ppo"]:
        run_rlhf(config, task_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Safe RLHF workflow.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="default_config.yaml",
        help="Path to the training configuration file.",
    )
    args = parser.parse_args()

    main(config_path=args.config_path)
