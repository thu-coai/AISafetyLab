from aisafetylab.defense.training_defense.utils.args import ModelArguments, DataArguments, TrainingArguments, GeneratingArguments
from aisafetylab.defense.training_defense.model.load import load_model_and_tokenizer
from transformers import default_data_collator
from aisafetylab.defense.training_defense.train.safe_unlearning.trainer import SafeUnlearningTrainer
from aisafetylab.defense.training_defense.data.dataset import SafetyDataset, create_dataset
import copy

def run_safe_unlearning(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    generating_args: GeneratingArguments
):
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    reference_model, _ = load_model_and_tokenizer(model_args, training_args)
    
    # load dataset 
    train_dataset, eval_dataset = create_dataset(data_args, tokenizer)
    
    # data collator
    data_collator = default_data_collator

    # Initialize our Trainer
    trainer = SafeUnlearningTrainer(
        reference_model=reference_model,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    
    trainer.train(resume_from_checkpoint=False)