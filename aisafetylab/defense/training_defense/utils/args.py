from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments as HfTrainingArguments

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    def __post_init__(self) -> None:
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name or path of the dataset to use."}
    )
    
    cutoff_len: Optional[int] = field(
        default=4096,
        metadata={"help": "The maximum length of input sequences after tokenization."}
    )
    
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    
    max_length: Optional[int] = field(
        default=256,
        metadata={"help": "The maximum length of input sequences after tokenization."}
    )
    
    max_input_length: Optional[int] = field(
        default=4096,
        metadata={"help": "The maximum length of input sequences before tokenization."}
    )
    
    max_output_length: Optional[int] = field(
        default=256,
        metadata={"help": "The maximum length of output sequences before tokenization."}
    )

    def __post_init__(self) -> None:
        if self.dataset is None:
            raise ValueError("Please provide `dataset`.")

@dataclass 
class TrainingArguments(HfTrainingArguments):
    """Arguments pertaining to training configuration."""
    
    method: str = field(
        default="sft",
        metadata={"help": "Training method to use (e.g., 'sft' for safety supervised fine-tuning)."}
    )
    
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether to plot loss during training."}
    )
    
    unlearning_alpha: float = field(
        default=0.0,
        metadata={"help": "The alpha parameter for unlearning."}
    )   
    
    unlearning_beta: float = field(
        default=0.0,
        metadata={"help": "The beta parameter for unlearning."}
    )
    
    unlearning_theta: float = field(
        default=0.0,
        metadata={"help": "The theta parameter for unlearning."}
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.output_dir is None:
            raise ValueError("Please provide `output_dir`.")
        if self.deepspeed is None:
            raise ValueError("Please provide `deepspeed`.")
        if self.method not in ["safe_tuning", "safe_unlearning", "rlhf"]:  # Add other valid methods as needed
            raise ValueError(f"Unknown training method: {self.method}")


@dataclass
class GeneratingArguments:
    """Arguments to control the model's text generation behavior."""
    
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "The maximum numbers of tokens to generate."}
    )
    
    temperature: float = field(
        default=0.7,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    
    top_p: float = field(
        default=0.9,
        metadata={"help": "If set to float < 1, only keep the top p tokens with cumulative probability >= top_p."}
    )
    
    top_k: int = field(
        default=20,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    
    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling; use greedy decoding otherwise."}
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("`temperature` must be in [0.0, 2.0]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("`top_p` must be in (0.0, 1.0]")
        if self.top_k < 0:
            raise ValueError("`top_k` must be a positive integer")
        if self.num_beams < 1:
            raise ValueError("`num_beams` must be at least 1")