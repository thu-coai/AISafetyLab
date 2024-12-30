from transformers import Trainer
import torch

class SafeTuningTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, *args, **kwargs):
        
        # safe tuning do not use loss_type
        types = inputs.pop("loss_type")
        loss = compute_crossentropy_loss(model, inputs)
        
        return loss
    
def compute_crossentropy_loss(model, batch):
    outputs = model(**batch)
    
    logits = outputs.logits
    labels = batch.get("labels")
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    )

    loss = loss.view(shift_logits.size(0), -1)
        
    return loss.mean()