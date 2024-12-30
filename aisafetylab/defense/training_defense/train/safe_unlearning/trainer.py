from transformers import Trainer
import torch
import torch
from torch.utils.data import Sampler, Dataset
from typing import Optional
from typing import Iterator, Optional
import numpy as np
import random

class GroupRandomSampler(Sampler[int]):
    r"""Random sampling in a grouped manner

    Args:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source: Dataset) -> None:
        self.data_source = data_source
        self.group_types = sorted(set(item['loss_type'] for item in data_source))

    def __iter__(self) -> Iterator[int]:
        groups = {group_type: [] for group_type in self.group_types}
        for idx, item in enumerate(self.data_source):
            groups[item['loss_type']].append(idx)
            
        # Extract all groups
        # Shuffle the list of groups
        # Generate a unique random seed for each epoch
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        min_length = np.min([len(item) for item in groups.values()])
        group_ids = list(range(min_length))
        group_scale = [len(item) for item in groups.values()] / min_length
        
        random.seed(seed)
        random.shuffle(group_ids)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        normal_indices = torch.randperm(len(groups[0]), generator=generator).tolist()
        
        normal_iter = -1
        for group_id in group_ids:
            for id, type_id in enumerate(self.group_types):
                for i in range(int(group_scale[id])):
                    if type_id == 0:
                        normal_iter += 1
                        normal_iter %= len(groups[0])
                        yield groups[type_id][normal_indices[normal_iter]]
                    else:
                        yield groups[type_id][group_id *int(group_scale[id]) + i]
                    
    def __len__(self) -> int:
        return len(self.data_source)
    
class SafeUnlearningTrainer(Trainer):
    def __init__(self, reference_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args = kwargs.pop('args', None)
        self.alpha = training_args.unlearning_alpha
        self.beta = training_args.unlearning_beta
        self.theta = training_args.unlearning_theta
        device = self.accelerator.device
        self.reference_model = reference_model.to(device)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    def compute_logps(self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
        mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
        per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2, 
                                       index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)).squeeze(2)
        return torch.mul(per_token_logps, mask.to(dtype=torch.bfloat16)).sum(dim=1).to(dtype=torch.float64) / mask.sum(dim=1).to(dtype=torch.float64)
        
    def compute_loss(self, model, inputs):
        types = inputs.pop("loss_type")
        
        labels = inputs.get("labels")
        neg_idxs = (types == 2)
        pos_idxs = (types == 1)
        normal_idxs = (types == 0)
        
        loss, logits = compute_crossentropy_loss(model, inputs, mean=False, return_logits=True)
        
        normal_loss = get_subbatch_loss(loss, normal_idxs, labels)
        if pos_idxs.sum() > 0 and self.theta > 0:
            pos_loss = self.theta * get_subbatch_loss(loss, pos_idxs, labels)
        
            loss_sft = normal_loss + pos_loss
        else:
            loss_sft = normal_loss
        
        if neg_idxs.sum() > 0 and self.alpha > 0:
            attention_mask = inputs['attention_mask']
            prompt_attention_mask = attention_mask.clone()
            prompt_attention_mask[labels != -100] = 0.
            
            log_probs = self.compute_logps(prompt_attention_mask, inputs['input_ids'], attention_mask, logits)
            
            neg_prob = log_probs[neg_idxs]
            
            with torch.no_grad():
                ref_loss, ref_logits = compute_crossentropy_loss(self.reference_model, inputs, mean=False, return_logits=True)
            
            ref_log_probs = self.compute_logps(prompt_attention_mask, inputs['input_ids'], attention_mask, ref_logits)
            ref_neg_prob = ref_log_probs[neg_idxs]
            
            log_delta = -self.beta * (neg_prob - ref_neg_prob)
            unlearn_loss = torch.log(torch.nn.functional.sigmoid(log_delta))
            
            loss = loss_sft - self.alpha * torch.mean(unlearn_loss)
            
        else:
            loss = loss_sft
        
        self.log({'Unlearning Loss': -self.alpha * torch.mean(unlearn_loss).item() if 'unlearn_loss' in locals().keys() else 0.,
                'Normal Loss': normal_loss.item(),
                'Positive Loss': pos_loss.item() if 'pos_loss' in locals().keys() else 0.,
                'Normal prob': -normal_loss.item(),
                'Positive prob': -pos_loss.item() if 'pos_loss' in locals().keys() else 0.,
                'Negative prob': torch.mean(neg_prob).item() if 'neg_prob' in locals().keys() else 0.})   
        
        return loss
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # return RandomSampler(self.train_dataset)
        return GroupRandomSampler(self.train_dataset) # We use this in our experiments, but RandomSampler is also ok

def get_subbatch_loss(loss, idxs, labels):
    loss = loss[idxs]
    labels = labels[idxs]
    shift_labels = labels[..., 1:].contiguous()
    valid_counts = (shift_labels != -100).float().sum(dim=1)
    
    loss = loss.sum() / valid_counts.sum()
    return loss
    
def compute_crossentropy_loss(model, batch, mean=True, return_logits=False, eos_token_id=None):
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
        
    if not return_logits:
        if mean:
            valid_counts = (shift_labels != -100).sum().float()
            loss = loss.sum() / valid_counts
            return loss
        else:
            return loss
    else:
        if mean:
            valid_counts = (shift_labels != -100).sum().float()
            loss = loss.sum() / valid_counts
            return loss, logits
        else:
            return loss, logits
