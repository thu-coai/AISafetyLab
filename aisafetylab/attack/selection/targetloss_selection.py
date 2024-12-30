import torch
import gc
from .base_selection import BaseSelection
from aisafetylab.utils import forward_with_batches
from loguru import logger

class TargetLossSelection(BaseSelection):
    def __init__(self):
        super().__init__()

    def select(self, model, input_ids_list, loss_slices, pad_token_id, batch_size=None):
        device = model.device
        max_input_length = max([ids.size(0) for ids in input_ids_list])
        padded_input_ids_list = []
        for ids in input_ids_list:
            pad_length = max_input_length - ids.size(0)
            padded_ids = torch.cat([ids, torch.full((pad_length,), pad_token_id, device=device)], dim=0)
            padded_input_ids_list.append(padded_ids)
        input_ids_tensor = torch.stack(padded_input_ids_list, dim=0)
        # FIXME: maybe it is better to obtain attention mask according to original seq length
        attn_mask = (input_ids_tensor != pad_token_id).type(input_ids_tensor.dtype).to(device)
        if batch_size is None:
            batch_size = 16
            
        logger.debug(f'max input length: {max_input_length}')
        
        logits = forward_with_batches(model, input_ids=input_ids_tensor, attention_mask=attn_mask, batch_size=batch_size)['logits']
        
        losses = []
        crit = torch.nn.CrossEntropyLoss(reduction='mean')
        for idx, loss_slice in enumerate(loss_slices):
            logits_slice = logits[idx, loss_slice, :].unsqueeze(0).transpose(1, 2)
            target_slice = slice(loss_slice.start + 1, loss_slice.stop + 1)
            targets = input_ids_tensor[idx, target_slice].unsqueeze(0)
            loss = crit(logits_slice, targets)
            losses.append(loss)

        del input_ids_tensor, attn_mask
        gc.collect()
        losses = torch.stack(losses)

        score_list = losses.cpu().numpy().tolist()
        best_id = losses.argmin()
        current_loss = losses[best_id]

        return score_list, current_loss, best_id
    