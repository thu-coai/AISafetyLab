import torch
import gc
from torch import nn

class LogitsFeedback:
    @staticmethod
    @torch.no_grad()
    def get_logits_with_control_tokens(model,
            input_ids,
            test_controls,
            control_slice,
            tokenizer = None,
            return_ids=False):
        pad_tok = -1
        if isinstance(test_controls, torch.Tensor):
            if len(test_controls.shape) == 1:
                test_controls = test_controls.unsqueeze(0)
            test_ids = test_controls.to(model.device)
        elif not isinstance(test_controls, list):
            test_controls = [test_controls]
        elif isinstance(test_controls[0], str):
            max_len = control_slice.stop - control_slice.start
            test_ids = [
                torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
                for control in test_controls
            ]
            pad_tok = 0
            while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
                pad_tok += 1
            nested_ids = torch.nested.nested_tensor(test_ids)
            test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
        else:
            raise ValueError(f"test_controls must be a list of strings or a tensor of token ids, got {type(test_controls)}")
        
        if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
            raise ValueError((
                f"test_controls must have shape "
                f"(n, {control_slice.stop - control_slice.start}), " 
                f"got {test_ids.shape}"
            ))
        
        locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
        ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None

        if return_ids:
            del locs, test_ids ; gc.collect()
            return model(input_ids=ids, attention_mask=attn_mask).logits, ids
        else:
            del locs, test_ids
            logits = model(input_ids=ids, attention_mask=attn_mask).logits
            del ids ; gc.collect()
            return logits