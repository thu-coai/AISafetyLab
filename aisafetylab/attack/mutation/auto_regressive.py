from aisafetylab.attack.mutation import BaseMutation
import torch
from types import SimpleNamespace

class AutoRegressiveMutate(BaseMutation):
    @staticmethod 
    def beam_sample_next_token(model, num_samples_per_beam, input_ids, attention_mask, temperature=1.0, next_token_logits = None, past_key_values=None):
    
        if next_token_logits is None:

            past_input_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            input_ids_new = input_ids[...,past_input_length:]
            model_forward_res = model.forward(
                input_ids = input_ids_new,
                attention_mask = attention_mask,
                past_key_values = past_key_values,
            )
            past_key_values = model_forward_res.past_key_values
            next_token_logits = model_forward_res.logits[:, -1, :]

        next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token_ids = torch.multinomial(next_token_probs, num_samples=num_samples_per_beam)

        input_ids = input_ids.repeat_interleave(num_samples_per_beam, dim=0)
        
        beam_expanded = torch.cat([input_ids, next_token_ids.flatten().unsqueeze(-1)], dim=-1)
        attention_mask_expanded = attention_mask.repeat_interleave(num_samples_per_beam, dim=0)
        attention_mask_expanded = torch.cat([attention_mask_expanded, torch.ones(attention_mask_expanded.shape[0],1)], dim=-1)
        past_key_values_expanded = [[key_value.repeat_interleave(num_samples_per_beam, dim=0) for key_value in layer] for layer in past_key_values]


        return SimpleNamespace(
            next_token_ids = next_token_ids,
            beam_expanded = beam_expanded,
            attention_mask_expanded = attention_mask_expanded,
            past_key_values_expanded = past_key_values_expanded,
        )