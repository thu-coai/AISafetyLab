from aisafetylab.utils import prepare_labels
from aisafetylab.utils import foward_with_separated_losses
from tqdm import trange
import torch
import gc
from types import SimpleNamespace
from torch import nn

class LossFeedback:
    @staticmethod
    def get_sliced_loss(logits, ids,target_slice, loss_slice=None):
        if loss_slice is None:
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        crit = nn.CrossEntropyLoss(reduction='none')
        loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
        return loss
    @staticmethod
    def compute_separated_losses(model, prompt_with_output_encoded, output_indices, attention_mask=None, past_key_values=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_with_output_encoded, device=model.device).long()
        past_key_values_in = past_key_values
        past_input_length = past_key_values_in[0][0].shape[2] if past_key_values_in is not None else 0
        print('past_input_length', past_input_length)
        past_key_values = []
        input_embeddings = model.get_input_embeddings().weight[prompt_with_output_encoded[..., past_input_length:]]
        print('input_embeddings.shape', input_embeddings.shape)
        # labels = prepare_labels(prompt_with_output_encoded, output_indices, ignore_index=-100)
        labels = prompt_with_output_encoded

        loss = []
        logits = []
        for chunk in trange(0, len(input_embeddings), 100, desc="Processing chunks"):
            input_chunk = input_embeddings[chunk:chunk + 100]
            labels_chunk = labels[chunk:chunk + 100]
            attention_mask_chunk = attention_mask[chunk: chunk+100]
            past_key_values_chunk = [[key_value[chunk: chunk+100] for key_value in layer] for layer in past_key_values_in  ] if past_key_values_in is not None else None
            # print("past_key_values_chunk[0][0].shape", past_key_values_chunk[0][0].shape)
            res = foward_with_separated_losses(model, input_chunk, labels_chunk, output_indices,attention_mask_chunk, past_key_values_chunk)
            loss_chunk = res.loss.tolist()
            past_key_values_chunk = res.past_key_values
            logits_chunk = res.logits
            logits.append(logits_chunk)
            past_key_values.append(past_key_values_chunk)
            loss.extend(loss_chunk)
            torch.cuda.empty_cache()
            gc.collect()

        
        new_past_key_values = []
        for layer_idx in range(len(past_key_values[0])):
            layer = []
            for key_value_idx in range(len(past_key_values[0][0])):
                layer.append(torch.cat([past_key_values[chunk_idx][layer_idx][key_value_idx] for chunk_idx in range(len(past_key_values))]))
            new_past_key_values.append(layer)

        
        return SimpleNamespace(
            loss = torch.tensor(loss),
            past_key_values = new_past_key_values,
            logits = torch.cat(logits, dim=0)
        )
    


class LossFeedbackAdvprompter:
    @staticmethod
    def compute_separated_losses_advprompter(model, prompt_with_output_encoded, output_indices, attention_mask, past_key_values=None, suffix_indices=None):
        past_key_values_in = past_key_values
        past_input_length = past_key_values_in[0][0].shape[2] if past_key_values_in is not None else 0
        print('past_input_length', past_input_length)
        past_key_values = []
        input_embeddings = model.get_input_embeddings().weight[prompt_with_output_encoded[..., past_input_length:]]
        print('input_embeddings.shape', input_embeddings.shape)
        # labels = prepare_labels(prompt_with_output_encoded, output_indices, ignore_index=-100)
        labels = prompt_with_output_encoded

        loss = []
        logits = []
        for chunk in trange(0, len(input_embeddings), 100, desc="Processing chunks"):
            input_chunk = input_embeddings[chunk:chunk + 100]
            labels_chunk = labels[chunk:chunk + 100]
            attention_mask_chunk = attention_mask[chunk: chunk+100]
            past_key_values_chunk = [[key_value[chunk: chunk+100] for key_value in layer] for layer in past_key_values_in  ] if past_key_values_in is not None else None
            # print("past_key_values_chunk[0][0].shape", past_key_values_chunk[0][0].shape)
            res = foward_with_separated_losses_advprompter(model, input_chunk, labels_chunk, output_indices,attention_mask_chunk, past_key_values_chunk, suffix_indices)
            loss_chunk = res.loss.tolist()
            past_key_values_chunk = res.past_key_values
            logits_chunk = res.logits
            logits.append(logits_chunk)
            past_key_values.append(past_key_values_chunk)
            loss.extend(loss_chunk)
            torch.cuda.empty_cache()
            gc.collect()

        
        new_past_key_values = []
        for layer_idx in range(len(past_key_values[0])):
            layer = []
            for key_value_idx in range(len(past_key_values[0][0])):
                layer.append(torch.cat([past_key_values[chunk_idx][layer_idx][key_value_idx] for chunk_idx in range(len(past_key_values))]))
            new_past_key_values.append(layer)

        
        return SimpleNamespace(
            loss = torch.tensor(loss),
            past_key_values = new_past_key_values,
            logits = torch.cat(logits, dim=0)
        )