import numpy as np
import torch
import random
import gc
from types import SimpleNamespace

class RouletteWheelSelection:
    @staticmethod
    def select_roulette_wheel(data_list, score_list, num_selected):
        r"""
        apply roulette_wheel_selection on data_list
        """
        selection_probs = np.exp(score_list - np.max(score_list))
        selection_probs = selection_probs / selection_probs.sum()
        selected_indices = np.random.choice(len(data_list), size=num_selected, p=selection_probs, replace=True)
        selected_data = [data_list[i] for i in selected_indices]
        selected_data_scores = [score_list[i] for i in selected_indices]
        return selected_data, selected_data_scores

    @staticmethod
    def word_roulette_wheel_selection(word, word_scores):
        if not word_scores:
            return word
        min_score = min(word_scores.values())
        adjusted_scores = {k: v - min_score for k, v in word_scores.items()}
        total_score = sum(adjusted_scores.values())
        pick = random.uniform(0, total_score)
        current_score = 0
        for synonym, score in adjusted_scores.items():
            current_score += score
            if current_score > pick:
                if word.istitle():
                    return synonym.title()
                else:
                    return synonym


from types import SimpleNamespace
import torch
import gc
import torch.nn.functional as F
class TopKSelector:
    @staticmethod
    def select_top_k(prompt_with_output_encoded, loss, batch_size, tokenizer):
        k = min(batch_size, len(loss))
        topk_indices = torch.topk(-loss, k=k).indices
        prompt_with_output_encoded = prompt_with_output_encoded[topk_indices]
        losses = loss[topk_indices].tolist()
        adv_strings = tokenizer.batch_decode(prompt_with_output_encoded)
        torch.cuda.empty_cache()
        gc.collect()

        return SimpleNamespace(
            prompt_with_output_encoded = prompt_with_output_encoded,
            losses_chosen_candidates = losses,
            adv_strings_chosen_candidates = adv_strings,
            topk_indices = topk_indices,
        )

    @staticmethod
    def sample_top_k(prompt_with_output_encoded, loss, batch_size, tokenizer, temperature=1.0):
        # Calculate softmax with temperature applied
        scaled_loss = -loss / temperature  # Scale the loss with temperature
        probabilities = F.softmax(scaled_loss, dim=0)
        
        # Sample indices based on calculated probabilities
        k = min(batch_size, len(loss))
        sampled_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
        
        prompt_with_output_encoded = prompt_with_output_encoded[sampled_indices]
        losses = loss[sampled_indices].tolist()
        adv_strings = tokenizer.batch_decode(prompt_with_output_encoded)
        torch.cuda.empty_cache()
        gc.collect()

        return SimpleNamespace(
            prompt_with_output_encoded=prompt_with_output_encoded,
            losses_chosen_candidates=losses,
            adv_strings_chosen_candidates=adv_strings,
            topk_indices=sampled_indices,
        )
