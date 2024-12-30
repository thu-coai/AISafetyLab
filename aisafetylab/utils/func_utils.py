from time import time
import torch
import random
import wandb
import numpy as np
# from aisafetylab.models import LocalModel
class Timer:

    def __init__(self, t):
        self.last_time = t

    @staticmethod
    def start():
        return Timer(time())

    def end(self):
        t = self.last_time
        self.last_time = time()
        return time() - t


def schedule_n_to_change_fixed(max_n_to_change, it):
    """Piece-wise constant schedule for `n_to_change` (both characters and tokens)"""
    # it = int(it / n_iters * 10000)

    if 0 < it <= 10:
        n_to_change = max_n_to_change
    elif 10 < it <= 25:
        n_to_change = max_n_to_change // 2
    elif 25 < it <= 50:
        n_to_change = max_n_to_change // 4
    elif 50 < it <= 100:
        n_to_change = max_n_to_change // 8
    elif 100 < it <= 500:
        n_to_change = max_n_to_change // 16
    else:
        n_to_change = max_n_to_change // 32

    n_to_change = max(n_to_change, 1)

    return n_to_change


def schedule_n_to_change_prob(max_n_to_change, prob, target_model):
    """Piece-wise constant schedule for `n_to_change` based on the best prob"""
    # it = int(it / n_iters * 10000)

    # need to adjust since llama and r2d2 are harder to attack

    # if not isinstance(target_model, LocalModel):
    #     raise ValueError("The target_model must be an instance of LocalModel.")

    model_name = target_model.model_name

    if 'llama' in model_name.lower():

        if 0 <= prob <= 0.01:
            n_to_change = max_n_to_change // 2
        elif 0.01 < prob <= 0.1:
            n_to_change = max_n_to_change // 4
        elif 0.1 < prob <= 1.0:
            n_to_change = max_n_to_change // 8
        else:
            raise ValueError(f"Wrong prob {prob}")
    elif 'r2d2' in model_name.lower():
        if 0 <= prob <= 0.01:
            n_to_change = max_n_to_change // 2
        elif 0.01 < prob <= 0.1:
            n_to_change = max_n_to_change // 4
        elif 0.1 < prob <= 1.0:
            n_to_change = max_n_to_change // 8
        else:
            raise ValueError(f"Wrong prob {prob}")
    else:
        if 0 <= prob <= 0.01:
            n_to_change = max_n_to_change
        elif 0.01 < prob <= 0.1:
            n_to_change = max_n_to_change // 2
        elif 0.1 < prob <= 1.0:
            n_to_change = max_n_to_change // 4
        else:
            raise ValueError(f"Wrong prob {prob}")
    
    return n_to_change
    n_to_change = max(n_to_change, 1)

    return n_to_change


def extract_logprob(logprob_dict, target_token):
    logprobs = []
    if " " + target_token in logprob_dict:
        logprobs.append(logprob_dict[" " + target_token])
    if target_token in logprob_dict:
        logprobs.append(logprob_dict[target_token])

    if logprobs == []:
        return -np.inf
    else:
        return max(logprobs)


def early_stopping_condition(
    best_logprobs,
    target_model,
    logprob_dict,
    target_token,
    determinstic_jailbreak,
    no_improvement_history=750,
    prob_start=0.02,
    no_improvement_threshold_prob=0.01,
):
    if determinstic_jailbreak and logprob_dict != {}:
        argmax_token = max(logprob_dict, key=logprob_dict.get)
        if argmax_token in [target_token, " " + target_token]:
            return True
        else:
            return False

    if len(best_logprobs) == 0:
        return False

    best_logprob = best_logprobs[-1]
    if no_improvement_history < len(best_logprobs):
        prob_best, prob_history = np.exp(best_logprobs[-1]), np.exp(
            best_logprobs[-no_improvement_history]
        )
        no_sufficient_improvement_condition = (
            prob_best - prob_history < no_improvement_threshold_prob
        )
    else:
        no_sufficient_improvement_condition = False
    if (
        np.exp(best_logprob) > prob_start
        and no_sufficient_improvement_condition
    ):
        return True
    if np.exp(best_logprob) > 0.1:
        return True
    # for all other models
    # note: for GPT models, `best_logprob` is the maximum logprob over the randomness of the model (so it's not the "real" best logprob)
    if np.exp(best_logprob) > 0.4:
        return True
    return False


def get_batch_topK_indices(tensor, topK=10):
    """
    Find the top-K indices for each batch in a tensor.

    Args:
    - tensor (torch.Tensor): The input tensor with shape [batch_size, seq_len, vocab_size].
    - topK (int): The number of top values to select.

    Returns:
    - tuple: A tuple containing:
        - topk_values (torch.Tensor): The top-K values for each batch.
        - original_indices (torch.Tensor): The indices of the top-K values in the original tensor's shape.
    """
    # tensor = -one_hot.grad
    K = topK
    flattened_tensor = tensor.view(tensor.size(0), -1)
    topk_values, topk_indices = torch.topk(flattened_tensor, k=K, dim=1)
    original_indices_row = torch.div(
        topk_indices, tensor.size(2), rounding_mode="floor"
    )
    original_indices_col = topk_indices % tensor.size(2)
    original_indices = torch.stack((original_indices_row, original_indices_col), dim=2)
    return topk_values, original_indices


def get_variants(
    prompt_with_output_encoded,
    adv_indices,
    topk_indices,
    topk_values,
    replace_size,
    beam_size,
):
    """
    Generate variants of a given tensor by substituting tokens with top-K alternatives.

    Args:
    - prompt_with_output_encoded (torch.Tensor): Tensor containing encoded prompts with output tokens.
    - adv_indices (list): Indices where substitution should occur.
    - topk_indices (torch.Tensor): Top-K token indices for substitution.
    - topk_values (torch.Tensor): Top-K values for each token.
    - replace_size (int): Number of replacements to perform per variant.
    - beam_size (int): Number of variants to generate per batch.

    Returns:
    - torch.Tensor: Tensor containing all generated variants.
    """
    variants_list = []
    tensor = prompt_with_output_encoded[..., adv_indices]
    # print(tensor.shape)
    for batch_idx in range(tensor.size(0)):
        batch_tensor = tensor[batch_idx]
        batch_indices = topk_indices[batch_idx]
        batch_values = topk_values[batch_idx]

        for _ in range(beam_size):
            variant_tensor = batch_tensor.clone()
            # sampled_indices = random.choices(range(len(batch_indices)), weights=batch_values, k=replace_size)
            sampled_indices = random.choices(range(len(batch_indices)), k=replace_size)
            for sampled_index in sampled_indices:
                src_index = batch_indices[sampled_index]
                variant_tensor[src_index[0]] = src_index[1]
            variants_list.append(variant_tensor)

    variants_tensor = torch.stack(variants_list)
    return variants_tensor


### Hierarchical Substitution
def random_topk_substitute(
    signal, original, num_sub_positions=10, topK=20, beam_size=10
):
    """
    Perform hierarchical substitution by replacing tokens with top-K candidates at random positions. Tok-K indices is determined by signal parameter.

    Args:
    - signal (torch.Tensor): The gradient or signal tensor used for substitution, determining top-k candidates.
    - original (torch.Tensor): The original input tensor to be modified, this tensor only contains adv tokens, i.e., prompt_with_output_encoded[..., adv_indices]
    - num_sub_positions (int): Number of positions to substitute.
    - topK (int): Number of top candidates to consider for substitution.
    - beam_size (int): Number of variants to generate per batch.

    Returns:
    - torch.Tensor: Tensor containing all generated variants.
    """
    topk = torch.topk(signal, dim=-1, k=topK)
    topk_indices = topk.indices
    topk_values = topk.values
    variants_tensor = []
    for index, (batch_tensor, batch_topK_indices) in enumerate(
        zip(original, topk_indices)
    ):
        seq_len = batch_tensor.shape[0]
        for _ in range(
            beam_size
        ):  
            sub_positions = random.choices(list(range(seq_len)), k=num_sub_positions)
            batch_tensor = batch_tensor.clone()
            # print(batch_tensor.shape)
            for sub_position in sub_positions:
                candidates = batch_topK_indices[sub_position]
                candidate = random.choice(candidates.tolist())
                batch_tensor[sub_position] = candidate
            variants_tensor.append(batch_tensor)
    return torch.stack(variants_tensor)


def hit_rate_at_n(jb_stat, n):
    jb_sum_at_n = np.sum(jb_stat[:, :n], axis=1)
    return np.where(jb_sum_at_n > 0, 1.0, jb_sum_at_n).mean()

def add_dummy_dim_to_slice(slice_obj):
    # First, check if slice_obj is a single slice or a tuple of slices
    if not isinstance(slice_obj, tuple):
        slice_obj = (slice_obj,)

    # Modify the slice_obj to add a new axis where necessary
    new_slice = []
    for sl in slice_obj:
        # Check if it is a single index (int) and add new axis after this dimension
        if isinstance(sl, int):
            new_slice.append(sl)
            new_slice.append(None)
        else:
            new_slice.append(sl)

    return tuple(new_slice)



def list_avg(_list):
    return sum(_list) / len(_list)

def check_jailbroken(seq, test_prefixes):
    jailbroken_list = [
        all([prefix not in text for prefix in test_prefixes]) for text in seq.text
    ]
    jailbroken_avg = list_avg(jailbroken_list)
    return jailbroken_avg, jailbroken_list




def check_affirmative(seq, affirmative_prefixes):
    affirmative_list = [
        any(
            [
                text[: len(prefix)].lower() == prefix.lower()
                for prefix in affirmative_prefixes
            ]
        )
        for text in seq.text
    ]
    affirmative_avg = list_avg(affirmative_list)
    return affirmative_avg, affirmative_list

def check_success(seq, target_seq):
    success_list = [
        target_seq.text[i].lower() in text.lower() for i, text in enumerate(seq.text)
    ]
    success_avg = list_avg(success_list)
    return success_avg, success_list


class Metrics:
    def __init__(self, prefix=""):
        self.metrics = {}
        self.prefix = prefix

    def log(self, key, value, step=None, log_to_wandb=False):
        key = self.prefix + key
        if key in self.metrics:
            self.metrics[key].append(value)
        else:
            self.metrics[key] = [value]
        if log_to_wandb:
            assert step is not None
            wandb.log(dict({key: value}), step=step)

    def get_combined(self, fn, prefix="", step=None, log_to_wandb=False):
        average_metrics = {}
        for key, values in self.metrics.items():
            average_metrics[f"{prefix}{key}"] = fn(values)
        if log_to_wandb:
            assert step is not None
            wandb.log(average_metrics, step=step)
        return average_metrics

    def get_avg(self, prefix="avg/", step=None, log_to_wandb=False):
        return self.get_combined(
            fn=list_avg, prefix=prefix, step=step, log_to_wandb=log_to_wandb
        )

    def get_max(self, prefix="max/", step=None, log_to_wandb=False):
        return self.get_combined(
            fn=max, prefix=prefix, step=step, log_to_wandb=log_to_wandb
        )

    def get_min(self, prefix="min/", step=None, log_to_wandb=False):
        return self.get_combined(
            fn=min, prefix=prefix, step=step, log_to_wandb=log_to_wandb
        )

    def log_dict(self, metrics_dict, step=None, log_to_wandb=False):
        for key, value in metrics_dict.items():
            self.log(key=key, value=value, step=step, log_to_wandb=log_to_wandb)

    def reset(self):
        self.metrics = {}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__