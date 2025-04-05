import torch
import torch.nn.functional as F
import torch.nn as nn
import gc
from types import SimpleNamespace
from .func_utils import Timer
from loguru import logger
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
)

def get_developer(model_name):
    r"""
    get model developer
    """
    developer_dict = {"llama-2": "Meta", "vicuna": "LMSYS"}
    return developer_dict.get(model_name, "ModelKeeper")


def prepare_labels(prompt_with_output_encoded, output_indices, ignore_index=-100):
    """
    Prepare labels by masking out irrelevant token (i.e., non-output tokens) with ignore_index.

    Args:
    - prompt_with_output_encoded (torch.Tensor): The encoded tensor with prompts and outputs.
    - output_indices (list or Tensor): Indices of the output tokens to retain.
    - ignore_index (int): Index to use for masking out irrelevant tokens.

    Returns:
    - torch.Tensor: Labels with only the specified output indices retained.
    """
    labels_all = prompt_with_output_encoded.clone()
    labels = torch.full_like(labels_all, ignore_index)
    labels[..., output_indices] = labels_all[..., output_indices]
    return labels


def prepare_input_embeddings(prompt_with_output_encoded, embedding_matrix, adv_indices):
    """
    Prepare input embeddings by replacing adversarial tokens with one-hot derivable vectors and corresponding embeddings.

    Args:
    - prompt_with_output_encoded (torch.Tensor): Encoded tensor with prompt and output tokens.
    - embedding_matrix (torch.Tensor): Embedding matrix for token embeddings.
    - adv_indices (list or Tensor): Indices of adversarial tokens.

    Returns:
    - tuple: One-hot tensor for adversarial tokens and the corresponding input embeddings.
    """
    num_classes = embedding_matrix.size(0)
    input_embeddings = embedding_matrix[prompt_with_output_encoded]
    adv_token_ids = prompt_with_output_encoded[
        ..., adv_indices
    ]  # batch_size, adv_length

    one_hot = (
        F.one_hot(adv_token_ids, num_classes=num_classes)
        .clone()
        .detach()
        .to(embedding_matrix.dtype)
        .to(embedding_matrix.device)
    )
    one_hot.requires_grad_(True)
    adv_embeddings = torch.matmul(
        one_hot, embedding_matrix
    )  # batch_size, adv_length, embedding_dim
    input_embeddings[..., adv_indices, :] = adv_embeddings  
    return one_hot, input_embeddings


def calculate_tensor_size(tensor):
    """
    Recursively calculate the size of a tensor or a nested tuple/list of tensors.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()
    elif isinstance(tensor, (tuple, list)):
        return sum(calculate_tensor_size(t) for t in tensor)
    else:
        return 0


def forward_with_batches(
    model,
    input_ids=None,
    input_embeddings=None,
    labels=None,
    do_backward=False,
    batch_size=None,
    attention_mask=None,
    **kwargs,
):
    """
    Perform forward pass with automatic batching to avoid OOM errors.

    CAUTION::
    https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408/4
    Such as Training a causal language model from scratch - Hugging Face NLP Course 91
    Where you can clearly see that they highlighted the text over there:

    :warning: Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.
    Therefore, I’d also like to ask the HF team to make it clear in the AutoModelForCausalLM clearly that the auto-regressive models loaded from the transformers library will shift the label_ids automatically in the forward() function, and all we need to feed the data collator or a custom dataloader is just ['input_ids', 'attention_mask', 'label_ids'], and the label_ids should just be a copy of the input_ids and we shouldn’t shift it manually in the data preparation phrase, otherwise, it will cause two token gap between label and logits in the forward function.

    Args:
        model (torch.nn.Module): The model to perform forward pass.
        input_embeddings (torch.Tensor): The input embeddings.
        labels (torch.Tensor): The labels.
        batch_size (int, optional): The batch size. If None, it will be automatically determined based on GPU memory.
        **kwargs: Additional arguments to pass to the model's forward method.

    Returns:
        dict: The combined outputs from all batches.
    """
    # print(input_embeddings.shape)
    device = next(model.parameters()).device
    if input_embeddings is not None:
        input_embeddings = input_embeddings.to(device)
    if labels is not None:
        labels = labels.to(device)

    # Determine batch size if not provided
    if batch_size is None:
        total_memory = torch.cuda.get_device_properties(device).total_memory
        # Estimate the memory required for a single forward pass
        # This is a rough estimate and may need tuning based on your specific model and data
        # print(labels)
        dummy_output = model.forward(
            inputs_embeds=input_embeddings[:1],
            labels=labels[:1],
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True,
        )
        # dummy_output_size = sum(p.numel() * p.element_size() for p in dummy_output.values())
        # dummy_output_size = sum(p.numel() * p.element_size() for p in dummy_output.values() if isinstance(p, torch.Tensor))
        dummy_output_size = sum(calculate_tensor_size(p) for p in dummy_output.values())

        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - reserved_memory
        batch_size = int(available_memory / dummy_output_size / 16)

    # Ensure batch size is at least 1
    batch_size = max(1, batch_size)
    # print(f" Available memory: {available_memory / 1024 ** 3:.2f} GB")
    # print(f" Reserved memory: {reserved_memory / 1024 ** 3:.2f} GB")
    # print(f" Total memory: {total_memory / 1024 ** 3:.2f} GB")
    # print(f"The batch size is {batch_size}")

    # Split input_embeddings and labels into batches
    batch_outputs_logits = []
    batch_outputs_loss = []
    tot_len = (
        input_embeddings.size(0) if input_embeddings is not None else input_ids.size(0)
    )

    for start_idx in range(0, tot_len, batch_size):
        end_idx = start_idx + batch_size
        if input_ids is not None:
            batch_input_ids = input_ids[start_idx:end_idx]
        else:
            batch_input_ids = None
        if input_embeddings is not None:
            batch_input_embeddings = input_embeddings[start_idx:end_idx]
        else:
            batch_input_embeddings = None
        if labels is not None:
            batch_labels = labels[start_idx:end_idx]
        else:
            batch_labels = None
        if attention_mask is not None:
            batch_attention_mask = attention_mask[start_idx:end_idx]
        else:
            batch_attention_mask = None
        # Perform forward pass for the current batch
        # print(batch_input_embeddings.shape)
        # print(batch_labels.shape)
        # timer = Timer.start()
        batch_output = model(
            input_ids=batch_input_ids,
            inputs_embeds=batch_input_embeddings,
            labels=batch_labels,
            attention_mask=batch_attention_mask,
            **kwargs,
        )
        # logger.debug(f"Forward pass time: {timer.end()} seconds")
        # batch_outputs.append(batch_output)
        if do_backward:
            batch_output["loss"].backward()

        timer = Timer.start()
        batch_outputs_logits.append(batch_output["logits"].detach())
        # batch_outputs_loss.append(batch_output['loss'])
        del batch_input_embeddings
        del batch_labels
        del batch_output
        # logger.debug(f'del cost time: {timer.end()} seconds')
        torch.cuda.empty_cache()

        # logger.debug(f'empty cache cost time: {timer.end()} seconds')
        torch.cuda.ipc_collect()
        # logger.debug(f'ipc collect cost time: {timer.end()} seconds')
        gc.collect()
        # logger.debug(f'gc collect cost time: {timer.end()} seconds')

    # return batch_outputs
    # Combine outputs from all batches
    combined_outputs = {}
    combined_outputs["logits"] = torch.cat(
        [batch_output for batch_output in batch_outputs_logits], dim=0
    )
    # combined_outputs['loss'] = torch.stack([batch_output for batch_output in batch_outputs_loss]).mean()

    return combined_outputs


def foward_with_separated_losses(
    model,
    input_embeddings_chunk,
    labels_chunk,
    output_indices,
    attention_mask_chunk,
    past_key_values_chunk=None,
):
    """
    Perform forward pass and compute loss per sequence position.

    Args:
    - model (torch.nn.Module): Model for forward pass.
    - input_embeddings_chunk (torch.Tensor): Embeddings for input.
    - labels_chunk (torch.Tensor): Labels for input.
    - output_indices (list): Indices for output tokens.

    Returns:
    - list: Loss values for each sequence.
    """
    # outputs = forward_with_batches(model, input_embeddings_chunk, labels_chunk, do_backward=False, batch_size=batch_size, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states)

    print("output_indices", output_indices)
    past_input_length = (
        past_key_values_chunk[0][0].shape[2] if past_key_values_chunk is not None else 0
    )
    output_indices = [i - past_input_length for i in output_indices]
    print("past_innput_length", past_input_length)
    print("output_indices", output_indices)
    print("labels_chunk.shape", labels_chunk.shape)
    labels_chunk = labels_chunk[..., past_input_length:]
    print("labels_chunk.shape", labels_chunk.shape)

    outputs = model.forward(
        inputs_embeds=input_embeddings_chunk,
        attention_mask=attention_mask_chunk,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        past_key_values=past_key_values_chunk,
    )
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]

    shift_logits = logits[..., :-1, :].contiguous().transpose(1, 2)
    shift_labels = labels_chunk[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels.to(shift_logits.device))
    loss = loss[..., output_indices].mean(-1)

    del outputs
    # del logits
    del shift_logits
    del shift_labels
    torch.cuda.empty_cache()
    gc.collect()

    return SimpleNamespace(
        loss=loss,
        past_key_values=past_key_values,
        logits=logits,
    )


def foward_with_separated_losses_advprompter(
    model,
    input_embeddings_chunk,
    labels_chunk,
    output_indices,
    attention_mask_chunk,
    past_key_values_chunk=None,
    suffix_indices=None,
):
    """
    Perform forward pass and compute loss per sequence position.

    Args:
    - model (torch.nn.Module): Model for forward pass.
    - input_embeddings_chunk (torch.Tensor): Embeddings for input.
    - labels_chunk (torch.Tensor): Labels for input.
    - output_indices (list): Indices for output tokens.

    Returns:
    - list: Loss values for each sequence.
    """
    # outputs = forward_with_batches(model, input_embeddings_chunk, labels_chunk, do_backward=False, batch_size=batch_size, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states)

    print("output_indices", output_indices)
    past_input_length = (
        past_key_values_chunk[0][0].shape[2] if past_key_values_chunk is not None else 0
    )
    output_indices = [i - past_input_length for i in output_indices]
    print("past_innput_length", past_input_length)
    print("output_indices", output_indices)
    print("labels_chunk.shape", labels_chunk.shape)
    labels_chunk = labels_chunk[..., past_input_length:]
    print("labels_chunk.shape", labels_chunk.shape)

    outputs = model.forward(
        inputs_embeds=input_embeddings_chunk,
        attention_mask=attention_mask_chunk,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        past_key_values=past_key_values_chunk,
    )
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]

    shift_logits = logits[..., :-1, :].contiguous().transpose(1, 2)
    shift_labels = labels_chunk[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits, shift_labels.to(shift_logits.device))
    suffix_loss = (
        token_losses[..., suffix_indices].mean(-1) if suffix_indices is not None else 0
    )

    # compute weighting \gamma_t = 1/t for each token
    sequence_length = len(output_indices)
    weights = torch.tensor(
        [1.0 / (t + 1) for t in range(sequence_length)], device=token_losses.device
    )
    output_loss = (token_losses[..., output_indices] * weights).mean(-1)

    lambda_reg = 1.0
    loss = output_loss + lambda_reg * suffix_loss

    del outputs
    # del logits
    del shift_logits
    del shift_labels
    torch.cuda.empty_cache()
    gc.collect()

    return SimpleNamespace(
        loss=loss,
        past_key_values=past_key_values,
        logits=logits,
    )


def beam_sample_next_token(
    model, num_samples_per_beam, input_ids, attention_mask, temperature=1.0
):
    model_forward_res = model(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )

    next_token_logits = model_forward_res.logits[:, -1, :]
    next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)

    next_token_ids = torch.multinomial(
        next_token_probs, num_samples=num_samples_per_beam
    )

    input_ids = input_ids.repeat_interleave(num_samples_per_beam, dim=0)
    beam_expanded = torch.cat(
        [input_ids, next_token_ids.flatten().unsqueeze(-1)], dim=-1
    )
    return SimpleNamespace(
        next_token_ids=next_token_ids,
        beam_expanded=beam_expanded,
    )


def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif (
        isinstance(model, LlamaForCausalLM)
        or isinstance(model, MistralForCausalLM)
        or isinstance(model, GemmaForCausalLM)
        or isinstance(model, Qwen2ForCausalLM)
    ):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    elif isinstance(model, FalconForCausalLM):
        return model.transformer.word_embeddings.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        # raise ValueError(f"Unknown model type: {type(model)}")
        return model.model.embed_tokens(input_ids)

def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)

def apply_repetition_penalty(logits, prev_ids, penalty):
    _logits = torch.gather(input=logits, dim=1, index=prev_ids)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous
    # token probability
    _logits = torch.where(_logits < 0, _logits * penalty, _logits / penalty)

    logits_penalized = torch.scatter(input=logits, dim=1, index=prev_ids, src=_logits)
    return logits_penalized

def compute_perplexity(id_seq, likelihood_seq):
    logprobs = torch.gather(
        likelihood_seq.logprobs, dim=2, index=id_seq.ids.unsqueeze(2)
    ).squeeze(2)

    perplexity_per_token_masked = torch.exp(-logprobs) * id_seq.mask
    perplexity = torch.exp(
        -torch.sum(logprobs * id_seq.mask, dim=1)
        / (torch.sum(id_seq.mask, dim=1) + 1e-8)
    )

    return perplexity, perplexity_per_token_masked


class ReturnStruct:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def clone(self):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.clone()
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

    def detach(self):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.detach()
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

    def _detach(self):
        for k, v in self.__dict__.items():
            try:
                v._detach()
            except:
                pass

    def to(self, device):
        new_kwargs = {}
        for k, v in self.__dict__.items():
            try:
                new_kwargs[k] = v.to(device)
            except:
                new_kwargs[k] = v
        return ReturnStruct(**new_kwargs)

def ce_loss(pred_seq, target_seq, hard_labels, reweight_loss=False, **kwargs):
    if hard_labels:
        loss = F.cross_entropy(
            pred_seq.logits.transpose(1, 2), target_seq.ids, reduction="none", **kwargs
        )
    else:
        loss = F.cross_entropy(
            pred_seq.logits.transpose(1, 2),
            target_seq.probs.transpose(1, 2),
            reduction="none",
            **kwargs,
        )
    if reweight_loss:
        factor = torch.arange(loss.shape[1], dtype=loss.dtype, device=loss.device) + 1
        loss = loss / factor[None, :]
    return loss


def loss_seqs(pred_seq, target_seq, **loss_params):
    if torch.isnan(pred_seq.logits).any():
        raise ValueError(f"Nan in logits: {pred_seq.logits}")
    _loss = ce_loss(pred_seq, target_seq, **loss_params)
    mask = target_seq.mask
    loss_masked = _loss * mask
    loss_batch = torch.sum(loss_masked, dim=1) / (mask.sum(dim=1) + 1e-10)
    loss = loss_batch.mean()

    ce_return = ReturnStruct(
        loss=loss,
        loss_masked=loss_masked,
        loss_batch=loss_batch,
        pred=pred_seq,
        label=target_seq,
    )
    return ce_return


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f" trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

from functools import wraps
def autocast_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if "cuda" in self.device:
            device = "cuda"
        else:
            device = self.device

        with torch.autocast(device):
            return func(self, *args, **kwargs)

    return wrapper


def get_total_allocated_memory():
    devices = torch.cuda.device_count()
    total_allocated_memory = 0
    for i in range(devices):
        total_allocated_memory += torch.cuda.memory_allocated(f"cuda:{i}")
    return total_allocated_memory / 1e9


from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    FalconForCausalLM,
    GemmaForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
)
def llm_loader(checkpoint, dtype, device, freeze=False, lora_checkpoint=None, lora_config=None, verbose=False):
    logger.info(
        f" Loading model from {checkpoint}...",
    )
    mem_before = get_total_allocated_memory()
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Cannot load model with dtype {dtype}")
    if checkpoint == "stas/tiny-random-llama-2":
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            padding_side="right",
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=dtype
        ).to(device)
    else:
        use_fast = "pythia" in checkpoint
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            model_max_length=1024,
            padding_side="right",
            use_fast=use_fast,
            legacy=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            device_map=device,
        )
    mem_after = get_total_allocated_memory()

    if verbose:
        logger.info(f" Loaded model: {model}")
    logger.info(
        f" Mem usage model: {mem_after - mem_before:.2f} GB | Total Mem usage: {get_total_allocated_memory():.2f} GB",
    )

    embedding_matrix = get_embedding_matrix(model).to(device)

    if freeze:
        logger.info(" Freezing model...")
        for k, v in model.named_parameters():
            v.requires_grad = False

    if (lora_checkpoint is not None) or (lora_config is not None):
        if lora_checkpoint is not None:
            logger.info(
                f" Loading LoRA checkpoint: {lora_checkpoint}",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_checkpoint,
                is_trainable=not freeze,
            )
        else:
            logger.info(" Transforming to LoRA model...")
            lora_config_dct = dict(lora_config)
            lora_config_dct["target_modules"] = [
                m for m in lora_config["target_modules"]
            ]
            lora_config = LoraConfig(**lora_config_dct)
            # model = get_peft_model(model, lora_config)
            model.add_adapter(lora_config, adapter_name='lora')

            model.enable_adapters()

    print_trainable_parameters(model)
    return model, tokenizer, embedding_matrix
