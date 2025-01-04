# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the original repo by Facebook.

import math

import torch
from tqdm import tqdm

import wandb
from aisafetylab.utils import (
    add_dummy_dim_to_slice,
    dotdict,
    expand_for_broadcast_tensor,
    expand_for_broadcast_list,
)


def msg_to_seq(msg, tokenizer, device, context=None):
    if isinstance(msg, str):
        if msg[0] == "{" and msg[-1] == "}":
            key = msg[1:-1]
            if key in context:
                msg = context[key]
            else:
                raise ValueError(f"Key {key} not found in context.")

    if isinstance(msg, Seq):
        seq = msg.to(device)
    elif isinstance(msg, str):
        seq = Seq(text=[msg], tokenizer=tokenizer, device=device)
    elif isinstance(msg, EmptySeq):
        seq = msg
    else:
        raise ValueError(f"Msg should be Seq or string. Got {msg} ({type(msg)}).")
    return seq


def stack_seqs(list_of_seqs):
    assert all([seq is not None for seq in list_of_seqs])

    dtypes = [seq.dtype for seq in list_of_seqs]
    assert len(set(dtypes)) <= 1

    seq_lens = [seq.seq_len for seq in list_of_seqs]
    assert len(set(seq_lens)) <= 1 or dtypes[0] == "text"

    tokenizers = [seq.tokenizer for seq in list_of_seqs]
    assert len(set(tokenizers)) <= 1

    devices = [seq.device for seq in list_of_seqs]
    assert len(set(devices)) <= 1

    if dtypes[0] == "logits":
        logits = torch.cat([seq.logits for seq in list_of_seqs], dim=0)
        mask = torch.cat([seq.mask for seq in list_of_seqs], dim=0)
        stacked_seqs = Seq(
            logits=logits, mask=mask, tokenizer=tokenizers[0], device=devices[0]
        )
    elif dtypes[0] == "probs":
        probs = torch.cat([seq.probs for seq in list_of_seqs], dim=0)
        mask = torch.cat([seq.mask for seq in list_of_seqs], dim=0)
        stacked_seqs = Seq(
            probs=probs, mask=mask, tokenizer=tokenizers[0], device=devices[0]
        )
    elif dtypes[0] == "ids":
        ids = torch.cat([seq.ids for seq in list_of_seqs], dim=0)
        mask = torch.cat([seq.mask for seq in list_of_seqs], dim=0)
        stacked_seqs = Seq(
            ids=ids, mask=mask, tokenizer=tokenizers[0], device=devices[0]
        )
    elif dtypes[0] == "text":
        text = [item for seq in list_of_seqs for item in seq.text]
        stacked_seqs = Seq(text=text, tokenizer=tokenizers[0], device=devices[0])
    else:
        raise RuntimeError("No data to stack.")
    return stacked_seqs





class MergedSeq:
    def __init__(self, seqs):
        assert all([seq is not None for seq in seqs])
        self._seqs = [seq for seq in seqs if not seq.is_empty]
        self.tokenizer = self._seqs[0].tokenizer
        self.device = self._seqs[0].device
        assert all([seq.tokenizer == self.tokenizer for seq in self._seqs])
        assert all([seq.device == self.device for seq in self._seqs])

    @property
    def logits(self):
        logits_list = expand_for_broadcast_tensor(
            [seq.logits for seq in self._seqs], dim=0
        )
        logits = torch.cat(logits_list, dim=1)
        return logits

    @property
    def probs(self):
        probs_list = expand_for_broadcast_tensor(
            [seq.probs for seq in self._seqs], dim=0
        )
        probs = torch.cat(probs_list, dim=1)
        return probs

    @property
    def ids(self):
        ids_list = expand_for_broadcast_tensor([seq.ids for seq in self._seqs], dim=0)
        ids = torch.cat(ids_list, dim=1)
        return ids

    @property
    def text(self):
        text_list = expand_for_broadcast_list([seq.text for seq in self._seqs])
        separator = ""
        text = []
        for i in range(len(text_list[0])):
            text.append(
                separator.join([text_list[j][i] for j in range(len(text_list))])
            )
        return text

    @property
    def mask(self):
        mask_list = expand_for_broadcast_tensor([seq.mask for seq in self._seqs], dim=0)
        mask = torch.cat(mask_list, dim=1)
        return mask

    @property
    def logprobs(self):
        return self.to_seq(merge_dtype="logits").logprobs

    # derived properties
    def get_embed(self, embedding_matrix):
        embeds_list = expand_for_broadcast_tensor(
            [seq.get_embed(embedding_matrix) for seq in self._seqs],
            dim=0,
        )
        embeds = torch.cat(embeds_list, dim=1)
        return embeds

    def get_entropy(self, average=True):
        entropies_list = expand_for_broadcast_tensor(
            [seq.get_entropy(average=False) for seq in self._seqs],
            dim=0,
        )
        normalized_entropy_per_token = torch.cat(entropies_list, dim=1)
        if average:
            entropy_batch = torch.sum(
                normalized_entropy_per_token * self.mask, dim=1
            ) / (torch.sum(self.mask, dim=1) + self.eps)
            entropy = torch.mean(entropy_batch)
        else:
            entropy = normalized_entropy_per_token
        return entropy

    @property
    def bs(self):
        batch_sizes = [seq.bs for seq in self._seqs]
        bs = max(batch_sizes)
        assert all([batch_size == bs or batch_size == 1 for batch_size in batch_sizes])
        return bs

    @property
    def seq_len(self):
        seq_len = sum([seq.seq_len for seq in self._seqs])
        return seq_len

    def __len__(self):
        raise NotImplementedError

    @property
    def is_hard(self):
        is_hard_list = [seq.is_hard for seq in self._seqs]
        is_hard = all(is_hard_list)
        return is_hard

    def to_html(self, colors=None, normalize=False, color_scheme=1):
        if colors is None:
            colors = self.get_entropy(average=False)
        html = self.to_seq(merge_dtype="ids").to_html(
            colors=colors, normalize=normalize, color_scheme=color_scheme
        )
        return html

    def to_seq(self, merge_dtype):
        if merge_dtype == "ids":
            seq = Seq(
                ids=self.ids,
                mask=self.mask,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif merge_dtype == "logits":
            seq = Seq(
                logits=self.logits,
                mask=self.mask,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif merge_dtype == "probs":
            seq = Seq(
                probs=self.probs,
                mask=self.mask,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif merge_dtype == "text":
            seq = Seq(
                text=self.text,
                mask=self.mask,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid merge_dtype: {merge_dtype}")
        return seq

    def clone(self):
        return MergedSeq(seqs=[seq.clone() for seq in self._seqs])

    def detach(self):
        return MergedSeq(seqs=[seq.detach() for seq in self._seqs])

    def repeat_interleave(self, times, dim=0):
        return MergedSeq(
            seqs=[seq.repeat_interleave(times=times, dim=dim) for seq in self._seqs]
        )

    def to(self, device):
        return MergedSeq(seqs=[seq.to(device) for seq in self._seqs])

    def detach_(self):
        for seq in self._seqs:
            seq.detach_()

    @property
    def is_empty(self):
        return False


class EmptySeq:
    def __init__(self, tokenizer, device) -> None:
        self.tokenizer = tokenizer
        self.device = device

    @property
    def is_empty(self):
        return True

    def __len__(self):
        raise NotImplementedError


class Seq:
    def __init__(
        self,
        tokenizer,
        device,
        ids=None,
        tokens=None,
        text=None,
        logits=None,
        probs=None,
        mask=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device

        self._logits = None
        self._probs = None
        self._ids = None
        self._text = None
        self.mask = None

        self.eps = 1e-8

        if logits is not None:
            self.logits = logits
        elif probs is not None:
            self.probs = probs
        elif ids is not None:
            self.ids = ids
        elif text is not None:
            self.text = text
        elif tokens is not None:
            self.tokens = tokens
        else:
            raise RuntimeError("At least one field should be provided.")

        if mask is not None:
            self.mask = mask
        if self.mask is None:
            self.mask = torch.ones((self.bs, self.seq_len), device=self.device).bool()

    @property
    def logits(self):
        if self.dtype == "logits":
            logits = self._logits
        else:
            logits = torch.log(self.probs + self.eps)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(f"NaN in retrieved logits: {logits}")
        return logits

    @logits.setter
    def logits(self, value):
        self._logits = value
        if torch.isnan(self._logits).any() or torch.isinf(self._logits).any():
            raise RuntimeError(f"NaN in set logits: {self._logits}")

    @property
    def probs(self):
        if self.dtype == "probs":
            probs = self._probs
        elif self.dtype == "logits":
            probs = torch.nn.functional.softmax(self.logits, dim=2)
        else:
            probs = self.onehot
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            raise RuntimeError(f"NaN in retrieved probs: {probs}")
        return probs

    @probs.setter
    def probs(self, value):
        self._probs = value
        if torch.isnan(self._probs).any() or torch.isinf(self._probs).any():
            raise RuntimeError(f"NaN in set probs: {self._probs}")

    @property
    def ids(self):
        if self.dtype == "ids":
            ids = self._ids
        elif self.dtype == "logits":
            ids = torch.argmax(self.logits, dim=2)
        elif self.dtype == "probs":
            ids = torch.argmax(self.probs, dim=2)
        elif self.dtype == "text":
            ids = self.tokenizer.batch_encode_plus(
                self._text, return_tensors="pt", add_special_tokens=False, padding=True
            ).input_ids.to(self.device)
        else:
            raise RuntimeError("No data to retrieve ids from.")
        return ids

    @ids.setter
    def ids(self, value):
        self._ids = value
        self.mask = self._ids != self.tokenizer.pad_token_id

    @property
    def text(self):
        if self.dtype == "text":
            text = self._text
        else:
            ids = self.ids.tolist()
            mask = self.mask
            # remove masked tokens
            ids_sanitized = []
            for i in range(len(ids)):
                _ids = [id for id, m in zip(ids[i], mask[i]) if m]
                ids_sanitized.append(_ids)
            text = self.tokenizer.batch_decode(
                ids_sanitized,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
        return text

    @text.setter
    def text(self, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            raise ValueError(
                f"Text should be a list or tuple of strings. Got {type(value)}."
            )
        self._text = list(value)
        self.mask = self.ids != self.tokenizer.pad_token_id

    # derived properties
    @property
    def tokens(self):
        ids = self.ids
        tokens = [
            self.tokenizer.convert_ids_to_tokens(ids[i].tolist())
            for i in range(ids.shape[0])
        ]
        return tokens

    @tokens.setter
    def tokens(self, value):
        ids = [self.tokenizer.convert_tokens_to_ids(t) for t in value]
        self.ids = torch.tensor(ids, device=self.device)

    @property
    def onehot(self):
        ids = self.ids
        one_hot_mask = torch.zeros(
            (ids.shape[0], ids.shape[1], self.tokenizer.vocab_size), device=ids.device
        )
        one_hot_mask.scatter_(2, ids[:, :, None], 1)
        return one_hot_mask

    @onehot.setter
    def onehot(self, value):
        self.ids = torch.argmax(value, dim=2)

    @property
    def logprobs(self):
        return torch.log_softmax(self.logits, dim=-1)

    @logprobs.setter
    def logprobs(self, value):
        self.logits = value

    @property
    def bs(self):
        if self.dtype == "probs":
            bs = self.probs.shape[0]
        elif self.dtype == "logits":
            bs = self.logits.shape[0]
        elif self.dtype == "ids":
            bs = self.ids.shape[0]
        elif self.dtype == "text":
            bs = len(self._text)
        return bs

    @property
    def seq_len(self):
        if self.dtype == "probs":
            seq_len = self.probs.shape[1]
        elif self.dtype == "logits":
            seq_len = self.logits.shape[1]
        else:
            seq_len = self.ids.shape[1]
        return seq_len

    def __len__(self):
        raise NotImplementedError("len() is ambiguous, use .bs or .seq_len instead.")

    @property
    def is_hard(self):
        return self.dtype == "ids" or self.dtype == "text"

    def get_embed(self, embedding_matrix):
        if self.is_hard:
            embeds = embedding_matrix[self.ids.to(embedding_matrix.device)]
        else:
            embeds = torch.matmul(
                self.probs.to(embedding_matrix.device), embedding_matrix
            )
        if torch.isnan(embeds).any() or torch.isinf(embeds).any():
            raise RuntimeError(f"NaN in retrieved embeds: {embeds}")
        return embeds

    def get_entropy(self, average=True):
        if self.is_hard:
            if average:
                entropy = torch.zeros(1, device=self.device)[0]
            else:
                entropy = torch.zeros_like(self.ids)
        else:
            max_logit = torch.max(self.logits, dim=2, keepdim=True)[0]
            normalized_logits = self.logits - max_logit

            Z = torch.sum(torch.exp(normalized_logits), dim=2) + self.eps
            log_Z = torch.log(Z) + max_logit[:, :, 0]

            entropy_per_token = (
                log_Z - torch.sum(self.logits * torch.exp(normalized_logits), dim=2) / Z
            )

            if (
                torch.isnan(entropy_per_token).any()
                or torch.isinf(entropy_per_token).any()
            ):
                tqdm.write(
                    f"NaN/inf in entropy: {entropy_per_token}, Z = {Z}, log_Z = {log_Z}, logits = {self.logits}"
                )
            normalized_entropy_per_token = entropy_per_token / torch.log(
                torch.tensor(self.logits.shape[2]) + self.eps
            )
            if average:
                # ignore masked out tokens
                entropy_batch = torch.sum(
                    normalized_entropy_per_token * self.mask, dim=1
                ) / (torch.sum(self.mask, dim=1) + self.eps)
                entropy = torch.mean(entropy_batch)
            else:
                entropy = normalized_entropy_per_token
        return entropy

    def to_html(self, colors=None, normalize=False, color_scheme=1):
        if colors is None:
            colors = self.get_entropy(average=False)
        assert colors.shape == self.ids.shape
        if normalize:  # over dim 1
            colors = colors / torch.max(colors, dim=1, keepdim=True)[0]
        tokens = self.tokens
        masks = self.mask

        html_batch = []
        for tok, mask, entr in zip(tokens, masks, colors):
            html_list = []
            for t, m, e0 in zip(tok, mask, entr):
                e = 0 if math.isnan(e0) else e0
                if m == 0:
                    color = (0, 1, 0)
                else:
                    if color_scheme == 1:
                        color = (e, 0, 1 - e)
                    elif color_scheme == 2:
                        color = (e, e / 2, 0)
                color_str = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
                t_escaped = t.replace("<", "&lt;").replace(">", "&gt;")
                html = f"<span style='color: {color_str}'>{t_escaped}</span> "
                html_list.append(html)
            try:
                html_joined = wandb.Html("".join(html_list))
                html_batch.append(html_joined)
            except:
                tqdm.write("[Warning!!!] Saving HTML failed")
        return html_batch

    def __getitem__(self, _slice):
        original_slice = _slice
        if isinstance(_slice, tuple):
            assert self._text is None
        else:
            _slice = (_slice,)
        if len(_slice) == 1:
            _slice = list(_slice)
            _slice.append(slice(None, None, None))
            _slice = tuple(_slice)
        assert len(_slice) == 2

        new_slice = add_dummy_dim_to_slice(_slice)

        if self.dtype == "logits":
            sliced_seq = Seq(
                logits=self.logits[new_slice],
                mask=self.mask[new_slice],
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "probs":
            sliced_seq = Seq(
                probs=self.probs[new_slice],
                mask=self.mask[new_slice],
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "ids":
            sliced_seq = Seq(
                ids=self.ids[new_slice],
                mask=self.mask[new_slice],
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "text":
            if isinstance(original_slice, int):
                original_slice = [original_slice]
            sliced_seq = Seq(
                text=[self._text[i] for i in original_slice],
                mask=self.mask[new_slice],
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            raise RuntimeError("No data to slice.")
        return sliced_seq

    def clone(self):
        if self.dtype == "logits":
            cloned_seq = Seq(
                logits=self.logits.clone(),
                mask=self.mask.clone(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "probs":
            cloned_seq = Seq(
                probs=self.probs.clone(),
                mask=self.mask.clone(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "ids":
            cloned_seq = Seq(
                ids=self.ids.clone(),
                mask=self.mask.clone(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "text":
            cloned_seq = Seq(
                text=self.text,
                mask=self.mask.clone(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            raise RuntimeError("No data to clone.")
        return cloned_seq

    def detach(self):
        if self.dtype == "logits":
            detached_seq = Seq(
                logits=self.logits.detach(),
                mask=self.mask.detach(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "probs":
            detached_seq = Seq(
                probs=self.probs.detach(),
                mask=self.mask.detach(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "ids":
            detached_seq = Seq(
                ids=self.ids.detach(),
                mask=self.mask.detach(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "text":
            detached_seq = Seq(
                text=self.text,
                mask=self.mask.detach(),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            raise RuntimeError("No data to detach.")
        return detached_seq

    def repeat_interleave(self, times, dim=0):
        if self.bs == 1:
            raise ValueError(
                "Trying to repeat_interleave sequence with bs=1, this might break broadcasting."
            )
        if self.dtype == "logits":
            repeated_seq = Seq(
                logits=self.logits.repeat_interleave(times, dim=dim),
                mask=self.mask.repeat_interleave(times, dim=dim),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "probs":
            repeated_seq = Seq(
                probs=self.probs.repeat_interleave(times, dim=dim),
                mask=self.mask.repeat_interleave(times, dim=dim),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "ids":
            repeated_seq = Seq(
                ids=self.ids.repeat_interleave(times, dim=dim),
                mask=self.mask.repeat_interleave(times, dim=dim),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        elif self.dtype == "text":
            assert dim == 0
            repeated_seq = Seq(
                text=[item for item in self._text for i in range(times)],
                mask=self.mask.repeat_interleave(times, dim=dim),
                tokenizer=self.tokenizer,
                device=self.device,
            )
        else:
            raise RuntimeError("No data to repeat.")
        return repeated_seq

    def to(self, device):
        if self.dtype == "logits":
            moved_seq = Seq(
                logits=self.logits.to(device),
                mask=self.mask.to(device),
                tokenizer=self.tokenizer,
                device=device,
            )
        elif self.dtype == "probs":
            moved_seq = Seq(
                probs=self.probs.to(device),
                mask=self.mask.to(device),
                tokenizer=self.tokenizer,
                device=device,
            )
        elif self.dtype == "ids":
            moved_seq = Seq(
                ids=self.ids.to(device),
                mask=self.mask.to(device),
                tokenizer=self.tokenizer,
                device=device,
            )
        elif self.dtype == "text":
            moved_seq = Seq(
                text=self.text,
                mask=self.mask.to(device),
                tokenizer=self.tokenizer,
                device=device,
            )
        else:
            raise RuntimeError("No data to move.")
        return moved_seq

    def detach_(self):
        if self.dtype == "logits":
            self.logits.detach_()
        elif self.dtype == "probs":
            self.probs.detach_()
        elif self.dtype == "ids":
            self._ids.detach_()
        elif self.dtype == "text":
            pass
        else:
            raise RuntimeError("No data to detach.")
        self.mask.detach_()

    def append(self, seq):
        if self.bs != 1 and seq.bs != 1 and self.bs != seq.bs:
            raise ValueError(
                f"Cannot broadcaset sequences with batch sizes {self.bs} and {seq.bs}."
            )

        mask_list = expand_for_broadcast_tensor([self.mask, seq.mask], dim=0)
        self.mask = torch.cat(mask_list, dim=1)

        if self.dtype == "logits":
            logits_list = expand_for_broadcast_tensor([self.logits, seq.logits], dim=0)
            self.logits = torch.cat(logits_list, dim=1)
        elif self.dtype == "probs":
            probs_list = expand_for_broadcast_tensor([self.probs, seq.probs], dim=0)
            self.probs = torch.cat(probs_list, dim=1)
        elif self.dtype == "ids":
            ids_list = expand_for_broadcast_tensor([self.ids, seq.ids], dim=0)
            self.ids = torch.cat(ids_list, dim=1)
        elif self.dtype == "text":
            raise NotImplementedError("Appending to text is not implemented.")
        else:
            raise RuntimeError("No data to append.")

    @property
    def dtype(self):
        if self._logits is not None:
            dtype = "logits"
        elif self._probs is not None:
            dtype = "probs"
        elif self._ids is not None:
            dtype = "ids"
        elif self._text is not None:
            dtype = "text"
        else:
            raise RuntimeError("No data to retrieve dtype from.")
        return dtype

    @property
    def is_empty(self):
        return False
