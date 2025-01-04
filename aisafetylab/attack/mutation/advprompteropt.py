# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the original repo by Facebook.

import torch

from aisafetylab.models import MergedSeq, Seq, EmptySeq
from aisafetylab.utils import apply_repetition_penalty
from loguru import logger
from aisafetylab.attack.mutation import BaseMutation
class AdvPrompterOpt(BaseMutation):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = "AdvPrompterOpt"
    def mutate(self, instruct, target, prompter, target_llm, **kwargs):
        return advPrompterOpt(instruct, target, prompter, target_llm, **kwargs)


@torch.no_grad()
def advPrompterOpt( instruct, target, prompter, target_llm,  
                   max_new_tokens, repetition_penalty,
                   num_beams, top_k, num_chunks, candidates_do_sample, candidates_temperature, candidates_always_include_best, 
                   lambda_val, reweight_loss, 
                   beams_do_sample, beams_temperature, beams_always_include_best, 
                   verbose, 
                   **kwargs):
    if verbose:
        logger.info("\n Running AdvPrompterOpt: Generating optimized suffix...")
    if isinstance(instruct[0], str):
        instruct = Seq(text=instruct, tokenizer=prompter.tokenizer, device=prompter.device)
    if isinstance(target[0], str):
        target = Seq(text=target, tokenizer=target_llm.tokenizer, device=target_llm.device)

    # Compute the initial prediction losses without a suffix
    full_instruct = Seq(
        text=instruct.text, tokenizer=target_llm.tokenizer, device=target_llm.device
    )
    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(hard_labels=True, reweight_loss=reweight_loss),
    )
    losses = target_llm_tf.loss_batch.detach().to(prompter.device)  # (ebs, )

    # Initialize the beam scores
    beam_scores = torch.zeros_like(losses)  # (ebs, )
    suffix_beams = EmptySeq(tokenizer=prompter.tokenizer, device=prompter.device)

    for idx in range(max_new_tokens):
        if idx == 0:
            num_beams_in = 1
            num_beams_out = num_beams
        elif idx == max_new_tokens - 1:
            num_beams_in = num_beams
            num_beams_out = 1
        else:
            num_beams_in = num_beams
            num_beams_out = num_beams

        # expand the dimension of instruct and targets to match suffix beams
        instruct_rep = instruct.repeat_interleave(num_beams_in, dim=0)
        target_rep = target.repeat_interleave(num_beams_in, dim=0)

        next_dist_seq_prompter, next_dist_seq_basemodel = get_next_token_probabilities(
            instruct=instruct_rep, 
            suffix=suffix_beams, 
            prompter=prompter,
            repetition_penalty = repetition_penalty, 
        )

        next_token_candidate_ids, candidate_beam_scores, candidate_losses = (
            select_and_evaluate_next_token_candidates(
                
                
                instruct=instruct_rep,
                target=target_rep,
                suffix=suffix_beams,
                target_llm=target_llm,
                next_dist_seq_prompter=next_dist_seq_prompter,
                next_dist_seq_basemodel=next_dist_seq_basemodel,
                beam_scores=beam_scores,
                prev_losses=losses,
                num_beams_in=num_beams_in,
                top_k=top_k,
                num_chunks=num_chunks,
                candidates_do_sample=candidates_do_sample,
                candidates_temperature=candidates_temperature,
                candidates_always_include_best=candidates_always_include_best,
                lambda_val=lambda_val,
                reweight_loss=reweight_loss
            )
        )

        suffix_beams, losses, beam_scores = select_next_beams(
            suffix_beams=suffix_beams,
            next_token_candidate_ids=next_token_candidate_ids,
            candidate_beam_scores=candidate_beam_scores,
            candidate_losses=candidate_losses,
            num_beams_in=num_beams_in,
            num_beams_out=num_beams_out,
            beams_do_sample=beams_do_sample,
            beams_temperature=beams_temperature,
            beams_always_include_best=beams_always_include_best,
        )
        if verbose:
            logger.info(f" Beams[0] (iter {idx}): {suffix_beams[:num_beams_out].text}")

    if verbose:
        logger.info(
            f" AdvPrompterOpt completed. Generated suffix[0]: {suffix_beams[0].text}"
        )
    return suffix_beams


def get_next_token_probabilities(instruct, suffix, prompter, repetition_penalty):
    # get the next token probabilities from the prompter and the base model
    prompter_next = prompter.get_next_token(
        key="suffix", instruct=instruct, suffix=suffix
    )
    next_dist_seq_prompter = (
        prompter_next.response_dist.clone().detach()
    )  # (ebs, 1, vocab_size)
    prompter_next_basemodel = prompter.get_next_token(
        key="suffix", instruct=instruct, suffix=suffix, use_basemodel=True
    )
    next_dist_seq_basemodel = (
        prompter_next_basemodel.response_dist.clone().detach()
    )  # (bs, 1, vocab_size)

    # apply repetition penalty
    if not suffix.is_empty:
        next_dist_logits_basemodel = apply_repetition_penalty(
            logits=next_dist_seq_basemodel.logits.squeeze(1),
            prev_ids=suffix.ids,
            penalty=repetition_penalty,
        )
        next_dist_seq_basemodel = Seq(
            logits=next_dist_logits_basemodel[:, None, :],
            mask=next_dist_seq_basemodel.mask,
            tokenizer=next_dist_seq_basemodel.tokenizer,
            device=next_dist_seq_basemodel.device,
        )
        next_dist_logits_prompter = apply_repetition_penalty(
            logits=next_dist_seq_prompter.logits.squeeze(1),
            prev_ids=suffix.ids,
            penalty=repetition_penalty,
        )
        next_dist_seq_prompter = Seq(
            logits=next_dist_logits_prompter[:, None, :],
            mask=next_dist_seq_prompter.mask,
            tokenizer=next_dist_seq_prompter.tokenizer,
            device=next_dist_seq_prompter.device,
        )
    return next_dist_seq_prompter, next_dist_seq_basemodel


def select_and_evaluate_next_token_candidates(
    instruct,
    target,
    suffix,
    target_llm,
    next_dist_seq_prompter,
    next_dist_seq_basemodel,
    beam_scores,
    prev_losses,
    num_beams_in,
    top_k,
    num_chunks,
    candidates_do_sample,
    candidates_temperature,
    candidates_always_include_best,
    lambda_val,
    reweight_loss,
):
    num_chunks = num_chunks
    assert top_k % (num_chunks * num_beams_in) == 0
    num_samples_per_beam = top_k // (num_chunks * num_beams_in)
    all_next_token_candidate_ids = None

    for i in range(num_chunks):
        next_token_candidate_ids = select_next_token_candidates(
            next_dist_seq=next_dist_seq_prompter,
            previous_next_token_candidate_ids=all_next_token_candidate_ids,
            num_samples_per_beam=num_samples_per_beam,
            candidates_do_sample=candidates_do_sample,
            candidates_temperature=candidates_temperature,
            always_include_best=candidates_always_include_best,
        )  # (ebs = bs * num_beams_in, num_samples_per_beam)

        candidate_beam_scores, candidate_losses = evaluate_next_token_candidates(
            instruct=instruct,
            target=target,
            suffix=suffix,
            target_llm=target_llm,
            next_token_candidate_ids=next_token_candidate_ids,
            next_dist_seq_basemodel=next_dist_seq_basemodel,
            next_dist_seq_prompter=next_dist_seq_prompter,
            prev_beam_scores=beam_scores,
            prev_losses=prev_losses,
            lambda_val=lambda_val,
            reweight_loss=reweight_loss,
        )  # (ebs, num_samples_per_beam)

        if all_next_token_candidate_ids is None:
            all_next_token_candidate_ids = next_token_candidate_ids
            all_candidate_beam_scores = candidate_beam_scores
            all_candidate_losses = candidate_losses
        else:
            all_next_token_candidate_ids = torch.cat(
                (next_token_candidate_ids, all_next_token_candidate_ids), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_beam_scores = torch.cat(
                (candidate_beam_scores, all_candidate_beam_scores), dim=1
            )  # (ebs, i * num_samples_per_beam)
            all_candidate_losses = torch.cat(
                (candidate_losses, all_candidate_losses), dim=1
            )  # (ebs, i * num_samples_per_beam)
    return all_next_token_candidate_ids, all_candidate_beam_scores, all_candidate_losses


@torch.no_grad()
def select_next_token_candidates(
    next_dist_seq,
    previous_next_token_candidate_ids,
    num_samples_per_beam,
    candidates_do_sample,
    candidates_temperature,
    always_include_best,
):
    # clone is important here! We modify the logits but will also use the original dist
    next_dist_logits = next_dist_seq.logits.squeeze(1).clone()  # (ebs, vocab_size)

    if previous_next_token_candidate_ids is not None:
        previous_next_token_candidate_ids_khot = torch.scatter(
            torch.zeros_like(next_dist_logits), 1, previous_next_token_candidate_ids, 1
        )  # (ebs, vocab_size)
        next_dist_logits -= 1e10 * previous_next_token_candidate_ids_khot

    if candidates_do_sample:
        if always_include_best:
            next_dist_logits -= 1e10 * next_dist_seq.onehot.squeeze(1)

        probs = torch.softmax(
            next_dist_logits / candidates_temperature,
            dim=-1,
        )  # (ebs, vocab_size)
        next_token_candidate_ids = probs.multinomial(
            num_samples=num_samples_per_beam, replacement=False
        )  # (ebs, num_samples_per_beam)
        if always_include_best:
            next_token_candidate_ids = torch.cat(
                [next_dist_seq.ids, next_token_candidate_ids[:, :-1]], dim=1
            )
    else:
        next_token_candidate_ids = next_dist_logits.topk(
            k=num_samples_per_beam, dim=-1
        ).indices  # (ebs, num_samples_per_beam)
    return next_token_candidate_ids


@torch.no_grad()
def evaluate_next_token_candidates(
    instruct,
    target,
    suffix,
    target_llm,
    next_token_candidate_ids,
    next_dist_seq_basemodel,
    next_dist_seq_prompter,
    prev_beam_scores,
    prev_losses,
    lambda_val,
    reweight_loss,
):
    ebs, num_samples_per_beam = next_token_candidate_ids.shape

    q_next_token_candidate_ids = torch.reshape(
        next_token_candidate_ids, (ebs * num_samples_per_beam, 1)
    )
    q_sample_seq = Seq(
        ids=q_next_token_candidate_ids,
        tokenizer=next_dist_seq_prompter.tokenizer,
        device=next_dist_seq_prompter.device,
    )

    # extend to match the extended batch size
    instruct_rep = instruct.repeat_interleave(num_samples_per_beam, dim=0)
    target_rep = target.repeat_interleave(num_samples_per_beam, dim=0)
    if not suffix.is_empty:
        suffix_rep = suffix.repeat_interleave(num_samples_per_beam, dim=0)
    else:
        suffix_rep = suffix

    # compute the losses on each sample
    merged = MergedSeq(seqs=[instruct_rep, suffix_rep, q_sample_seq])
    full_instruct = Seq(
        text=merged.to_seq(merge_dtype="ids").text,
        tokenizer=target_llm.tokenizer,
        device=target_llm.device,
    )

    with torch.no_grad():
        target_llm_tf_q = target_llm.compute_pred_loss_teacher_forced(
            key="target",
            full_instruct=full_instruct,
            target=target_rep,
            loss_params=dict(hard_labels=True, reweight_loss=reweight_loss),
        )

    loss_batch = target_llm_tf_q.loss_batch.to(next_dist_seq_prompter.device)
    losses = torch.reshape(loss_batch, (ebs, num_samples_per_beam))
    loss_delta = losses - prev_losses[:, None]  # (ebs, num_samples_per_beam)

    next_dist_logprobs_basemodel = next_dist_seq_basemodel.logprobs.squeeze(1)
    selected_logprobs_basemodel = torch.gather(
        next_dist_logprobs_basemodel, dim=-1, index=next_token_candidate_ids
    )  # (ebs, num_samples_per_beam)

    factor = lambda_val
    beam_scores_delta = selected_logprobs_basemodel - loss_delta * factor
    new_beam_scores = prev_beam_scores[:, None] + beam_scores_delta
    return new_beam_scores, losses


@torch.no_grad()
def select_next_beams(
    suffix_beams,
    next_token_candidate_ids,
    candidate_beam_scores,
    candidate_losses,
    num_beams_in,
    num_beams_out,
    beams_do_sample,
    beams_temperature,
    beams_always_include_best,
):
    ebs, num_samples_per_beam = candidate_beam_scores.shape
    bs = ebs // num_beams_in

    candidate_beam_scores = candidate_beam_scores.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    if beams_do_sample:
        if beams_always_include_best:
            candidate_beam_scores_top_ids = candidate_beam_scores.argmax(dim=-1)
            candidate_beam_scores_onehot = torch.zeros_like(candidate_beam_scores)
            candidate_beam_scores_onehot.scatter_(
                1, candidate_beam_scores_top_ids[:, None], 1
            )
            candidate_beam_scores_corrected = (
                candidate_beam_scores - 1e10 * candidate_beam_scores_onehot
            )
            beam_probs = torch.softmax(
                candidate_beam_scores_corrected / beams_temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
        else:
            beam_probs = torch.softmax(
                candidate_beam_scores / beams_temperature,
                dim=-1,
            )  # (bs, num_beams_in * num_samples_per_beam)
        next_beam_indices = beam_probs.multinomial(
            num_samples=num_beams_out, replacement=False
        )  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]
        if beams_always_include_best:
            next_beam_indices = torch.cat(
                [candidate_beam_scores_top_ids[:, None], next_beam_indices[:, :-1]],
                dim=-1,
            )
    else:
        next_beam_indices = candidate_beam_scores.topk(
            k=num_beams_out, dim=-1, sorted=True
        ).indices  # (bs, num_beams_out) [0, num_beams_in * num_samples_per_beam]

    next_beam_indices_expanded = (
        next_beam_indices
        + torch.arange(0, bs, device=suffix_beams.device)[:, None]
        * num_beams_in
        * num_samples_per_beam
    )  # (bs, num_beams_out)
    next_beam_indices_expanded = next_beam_indices_expanded.reshape(-1)

    next_token_candidate_seq = Seq(
        ids=next_token_candidate_ids.reshape(
            bs * num_beams_in * num_samples_per_beam, 1
        ),
        tokenizer=suffix_beams.tokenizer,
        device=suffix_beams.device,
    )

    if suffix_beams.is_empty:
        next_suffix_beams = next_token_candidate_seq[next_beam_indices_expanded]
    else:
        beam_candidates = suffix_beams.repeat_interleave(num_samples_per_beam, dim=0)
        beam_candidates.append(next_token_candidate_seq)
        next_suffix_beams = beam_candidates[next_beam_indices_expanded]

    candidate_losses = candidate_losses.reshape(
        bs, num_beams_in * num_samples_per_beam
    )  # (bs, num_beams_in * num_samples_per_beam)
    selected_losses = candidate_losses.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    selected_losses = selected_losses.reshape(bs * num_beams_out).detach()

    selected_beam_scores = candidate_beam_scores.gather(
        dim=1, index=next_beam_indices
    )  # (bs, num_beams_out)
    selected_beam_scores = selected_beam_scores.reshape(bs * num_beams_out).detach()
    return next_suffix_beams, selected_losses, selected_beam_scores


@torch.no_grad()
def evaluate_prompt(
    instruct,
    suffix,
    full_instruct,
    target,
    prompter,
    target_llm,
    generate_target_llm_response,
    reweight_loss,
    print_idx=0,
    verbose=False,
):
    basemodel_tf = None
    if suffix is not None and not suffix.is_empty:
        basemodel_tf = prompter.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=instruct,
            suffix=suffix,
            use_basemodel=True,
            loss_params=dict(hard_labels=True),
        )

    target_llm_tf = target_llm.compute_pred_loss_teacher_forced(
        key="target",
        full_instruct=full_instruct,
        target=target,
        loss_params=dict(
            hard_labels=True,
            reweight_loss=reweight_loss,
        ),
    )
    if verbose:
        logger.info(f" --- Query[{print_idx}]: '{target_llm_tf.query.text[print_idx]}'")
        logger.info(f" --- Suffix[{print_idx}]: '{suffix.text[print_idx]}'")
        logger.info(f" --- Target[{print_idx}]: '{target.text[print_idx]}'")
        logger.info(
            f" --- TF Response[{print_idx}]: '{target_llm_tf.response_dist.text[print_idx]}'"
        )

    if generate_target_llm_response:
        target_llm_ar = target_llm.generate_autoregressive(
            key="target",
            full_instruct=full_instruct,
        )
        if verbose:
            logger.info(
                f" --- AR Response[{print_idx}]: '{target_llm_ar.response_sample.text[print_idx]}'"
            )
    else:
        target_llm_ar = None

    if verbose:
        logger.info(f" Evaluating suffix completed. TF Loss: {target_llm_tf.loss:.3f}")

    return target_llm_tf, target_llm_ar, basemodel_tf
