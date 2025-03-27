"""
AdvPrompter Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.
This part of code is based on the code from the paper.

Paper title: AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs
arXiv link: https://arxiv.org/abs/2404.16873
Source repository: https://github.com/facebookresearch/advprompter.git
"""

from copy import copy
import pandas as pd
import os
import numpy as np
import pytorch_lightning as pl
import setproctitle
import torch
import re
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tqdm import tqdm
from collections import defaultdict
import wandb
import warnings
from loguru import logger

from aisafetylab.models import LLM
from aisafetylab.models import MergedSeq, Seq, stack_seqs
from aisafetylab.utils import (
    Metrics,
    check_jailbroken,
    dotdict,
    hit_rate_at_n,
    check_affirmative,
)
from aisafetylab.dataset import get_dataloader
from aisafetylab.attack.mutation import AdvPrompterOpt, evaluate_prompt

# setproctitle.setproctitle("advprompter-train")
def collate_fn(list_of_data):
    (
        instruct_batch,
        target_batch,
        suffix_batch,
        priority_batch,
    ) = zip(*list_of_data)
    context = dotdict()
    context.instruct = stack_seqs(instruct_batch)
    context.target = stack_seqs(target_batch)
    context.suffix = stack_seqs(suffix_batch)
    return context, priority_batch

column_names = [
    "step",
    "split",
    "batch_idx",
    "sample_idx",
    # Prompt prediction
    "prompter/ar/query",
    "prompter/ar/response",  # auto-regressive prompter generation
    "prompter/ar/response_perplexity_basemodel",
    #
    # --- Evaluation of predicted prompt ---
    "target_llm/ar/query",
    "target_llm/ar/response",
    "target_llm/ar/jailbroken",
]


def log_data(
    log_table,
    metrics,
    step,
    split,
    batch_idx,
    test_prefixes,
    affirmative_prefixes,
    log_sequences_to_wandb,
    log_metrics_to_wandb,
    batch_size=None,
    target_llm_tf=None,
    target_llm_ar=None,
    prompter_ar=None,
    basemodel_tf=None,
    prompter_tf_opt=None,
):
    if batch_size is None and prompter_ar is None:
        raise ValueError("either batch_size or prompter_ar must be provided")
    bs = batch_size if batch_size is not None else prompter_ar.response_sample.bs
    log_dct = {}
    log_seqs = {
        "step": [step] * bs,
        "split": [split] * bs,
        "batch_idx": [batch_idx] * bs,
        "sample_idx": list(range(bs)),
    }

    if prompter_ar is not None:
        log_seqs["prompter/ar/query"] = prompter_ar.query.to_html()
        if basemodel_tf is not None:
            log_dct["prompter/ar/response_perplexity_basemodel"] = (
                basemodel_tf.perplexity.mean().item()
            )

            log_seqs["prompter/ar/response"] = prompter_ar.response_sample.to_html(
                colors=basemodel_tf.loss_masked, normalize=True, color_scheme=2
            )
            log_seqs["prompter/ar/response_perplexity_basemodel"] = (
                basemodel_tf.perplexity
            )
        else:
            log_seqs["prompter/ar/response"] = prompter_ar.response_sample.to_html()

    if target_llm_tf is not None:
        target_llm_tf_affirmative_avg, target_llm_tf_affirmative_list = (
            check_affirmative(
                seq=target_llm_tf.response_dist,
                affirmative_prefixes=affirmative_prefixes,
            )
        )

        log_dct["target_llm/tf/response_entropy"] = (
            target_llm_tf.response_dist.get_entropy().item()
        )
        log_dct["target_llm/tf/affirmative"] = target_llm_tf_affirmative_avg
        log_dct["target_llm/tf/loss"] = target_llm_tf.loss.item()

    if target_llm_ar is not None:
        target_llm_ar_jailbroken_avg, target_llm_ar_jailbroken_list = check_jailbroken(
            seq=target_llm_ar.response_sample, test_prefixes=test_prefixes
        )

        # log_dct["target_llm/ar/jailbroken"] = target_llm_ar_jailbroken_avg
        log_dct["target_llm/ar/jailbroken_sum"] = sum(target_llm_ar_jailbroken_list)

        log_seqs["target_llm/ar/query"] = target_llm_ar.query.to_html()
        log_seqs["target_llm/ar/response"] = target_llm_ar.response_sample.to_html()
        log_seqs["target_llm/ar/jailbroken"] = target_llm_ar_jailbroken_list

    if prompter_tf_opt is not None:
        log_dct["prompter/tf/opt/response_dist_entropy"] = (
            prompter_tf_opt.response_dist.get_entropy().item()
        )
        log_dct["prompter/tf/opt/loss"] = prompter_tf_opt.loss.item()

    metrics.log_dict(log_dct, step=step, log_to_wandb=log_metrics_to_wandb)
    if log_sequences_to_wandb:
        log_data_to_table(log_table, bs, log_seqs)


def log_data_to_table(log_table, bs, log_seqs):
    log_list = []

    for column_name in column_names:
        if column_name in log_seqs:
            log_list.append(log_seqs[column_name])
        else:
            log_list.append([None] * bs)

    for bi in range(bs):
        log_l = [x[bi] for x in log_list]
        log_table.add_data(*log_l)


class BaseWorkspace:
    def __init__(self, cfg):

        self._copy_from_cfg(cfg)
        self._load_models()
        self._load_prefixes_data()

        if self.enable_wandb:
            self._init_wandb()
        else:
            self.train_table = None
            self.eval_table = None

    @torch.no_grad()
    def _copy_from_cfg(self, cfg):
        pl.seed_everything(cfg.seed)
        self.step = 0
        self.cfg = cfg
        self.verbose = cfg.verbose
        self.enable_wandb = cfg.wandb_params.enable_wandb
    
    @torch.no_grad()
    def _init_wandb(self):
        tqdm.write("Initializing Wandb...")
        wandb_id = wandb.util.generate_id()
        config = self.cfg
        wandb.init(
            entity=self.cfg.wandb_params.entity,
            project=self.cfg.wandb_params.project,
            config=config,
            id=wandb_id,
            resume="allow",
        )

        self.train_table = wandb.Table(columns=column_names)
        self.eval_table = wandb.Table(columns=column_names)

    @torch.no_grad()
    def _load_models(self):
        logger.info("Initializing Prompter...")
        self.prompter = LLM(self.cfg.prompter, verbose=self.verbose)
        logger.info("Initializing Target LLM ...")
        self.target_llm = LLM(self.cfg.target_llm, verbose=self.verbose)
    @torch.no_grad()
    def _load_prefixes_data(self):
        from aisafetylab.evaluation.scorers import PatternScorer
        self.test_prefixes = PatternScorer().pattern_dict['fail']
        self.affirmative_prefixes = PatternScorer().pattern_dict['pass']
        # self.test_prefixes = pd.read_csv(self.cfg.data.test_prefixes_pth).to_dict(orient='records')
        # self.affirmative_prefixes = pd.read_csv(self.cfg.data.affirmative_prefixes_pth).to_dict(orient='records')

    def batch_to_context(self, batch): # convert text to seq, moving to corresponding device
        model_map = dict(
            instruct = self.prompter,
            suffix = self.prompter,
            target = self.target_llm,
            full_instruct = self.target_llm
        )
        context = dotdict()
        for key, model in model_map.items():
            if key in batch.keys():
                context[key] = Seq(
                    text = batch[key],
                    tokenizer = model.tokenizer,
                    device = model.device
                )
            else:
                context[key] = None
        return context
    @torch.no_grad()
    def save_prompter(self):
        save_path = os.path.join(self.cfg.train.model_save_dir, f"step_{self.step}")
        tqdm.write(f" Saving prompter to {save_path}...")
        self.prompter.save_pretrained(save_path=save_path)
    
class EvalSuffixDatasetsWorkspace(BaseWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.prompter.eval()
        self.target_llm.eval()
    
    @torch.no_grad()
    def eval_suffix_datasets(self, suffix_dataset_pth_dct):
        for suffix_dataset_key, suffix_dataset_pth in suffix_dataset_pth_dct.items():
            self.eval_suffix_dataset(
                suffix_dataset_key=suffix_dataset_key,
                suffix_dataset_pth=suffix_dataset_pth,
            )
    
    @torch.no_grad()
    def eval_suffix_dataset(self, suffix_dataset_key, suffix_dataset_pth):
        split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)
        eval_loader = get_dataloader(
            suffix_dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        num_batches = len(eval_loader)
        eval_metrics = Metrics(prefix=split + "_eval/")

        instruct_jb_dict = defaultdict(list)
        processed_samples, ppl_sum = 0, 0
        
        logger.info(f"Start evaluating suffix dataset {suffix_dataset_key}, {num_batches} batches in total")

        for batch_idx, batch in enumerate(eval_loader):

            context = self.batch_to_context(batch)
            instruct, suffix, full_instruct, target = context.instruct, context.suffix, context.full_instruct, context.target

            target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
                instruct=instruct,
                suffix=suffix,
                full_instruct=full_instruct,
                target=target,
                prompter = self.prompter,
                target_llm = self.target_llm,
                generate_target_llm_response=True,
                reweight_loss=self.cfg.reweight_loss,
                verbose=self.cfg.verbose,
                print_idx=0
            )

            _, jailbroken_list = check_jailbroken(
                seq=target_llm_ar.response_sample,
                test_prefixes=self.test_prefixes
            )

            assert instruct.bs == len(jailbroken_list)
            instruct_text = instruct.text
            for i in range(instruct.bs):
                instruct_jb_dict[instruct_text[i]].append(jailbroken_list[i])
            

            log_data(
                log_table=None,
                metrics=eval_metrics,
                step=self.step,
                split=split,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                batch_size=self.cfg.eval.batch_size,
                log_sequences_to_wandb=False,
                log_metrics_to_wandb=False,
                target_llm_tf=target_llm_tf,
                target_llm_ar=target_llm_ar,
                basemodel_tf=basemodel_tf,
            )
            processed_samples += instruct.bs
            if basemodel_tf is not None:
                ppl_sum += basemodel_tf.perplexity.sum().item()
            total_jailbroken = sum(
                eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"]
            )
            
            logger.info(f"Evaluating {suffix_dataset_key} | {batch_idx+1}/{num_batches} batches completed | {total_jailbroken}/{processed_samples} of processed samples are jailbroken |  PPL: {float(ppl_sum) / processed_samples:.2f}")

        avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
        avg_metrics["avg/" + split + "_eval/target_llm/ar/jailbroken_sum"] = (
            float(
                sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])
            )
            / processed_samples
        )

        logger.info(f" Loss: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.2f}")
        logger.info(f" Jailbroken: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.2f}")
        logger.info(f" PPL: {float(ppl_sum) / processed_samples:.2f}")

        jb_all = [jb_list for (instruct, jb_list) in instruct_jb_dict.items()]
        max_length = max(len(sublist) for sublist in jb_all)
        padded_list = [
            np.pad(sublist, (0, max_length - len(sublist)), "constant")
            for sublist in jb_all
        ]
        jb_stat_np = np.array(padded_list)
        for ti in range(1, jb_stat_np.shape[1] + 1):
            logger.info(
                f"{suffix_dataset_key} | hit rate @ {ti}: {hit_rate_at_n(jb_stat_np, ti)}"
            )
        if self.enable_wandb:
            wandb.log(avg_metrics, step=self.step)
            wandb.log(dict(eval_examples=copy(self.eval_table)), step=self.step)

class EvalWorkspace(EvalSuffixDatasetsWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def eval(self):
        suffix_dataset_pth_dct = self.generate_suffix_datasets()
        self.eval_suffix_datasets(suffix_dataset_pth_dct)
    @torch.no_grad()
    def generate_suffix_datasets(self):
        suffix_dataset_pth_dct = {}
        for dataset_key, dataset_pth in self.cfg.eval.data.dataset_pth_dct.items():
            suffix_dataset = self.generate_suffix_dataset(
                dataset_key=dataset_key, dataset_pth=dataset_pth
            )
            suffix_dataset_pth = self.save_suffix_dataset(
                suffix_dataset, dir=self.cfg.eval.data.suffix_dataset_dir
            )
            suffix_dataset_pth_dct[suffix_dataset.suffix_dataset_key] = (
                suffix_dataset_pth
            )
        return suffix_dataset_pth_dct
    @torch.no_grad()
    def generate_suffix_dataset(self, dataset_key, dataset_pth):
        if self.cfg.prompter.gen_params.do_sample:
            num_trials = self.cfg.eval.num_trials
        else:
            if self.cfg.eval.num_trials != 1:
                warnings.warn(
                    "Prompter generation is deterministic, but num_trials > 1. Setting num_trials to 1."
                )
            num_trials = 1

        data = []

        suffix_dataset_key = f"{dataset_key}_{self.step}"
        eval_loader = get_dataloader(
            data_pth=dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        
        logger.info(f"Generating suffix dataset {suffix_dataset_key}...")
        for batch in tqdm(eval_loader, desc=f"Generating suffix dataset {suffix_dataset_key}"):
            batch['instruct'] = batch['query'] # "instruct" key and "target" key is necessary
            context = self.batch_to_context(batch)
            instruct = context.instruct
            target = context.target
            batch_data = []
            for max_new_tokens in self.cfg.eval.prompter.max_new_tokens_list:
                trial_data = []
                for trial in range(num_trials):
                    prompter_ar = self.prompter.generate_autoregressive(
                        key="suffix",
                        max_new_tokens=max_new_tokens,
                        instruct=instruct,
                    )
                    suffix = prompter_ar.response_sample

                    full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(
                        merge_dtype="ids"
                    )

                    assert instruct.bs == target.bs == suffix.bs
                    datapoint = []
                    for i in range(instruct.bs):
                        datapoint.append(
                            (
                                instruct.text[i],
                                target.text[i],
                                suffix.text[i],
                                full_instruct.text[i],
                            )
                        )
                    trial_data.append(datapoint)
                batch_data.append(trial_data)

            # aggregate data of same instruct together
            for i in range(instruct.bs):
                for j in range(len(self.cfg.eval.prompter.max_new_tokens_list)):
                    for k in range(num_trials):
                        data.append(batch_data[j][k][i])
        suffix_dataset = dotdict(
            data=data,
            fields=["instruct", "target", "suffix", "full_instruct"],
            suffix_dataset_key=suffix_dataset_key,
        )

       
        return suffix_dataset
    
    @torch.no_grad()
    def save_suffix_dataset(self, suffix_dataset, dir):
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        suffix_dataset_pth = os.path.join(
            dir,
            suffix_dataset.suffix_dataset_key + ".csv",
        )
        logger.info(
            f" Saving {suffix_dataset.suffix_dataset_key} to {suffix_dataset_pth}"
        )

        suffix_dataset = pd.DataFrame(
            suffix_dataset.data, columns=suffix_dataset.fields
        )
        
        suffix_dataset.to_csv(suffix_dataset_pth, index=False)
        return suffix_dataset_pth


from dataclasses import dataclass, field
from typing import List, Optional

# Define structured config classes
@dataclass
class LoraConfig:
    r: int = 8  # Default from YAML
    lora_alpha: int = 16  # Default from YAML
    bias: str = "none"  # Default from YAML
    target_modules: List[str] = None  # Default from YAML (will be set below)

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "lm_head"]  # Default target modules

@dataclass
class LoraParams:
    warmstart: Optional[bool] = False  # Default from YAML
    lora_checkpoint: Optional[str] = None  # Default from YAML
    lora_config: Optional[LoraConfig] = None  # Default to None

   
@dataclass
class LLMParams:
    device: str = "cuda:1"  # Default from YAML
    freeze: bool = False  # Default from YAML
    dtype: str = "float32"  # Default from YAML
    model_name: str = "llama2-7b"  # Default from YAML
    checkpoint: str = "meta-llama/Llama-2-7b-hf"  # Default from YAML
    lora_params: LoraParams = field(default_factory=LoraParams)  # Default LoraParams if not provided


@dataclass
class PromptTemplate:
    key: str
    msg: str

@dataclass
class PromptManager:
    prompt_template: List[PromptTemplate]

@dataclass
class Prompter:
    llm_params: LLMParams = field(default_factory=LLMParams)  # Default LLMParams if not provided
    allow_non_ascii: Optional[bool] = False  # Default from YAML
    gen_params: Optional[dict] = None  # Default GenParams if not provided
    prompt_manager: PromptManager = None  # PromptManager should be passed when initializing

    def __post_init__(self):
        if self.prompt_manager is None:
            if "llama2" in self.llm_params.model_name.lower():
                # Default prompt manager with templates
                self.prompt_manager = PromptManager(
                    prompt_template=[
                        PromptTemplate(key="system_message", msg="<s>"),
                        PromptTemplate(key="hyper_instruct", msg="{instruct}"),
                        PromptTemplate(key="suffix", msg="{suffix}")
                    ]
                )
            else:
                raise ValueError("Unsupported model name: {}, you must set prompt_manager manually".format(self.llm_params.model_name))
        if self.gen_params is None:
            self.gen_params = dict(
                do_sample=True,
                temperature=1.0,
                top_p=0.9
            )



    


class FinetuneWorkspace(EvalWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
    def _init_train_components(self):
        self.prompter_optimizer = torch.optim.Adam(
            self.prompter.parameters(), **self.cfg.train.prompter_optim_params
        )
        sampler = PrioritizedSampler(
            max_capacity=self.cfg.train.replay_buffer.size,
            alpha=self.cfg.train.replay_buffer.priority_alpha,
            beta=1.0,
        )
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(self.cfg.train.replay_buffer.size),
            batch_size=self.cfg.train.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )
    def add_to_replay_buffer(
        self,
        instruct,
        suffix,
        target,
        target_llm_tf,
        target_llm_tf_opt,
        target_llm_ar_opt,
        replay_buffer,
    ):
        self.replay_buffer = replay_buffer

        loss_batch = target_llm_tf.loss_batch
        loss_opt_batch = target_llm_tf_opt.loss_batch
        # priority = priority_factor.loss_delta * relu(loss_delta) + priority_factor.jailbreaking * jailbreaking
        priority = (
            torch.relu(loss_batch - loss_opt_batch)
            * self.cfg.train.replay_buffer.priority_factor.loss_delta
        )
        if self.cfg.train.replay_buffer.priority_factor.jailbreaking > 0:
            _, target_llm_ar_opt_jailbroken_list = check_jailbroken(
                seq=target_llm_ar_opt.response_sample,
                test_prefixes=self.test_prefixes,
            )
            jailbroken = torch.tensor(
                target_llm_ar_opt_jailbroken_list, device=loss_batch.device
            )
            priority += (
                jailbroken * self.cfg.train.replay_buffer.priority_factor.jailbreaking
            )
        for i, prio in enumerate(priority):
            if prio > 0:
                datapoint = (
                    instruct[i],
                    target[i],
                    suffix[i],
                    priority[i],
                )
                idx = self.replay_buffer.add(datapoint)
                self.replay_buffer.update_priority(index=idx, priority=prio.item())
    

    def finetune_prompter_step(self, instruct, suffix, prompter_optimizer, step=0):

        self.prompter_optimizer = prompter_optimizer
        self.step = step

        self.prompter_optimizer.zero_grad()
        prompter_tf_opt = self.prompter.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=instruct,
            suffix=suffix,
            loss_params=dict(hard_labels=True),
        )
        loss = prompter_tf_opt.loss
        loss.backward()
        self.prompter_optimizer.step()
        if self.enable_wandb:
            wandb.log({"regression_loss": loss.item()}, step=self.step)
        return prompter_tf_opt
    
    def finetune_prompter_with_data_sampled_from_replay_buffer(self, prompter_optimizer, replay_buffer):
        self.prompter_optimizer = prompter_optimizer
        self.replay_buffer = replay_buffer

        if len(self.replay_buffer) < self.cfg.train.batch_size:
            logger.info("Replay buffer size is less than batch size, skipping finetuning step")
            return None
        
        if self.verbose:
            logger.info( f"Step: {self.step} | Sampling from replay buffer and finetuning prompter...")
        
        num_updates = min(
            self.cfg.train.replay_buffer.num_updates,
            len(self.replay_buffer) // self.cfg.train.batch_size,
        ) # either sample num_updates times or sample until replay buffer is exhausted (if the replay buffer is not big enough to sample num_updates times)

        for _ in range(num_updates):
            context, priority_batch = self.replay_buffer.sample(
                batch_size = self.cfg.train.batch_size,
            )
            prompter_tf_opt = self.finetune_prompter_step(
                instruct = context.instruct,
                suffix = context.suffix,
                prompter_optimizer = prompter_optimizer,
                step = self.step,
            )

            if self.verbose:
                logger.info(
                    f"Step: {self.step} | {_+1}/{num_updates} updates completed | Regressing Prompter to sampled target suffixes: Loss {prompter_tf_opt.loss:.3f}, Sample priorities {[p.item() for p in priority_batch]}"
                )
        return prompter_tf_opt
            
class AdvprompterWorkspace(FinetuneWorkspace):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mutator = AdvPrompterOpt()
    def train(self): # Main function
        logger.info("Initializing optimizer and replay buffer...")
        self._init_train_components()
        
        if self.cfg.train.do_initial_eval:
            logger.info("Doing initial eval before optionally pretraining and training...")
            self.eval()
        if self.cfg.pretrain.enable:
            logger.info("Starting pretraining...")
            self.pretrain()
            if self.cfg.train.model_save_dir is not None and self.cfg.train.save_pretrain:
                logger.info("Saving pretraining ckpts...")
                self.save_prompter()
        logger.info("Start training...")
        for epoch in range(self.cfg.train.epochs):
            logger.info(f"Epoch {epoch}/{self.cfg.train.epochs}")
            self.train_epoch(epoch=epoch)
            if (
                self.cfg.train.eval_every is not None
                and (epoch + 1) % self.cfg.train.eval_every == 0
                and (epoch + 1) < self.cfg.train.epochs
            ):
                if self.cfg.train.model_save_dir is not None and self.cfg.train.always_save_before_eval:
                    self.save_prompter()
                self.eval()
        if self.cfg.train.model_save_dir is not None:
            self.save_prompter()
        self.eval()

    def pretrain(self):
        logger.info("Starting pretraining...")
        for pretrain_epoch in tqdm(range(self.cfg.pretrain.epochs), desc="Warmstarting (epochs)"):
            self.pretrain_epoch()
        if self.cfg.pretrain.do_eval_after:
            self.eval()
    def pretrain_epoch(self):
        self.prompter.train()
        self.target_llm.eval()

        pretrain_metrics = Metrics(prefix="pretrain/")
        pretrain_loader = get_dataloader(
            data_pth=self.cfg.pretrain.dataset_pth,
            shuffle=True,
            augment_target=False,
            batch_size=self.cfg.pretrain.batch_size,
        )
        for batch_idx, batch in enumerate(pretrain_loader):
            context = self.batch_to_context(batch)
            instruct = context.instruct
            suffix = context.suffix
            prompter_tf_opt = self.finetune_prompter_step(
                instruct=instruct, suffix=suffix, prompter_optimizer=self.prompter_optimizer, step = self.step
            )
            log_data(
                log_table=self.train_table,
                metrics=pretrain_metrics,
                step=self.step,
                split=self.cfg.pretrain.dataset_key,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                batch_size=self.cfg.pretrain.batch_size,
                log_sequences_to_wandb=False,
                log_metrics_to_wandb=self.enable_wandb,
                prompter_tf_opt=prompter_tf_opt,
            )
            self.step += instruct.bs

        if self.enable_wandb:
            wandb.log(dict(train_examples=copy(self.train_table)), step=self.step)
        avg_metrics = pretrain_metrics.get_avg(
            step=self.step, log_to_wandb=self.enable_wandb
        )
        logger.info(
            f" Pretrain epoch opt loss: {avg_metrics['avg/pretrain/prompter/tf/opt/loss']:.2f}"
        )
    
    def train_epoch(self, epoch):
        self.prompter.train()
        self.target_llm.eval()

        train_metrics = Metrics(prefix="train/")
        train_loader = get_dataloader(
            data_pth=self.cfg.train.dataset_pth,
            shuffle=True,
            augment_target=self.cfg.train.augment_target,
            batch_size=self.cfg.train.batch_size,
        )
        
        data = []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
            # print("="*10)
            # print(f"Batch {batch_idx}")
            # print(batch)
            # print("="*10)
            
            batch['instruct'] = batch['query']

            context = self.batch_to_context(batch)
            instruct = context.instruct
            target = context.target
            log_sequences = (
                batch_idx % self.cfg.wandb_params.log_sequences_every.train == 0
            )
            with torch.no_grad():

                # generate initial suffix
                prompter_ar = self.prompter.generate_autoregressive(
                    key="suffix",
                    max_new_tokens=self.cfg.train.q_params.max_new_tokens,
                    instruct=instruct,
                )
                suffix = prompter_ar.response_sample

                # combine instruct and initial suffix to form initial full instruct
                full_instruct_text = (
                    MergedSeq(seqs=[instruct, suffix]).to_seq(merge_dtype="ids").text
                )
                full_instruct = Seq(
                    text=full_instruct_text,
                    tokenizer=self.target_llm.tokenizer,
                    device=self.target_llm.device,
                )

                # evaluate initial suffix
                if self.verbose:
                    logger.info(f"\nStep: {self.step} | Evaluating initial suffix...")
                target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
                    instruct=instruct,
                    suffix=suffix,
                    full_instruct=full_instruct,
                    target=target,
                    prompter=self.prompter,
                    target_llm=self.target_llm,
                    generate_target_llm_response=log_sequences,
                    reweight_loss=self.cfg.reweight_loss,
                    verbose=self.cfg.verbose,
                    print_idx=0
                )

                # generate optimized suffix
                suffix = self.mutator.mutate(
                    instruct=instruct,
                    target=target,
                    prompter=self.prompter,
                    target_llm=self.target_llm,
                    max_new_tokens=self.cfg.train.q_params.max_new_tokens,
                    repetition_penalty=self.cfg.train.q_params.repetition_penalty,
                    num_beams=self.cfg.train.q_params.num_beams,
                    top_k=self.cfg.train.q_params.top_k,
                    num_chunks=self.cfg.train.q_params.num_chunks,
                    candidates_do_sample=self.cfg.train.q_params.candidates.do_sample,
                    candidates_temperature=self.cfg.train.q_params.candidates.temperature,
                    candidates_always_include_best=self.cfg.train.q_params.candidates.always_include_best,
                    lambda_val=self.cfg.train.q_params.lambda_val,
                    reweight_loss=self.cfg.reweight_loss,
                    beams_do_sample=self.cfg.train.q_params.beams.do_sample,
                    beams_temperature=self.cfg.train.q_params.beams.temperature,
                    beams_always_include_best=self.cfg.train.q_params.beams.always_include_best,
                    verbose=self.cfg.verbose
                )

                # combine instruct and optimized suffix to form optimized full instruct
                full_instruct_text = MergedSeq(seqs=[instruct, suffix]).to_seq(
                    merge_dtype="ids"
                )
                full_instruct = Seq(
                    text=full_instruct_text.text,
                    tokenizer=self.target_llm.tokenizer,
                    device=self.target_llm.device,
                )

                # evaluate optimized suffix
                if self.verbose:
                    logger.info(f"\nStep: {self.step} | Evaluating optimized suffix...")
                target_llm_tf_opt, target_llm_ar_opt, basemodel_tf_opt = (
                    evaluate_prompt(
                        instruct=instruct,
                        suffix=suffix,
                        full_instruct=full_instruct,
                        target=target,
                        prompter=self.prompter,
                        target_llm=self.target_llm,
                        generate_target_llm_response=True,
                        reweight_loss=self.cfg.reweight_loss,
                        verbose=self.cfg.verbose,
                        print_idx=0
                    )
                )

                # store suffixes
                for i in range(instruct.bs):
                    data.append(
                        (
                            instruct.text[i],
                            target.text[i],
                            suffix.text[i],
                            full_instruct.text[i],
                        )
                    )

            self.add_to_replay_buffer(
                instruct=instruct,
                suffix=suffix,
                target=target,
                target_llm_tf=target_llm_tf,
                target_llm_tf_opt=target_llm_tf_opt,
                target_llm_ar_opt=target_llm_ar_opt,
                replay_buffer=self.replay_buffer,
            )

            prompter_tf_opt = self.finetune_prompter_with_data_sampled_from_replay_buffer(
                prompter_optimizer=self.prompter_optimizer,
                replay_buffer=self.replay_buffer,
            )

            log_data(
                log_table=self.train_table,
                metrics=train_metrics,
                step=self.step,
                split=self.cfg.train.dataset_key,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                log_sequences_to_wandb=log_sequences and self.enable_wandb,
                log_metrics_to_wandb=self.enable_wandb,
                prompter_ar=prompter_ar,
                target_llm_tf=target_llm_tf,
                target_llm_ar=target_llm_ar,
                basemodel_tf=basemodel_tf,
                prompter_tf_opt=prompter_tf_opt,
            )

            self.step += instruct.bs

        suffix_dataset = dotdict(
            data=data,
            fields = ["instruct", "target", "suffix", "full_instruct"],
            suffix_dataset_key = f"{self.cfg.train.dataset_key}_opt_{self.step}",
        )
        self.save_suffix_dataset(
            suffix_dataset, dir=self.cfg.train.suffix_opt_dataset_dir
        )

        if self.enable_wandb:
            wandb.log(dict(train_examples=copy(self.train_table)), step=self.step)

        avg_metrics = train_metrics.get_avg(
            step=self.step, log_to_wandb=self.enable_wandb
        )
        logger.info(
            f" Train loss epoch {epoch}: {avg_metrics['avg/train/target_llm/tf/loss']:.2f}"
        )







class AdvPrompter:
    """
    A class that manages an adversarial prompt generation process using a specified language model and LoRA (Low-Rank Adaptation).

    The `AdvPrompter` is designed to generate adversarially perturbed prompts by leveraging a language model with LoRA checkpoints. The class offers functionality for batching input text, 
    moving it to the appropriate device, and generating modified text (e.g., prompt mutations).

    Parameters
    ----------
    attacker_model_name : str
        The name of the model to use for the attack (e.g., 'vicuna-7b').
    attacker_model_path : str
        The file path to the model checkpoint for the attacker model.
    lora_checkpoint : str
        The file path to the LoRA checkpoint, which provides additional fine-tuning parameters.
    device : str, optional
        The device on which the model will run (default is 'cuda:0').
    dtype : str, optional
        The data type to use for the model (default is 'float32').
    verbose : bool, optional
        Whether to print detailed logs during execution (default is False).

    Attributes
    ----------
    prompter_config : Prompter
        Configuration object for the prompt generation, including model parameters and LoRA settings.
    verbose : bool
        Whether verbose output is enabled.
    prompter : LLM
        The language model used for prompt generation, initialized with the provided configuration.

    Methods
    -------
    batch_to_context(batch)
        Converts a batch of text input to a context object, moving data to the appropriate device.
    mutate(prompt, max_new_tokens=50)
        Generates a modified prompt by adding adversarial content to the input prompt using autoregressive sampling.
    """
    def __init__(self, attacker_model_name, attacker_model_path, lora_checkpoint, device='cuda:0', dtype='float32',  verbose=False):
        self.prompter_config = Prompter()
        self.prompter_config.llm_params = LLMParams(
            device = device,
            freeze = False,
            dtype = dtype,
            model_name = attacker_model_name,
            checkpoint = attacker_model_path,
            lora_params = LoraParams(
                lora_checkpoint = lora_checkpoint,
                warmstart=True
            )
        )
        self.verbose = verbose
        self.prompter = LLM(self.prompter_config, verbose=self.verbose)
    def batch_to_context(self, batch): # convert text to seq, moving to corresponding device
        model_map = dict(
            instruct = self.prompter,
        )
        context = dotdict()
        for key, model in model_map.items():
            if key in batch.keys():
                context[key] = Seq(
                    text = batch[key],
                    tokenizer = model.tokenizer,
                    device = model.device
                )
            else:
                context[key] = None
        return context
    
    def mutate(self, prompt:str,  max_new_tokens = 50):
        batch = dotdict()
        batch['instruct'] = [prompt]

        context = self.batch_to_context(batch)
        instruct = context.instruct

        prompter_ar = self.prompter.generate_autoregressive(
            key="suffix",
            max_new_tokens=max_new_tokens,
            instruct=instruct,
        )
        suffix = prompter_ar.response_sample

        full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(
            merge_dtype="ids"
        )

        return full_instruct.text[0]
