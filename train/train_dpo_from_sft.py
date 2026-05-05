"""
train_dpo_from_sft.py

Stage 2 of the sequential SFT → DPO ablation.

Pure DPO (no SFT term, no KL) starting from a Stage 1 SFT checkpoint. The
ref model defaults to the same Stage 1 checkpoint — this is the standard
post-training setup (InstructGPT / Llama-3 / DeepSeek): SFT first,
then DPO with the SFT model as the reference policy.

Loss:
    margin_p = β × [(c_θ - c_ref) - (r_θ - r_ref)]
    L_pair   = softplus(-margin_p)
    L_dpo    = (L_pair / std_g.detach()).mean()    # GRPO-style group norm
                                                   # (or plain mean during warmup)

Compared to train_contrastive_dpo_g_normalize.py:
  - NO SFT anchor term  (Stage 1 already did the SFT work)
  - NO sft_scale_mode knob
  - NO KL term          (DPO has implicit KL via ref baseline)
  - NO sft_weight, kl_weight flags
  - --ref_model_path defaults to --model_path (= Stage 1 output) so the
    common case "ref = SFT" needs no extra flag.
  - Group-norm warmup + eps cap retained — std_g still ≈ 0 at step 0
    because trained = ref = SFT, so the cold-start pathology is identical.

Usage (5k smoke; ref scores precomputed once and cached):
    python train/train_dpo_from_sft.py \
        --model_path runs/sft_only_5k/merged \
        --template model/qwen3_soft_switch.jinja2 \
        --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
        --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
        --output_dir runs/dpo_from_sft_5k \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 25 \
        --per_device_batch_size 24 \
        --lr 5e-5 --dpo_beta 0.1 \
        --merge_and_save
"""

import argparse
import hashlib
import json
import time
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from dataset import SYSTEM_PROMPT, build_history_text, sid_to_core_text
from utils import resolve_template

DEFAULT_G = 3


# ===========================================================================
# Callback
# ===========================================================================


class TimingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        now = time.time()
        elapsed = int(now - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        logs["elapsed"] = f"{h}:{m:02d}:{s:02d}"
        steps_since = state.global_step - self.last_log_step
        if steps_since > 0:
            logs["sec/step"] = f"{(now - self.last_log_time) / steps_since:.2f}"
        self.last_log_time = now
        self.last_log_step = state.global_step


# ===========================================================================
# Dataset / collate / sampler
# ===========================================================================


class GRPOPairDataset(Dataset):
    """v1_grpo per-pair view, sorted by (group_id, g) so any G consecutive
    rows form one group — required by GroupedSampler."""

    def __init__(self, parquet_path, tokenizer, G_per_group=DEFAULT_G,
                 max_hist=512, max_total_len=3072):
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(["group_id", "g"]).reset_index(drop=True)
        assert len(df) % G_per_group == 0, (
            f"len={len(df)} not divisible by G={G_per_group}"
        )
        self.df = df
        self.tokenizer = tokenizer
        self.G_per_group = G_per_group
        self.max_hist = max_hist
        self.max_total_len = max_total_len
        self.ref_scores = None  # populated by precompute_ref_scores

    def __len__(self):
        return len(self.df)

    @property
    def num_groups(self):
        return len(self.df) // self.G_per_group

    def subsample_groups(self, n_groups, seed):
        if n_groups >= self.num_groups:
            return self
        rng = torch.Generator().manual_seed(seed)
        all_groups = torch.randperm(self.num_groups, generator=rng).tolist()
        chosen = sorted(all_groups[:n_groups])
        keep_indices = []
        for gid in chosen:
            for g in range(self.G_per_group):
                keep_indices.append(gid * self.G_per_group + g)
        new_df = self.df.iloc[keep_indices].reset_index(drop=True)
        new_ds = object.__new__(GRPOPairDataset)
        new_ds.df = new_df
        new_ds.tokenizer = self.tokenizer
        new_ds.G_per_group = self.G_per_group
        new_ds.max_hist = self.max_hist
        new_ds.max_total_len = self.max_total_len
        new_ds.ref_scores = None  # always subsample BEFORE precompute
        return new_ds

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hist_sids = row["hist_sids"]
        chosen_sid = row["chosen_sid"]
        rejected_sid = row["rejected_sid"]
        group_id = int(row["group_id"])

        hist_text = build_history_text(hist_sids, max_hist=self.max_hist)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": hist_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<|sid_begin|>"

        chosen_text = prompt + sid_to_core_text(chosen_sid)
        rejected_text = prompt + sid_to_core_text(rejected_sid)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        chosen_ids = self.tokenizer(chosen_text, add_special_tokens=True)["input_ids"]
        rejected_ids = self.tokenizer(rejected_text, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        assert len(chosen_ids) == prompt_len + 3
        assert len(rejected_ids) == prompt_len + 3

        if self.max_total_len is not None and len(chosen_ids) > self.max_total_len:
            cut = len(chosen_ids) - self.max_total_len
            chosen_ids = chosen_ids[cut:]
            rejected_ids = rejected_ids[cut:]
            prompt_len = prompt_len - cut

        item = {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt_len": prompt_len,
            "group_id": group_id,
        }
        if self.ref_scores is not None:
            item["ref_chosen_score"] = float(self.ref_scores[idx, 0].item())
            item["ref_rejected_score"] = float(self.ref_scores[idx, 1].item())
        return item


def grpo_collate(batch, pad_token_id):
    B = len(batch)
    chosen = [b["chosen_input_ids"] for b in batch]
    rejected = [b["rejected_input_ids"] for b in batch]
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
    group_ids = torch.tensor([b["group_id"] for b in batch], dtype=torch.long)

    all_seqs = chosen + rejected
    max_len = max(s.size(0) for s in all_seqs)
    input_ids = torch.full((2 * B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((2 * B, max_len), dtype=torch.long)
    for i, s in enumerate(all_seqs):
        input_ids[i, :s.size(0)] = s
        attention_mask[i, :s.size(0)] = 1

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens,
        "group_ids": group_ids,
    }
    if "ref_chosen_score" in batch[0]:
        out["ref_chosen_scores"] = torch.tensor(
            [b["ref_chosen_score"] for b in batch], dtype=torch.float32,
        )
        out["ref_rejected_scores"] = torch.tensor(
            [b["ref_rejected_score"] for b in batch], dtype=torch.float32,
        )
    return out


class GroupedSampler(Sampler):
    def __init__(self, num_samples, G_per_group=DEFAULT_G, shuffle=True, seed=42):
        assert num_samples % G_per_group == 0
        self.num_samples = num_samples
        self.G = G_per_group
        self.num_groups = num_samples // G_per_group
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.num_groups, generator=gen).tolist()
        else:
            order = list(range(self.num_groups))
        for gid in order:
            for g in range(self.G):
                yield gid * self.G + g

    def __len__(self):
        return self.num_samples


# ===========================================================================
# Score / loss
# ===========================================================================


def compute_sid_logprobs(last_hidden, lm_head, input_ids, prompt_lens,
                         num_sid_tokens=3):
    """Returns (chosen_score, rejected_score) given (2B, L) packed input."""
    bsz = input_ids.size(0)
    half = bsz // 2
    H = last_hidden.size(-1)
    prompt_lens = prompt_lens.to(last_hidden.device, non_blocking=True)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)
    ks = torch.arange(num_sid_tokens, device=last_hidden.device)
    pred_pos = plens.unsqueeze(1) - 1 + ks
    token_pos = plens.unsqueeze(1) + ks

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)
    slice_logits = lm_head(slice_hidden).float()      # (2B, K, V) fp32

    target_ids = input_ids.gather(1, token_pos)
    token_logits = slice_logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    log_norm = torch.logsumexp(slice_logits, dim=-1)
    token_logp = token_logits - log_norm

    scores = token_logp.mean(dim=1)
    return scores[:half], scores[half:]


def compute_ref_scores(ref_model, input_ids, attention_mask, prompt_lens):
    transformer = ref_model.model
    lm_head = ref_model.lm_head
    with torch.no_grad():
        out = transformer(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        )
        ref_c, ref_r = compute_sid_logprobs(
            out.last_hidden_state, lm_head, input_ids, prompt_lens,
        )
    return ref_c.detach(), ref_r.detach()


def compute_dpo_loss(trained_chosen, trained_rejected,
                     ref_chosen, ref_rejected,
                     dpo_beta, G_per_group, eps=0.05, apply_group_norm=True):
    """
    margin_p = β × [(c_θ - c_ref) - (r_θ - r_ref)]
    l_pair   = softplus(-margin_p)
    L_dpo    = (l_pair / std_g.detach()+eps).mean()       if apply_group_norm
             = l_pair.mean()                              else (warmup)
    """
    margin = dpo_beta * (
        (trained_chosen - ref_chosen) - (trained_rejected - ref_rejected)
    )
    l_pair = F.softplus(-margin)
    n = l_pair.size(0)
    assert n % G_per_group == 0
    B = n // G_per_group
    l_grouped = l_pair.view(B, G_per_group)

    if apply_group_norm:
        std_g = l_grouped.std(dim=-1, keepdim=True, unbiased=False).detach() + eps
        l_dpo = (l_grouped / std_g).mean()
        inv_std_mean = (1.0 / std_g).mean().detach()
    else:
        l_dpo = l_grouped.mean()
        inv_std_mean = torch.tensor(1.0, device=l_pair.device)

    return l_dpo, l_pair.mean(), margin.mean(), inv_std_mean


# ===========================================================================
# Ref-score precompute + disk cache
# ===========================================================================


def precompute_ref_scores(ref_model, dataset, pad_token_id, batch_size,
                          device="cuda:0"):
    collate = partial(grpo_collate, pad_token_id=pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate, drop_last=False)
    out = torch.empty((len(dataset), 2), dtype=torch.float32)
    pos = 0
    ref_model.eval()
    for batch in tqdm(loader, desc=f"precompute ref ({len(dataset)} pairs)"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompt_lens = batch["prompt_lens"]
        bsz = prompt_lens.size(0)
        ref_c, ref_r = compute_ref_scores(
            ref_model, input_ids, attention_mask, prompt_lens,
        )
        out[pos:pos + bsz, 0] = ref_c.float().cpu()
        out[pos:pos + bsz, 1] = ref_r.float().cpu()
        pos += bsz
    assert pos == len(dataset)
    return out


def _ref_cache_path(cache_dir, split, parquet_path,
                    n_groups, subsample_seed,
                    max_hist, max_total_len, ref_model_path):
    parquet_path = Path(parquet_path)
    psize = parquet_path.stat().st_size if parquet_path.exists() else 0
    pmtime = int(parquet_path.stat().st_mtime) if parquet_path.exists() else 0
    key = "|".join(map(str, [
        parquet_path.resolve(), psize, pmtime,
        n_groups, subsample_seed,
        max_hist, max_total_len,
        ref_model_path,
    ]))
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"ref_{split}_n{n_groups}_{h}.pt"


def load_or_compute_ref_scores(ref_model, dataset, parquet_path, split,
                               n_groups, subsample_seed,
                               max_hist, max_total_len, ref_model_path,
                               batch_size, pad_token_id,
                               cache_dir, use_cache=True):
    cache_path = _ref_cache_path(
        cache_dir, split, parquet_path, n_groups, subsample_seed,
        max_hist, max_total_len, ref_model_path,
    )
    if use_cache and cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu",
                                 weights_only=False)
            scores = payload["scores"]
            if scores.shape == (len(dataset), 2):
                print(f"  [cache HIT] {split}: loaded {scores.shape[0]} "
                      f"ref scores from {cache_path.name}")
                return scores
            print(f"  [cache mismatch] {split}: shape "
                  f"{tuple(scores.shape)} vs ({len(dataset)}, 2) — recomputing")
        except Exception as e:
            print(f"  [cache load failed] {split}: "
                  f"{type(e).__name__}: {e} — recomputing")

    print(f"  [cache MISS] {split}: computing ref scores fresh "
          f"({len(dataset)} pairs) ...")
    scores = precompute_ref_scores(ref_model, dataset, pad_token_id, batch_size)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "scores": scores,
            "n_pairs": len(dataset),
            "n_groups_requested": n_groups,
            "subsample_seed": subsample_seed,
            "max_hist": max_hist,
            "max_total_len": max_total_len,
            "ref_model_path": str(ref_model_path),
            "parquet_path": str(parquet_path),
        }, cache_path)
        print(f"  [cache saved] {cache_path}")

    return scores


# ===========================================================================
# Trainer
# ===========================================================================


class DPOFromSFTTrainer(Trainer):
    """Pure DPO (no SFT, no KL). Uses cached ref scores when available."""

    def __init__(self, *args, dpo_beta=0.1,
                 group_norm_eps=0.05, group_norm_warmup=50,
                 G_per_group=DEFAULT_G, sampler_seed=42,
                 ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_beta = dpo_beta
        self.group_norm_eps = group_norm_eps
        self.group_norm_warmup = group_norm_warmup
        self.G_per_group = G_per_group
        self.sampler_seed = sampler_seed
        self.ref_model = ref_model  # None when ref scores are pre-cached

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        return GroupedSampler(len(ds), self.G_per_group,
                              shuffle=True, seed=self.sampler_seed)

    def _get_eval_sampler(self, eval_dataset):
        return GroupedSampler(len(eval_dataset), self.G_per_group,
                              shuffle=False, seed=self.sampler_seed)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        prompt_lens = inputs.pop("prompt_lens")
        group_ids = inputs.pop("group_ids", None)
        ref_chosen_cached = inputs.pop("ref_chosen_scores", None)
        ref_rejected_cached = inputs.pop("ref_rejected_scores", None)

        causal_lm = (
            model.get_base_model() if hasattr(model, "get_base_model") else model
        )
        transformer = causal_lm.model
        lm_head = causal_lm.lm_head

        out = transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        trained_chosen, trained_rejected = compute_sid_logprobs(
            out.last_hidden_state, lm_head, inputs["input_ids"], prompt_lens,
        )

        if ref_chosen_cached is not None:
            ref_chosen = ref_chosen_cached.to(trained_chosen.device,
                                              non_blocking=True)
            ref_rejected = ref_rejected_cached.to(trained_chosen.device,
                                                  non_blocking=True)
        else:
            assert self.ref_model is not None, (
                "ref_model is None and batch lacks cached ref scores"
            )
            ref_chosen, ref_rejected = compute_ref_scores(
                self.ref_model,
                inputs["input_ids"], inputs["attention_mask"], prompt_lens,
            )

        in_warmup = self.state.global_step < self.group_norm_warmup
        l_dpo, l_pair_mean, mean_margin, inv_std_mean = compute_dpo_loss(
            trained_chosen, trained_rejected,
            ref_chosen, ref_rejected,
            self.dpo_beta, self.G_per_group,
            eps=self.group_norm_eps,
            apply_group_norm=not in_warmup,
        )
        loss = l_dpo

        # Re-attach popped keys for prediction_step.
        inputs["prompt_lens"] = prompt_lens
        if group_ids is not None:
            inputs["group_ids"] = group_ids
        if ref_chosen_cached is not None:
            inputs["ref_chosen_scores"] = ref_chosen_cached
            inputs["ref_rejected_scores"] = ref_rejected_cached

        if return_outputs:
            scores = torch.stack([trained_chosen, trained_rejected], dim=-1)
            return loss, {"scores": scores}

        if self.state.global_step % 100 == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            warmup_tag = " [warmup]" if in_warmup else ""
            print(f"[mem] step={self.state.global_step}{warmup_tag} "
                  f"alloc={alloc:.2f}G peak={peak:.2f}G "
                  f"l_dpo={l_dpo.item():.3f} l_pair={l_pair_mean.item():.3f} "
                  f"margin={mean_margin.item():.3f} "
                  f"inv_std={inv_std_mean.item():.2f}")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only,
                        ignore_keys=None):
        with torch.no_grad():
            loss, extras = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        labels = torch.zeros(
            extras["scores"].size(0), dtype=torch.long,
            device=extras["scores"].device,
        )
        return (loss.detach(), extras["scores"].detach(), labels)


def compute_metrics(eval_pred):
    scores = eval_pred.predictions
    if isinstance(scores, tuple):
        scores = scores[0]
    pref_acc = float((scores[:, 0] > scores[:, 1]).mean())
    margin = float((scores[:, 0] - scores[:, 1]).mean())
    chosen_score = float(scores[:, 0].mean())
    rejected_score = float(scores[:, 1].mean())
    return {
        "pref_acc": pref_acc,
        "margin": margin,
        "chosen_score": chosen_score,
        "rejected_score": rejected_score,
    }


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Stage 1 SFT checkpoint (its merged/ dir).")
    parser.add_argument("--ref_model_path", default=None,
                        help="Default = --model_path. Standard SFT→DPO uses "
                             "the SFT model as the reference policy.")
    parser.add_argument("--template", default=None)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--valid_parquet", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--G", type=int, default=DEFAULT_G,
                        help="Pairs per group; must match the v1_grpo "
                             "dataset's G. Default 3.")

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=24,
                        help="Pairs per micro-batch; must be a multiple of --G.")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=-1)

    # DPO
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                        help="DPO scaling. Standard 0.1 (DeepSeek/Llama-3).")
    parser.add_argument("--group_norm_eps", type=float, default=0.05,
                        help="Cap on 1/std amplification (default 20×).")
    parser.add_argument("--group_norm_warmup", type=int, default=50,
                        help="Steps where group-norm is disabled (cold-start "
                             "fix: trained=ref=SFT → std≈0 at step 0).")

    # Ref cache
    parser.add_argument("--ref_cache_dir", default="",
                        help="Empty (default) → "
                             "<train_parquet's parent>/_ref_cache/. Cache "
                             "filename hashes (parquet, ref_model, "
                             "subsample, max_hist, max_total_len) so changing "
                             "any of them auto-invalidates.")
    parser.add_argument("--no_ref_cache", action="store_true")

    # Data
    parser.add_argument("--max_hist", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=3072)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_groups", type=int, default=-1,
                        help="Subsample to this many groups. -1 = full.")
    parser.add_argument("--max_eval_groups", type=int, default=1000)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--merge_and_save", action="store_true")

    # Logging / scheduling
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="If > 0, schedule N evenly-spaced ckpts + evals.")
    parser.add_argument("--eval_steps", type=int, default=1500)
    parser.add_argument("--save_steps", type=int, default=1500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--best_metric",
        choices=["chosen_score", "pref_acc", "margin"],
        default="chosen_score",
        help="Default = chosen_score. Reasoning: Stage 1 SFT pushed chosen "
             "up; Stage 2 DPO must NOT undo that gain. The previously-seen "
             "GRPO-only failure mode (chosen 0.0093 → 0.0043, while pref_acc "
             "rose) would be REWARDED by metric_for_best_model=pref_acc and "
             "load_best_model_at_end would silently pick a ckpt where SFT's "
             "gain has been thrown away. chosen_score correlates with "
             "recall_chosen and protects against that. (You can still inspect "
             "all 5 ckpts post-hoc with diagnose/checkpoint_recall_trend.py.)",
    )

    args = parser.parse_args()

    assert args.per_device_batch_size % args.G == 0, (
        f"--per_device_batch_size ({args.per_device_batch_size}) must be a "
        f"multiple of --G ({args.G})."
    )

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA not available"

    print(f"Loading tokenizer + model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    template_path = resolve_template(args.template)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"[attn] using attn_implementation={attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": "cuda:0"},
        attn_implementation=attn_impl,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    print("Applying LoRA adapters ...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.4f}%)")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print(f"Loading datasets (G={args.G}) ...")
    train_set = GRPOPairDataset(args.train_parquet, tokenizer,
                                G_per_group=args.G,
                                max_hist=args.max_hist,
                                max_total_len=args.max_total_len)
    valid_set = GRPOPairDataset(args.valid_parquet, tokenizer,
                                G_per_group=args.G,
                                max_hist=args.max_hist,
                                max_total_len=args.max_total_len)
    train_full = train_set.num_groups
    valid_full = valid_set.num_groups
    if args.max_train_groups > 0:
        train_set = train_set.subsample_groups(args.max_train_groups, seed=12345)
    if args.max_eval_groups > 0:
        valid_set = valid_set.subsample_groups(args.max_eval_groups, seed=67890)

    # Auto-schedule N evenly-spaced ckpts.
    if args.num_checkpoints > 0:
        batch_per_step = args.per_device_batch_size * args.grad_accum
        steps_per_epoch = max(1, len(train_set) // batch_per_step)
        if args.max_steps > 0:
            total_steps = args.max_steps
        else:
            total_steps = int(steps_per_epoch * args.epochs)
        interval = max(1, total_steps // args.num_checkpoints)
        args.save_steps = interval
        args.eval_steps = interval
        args.save_total_limit = args.num_checkpoints
        print(f"  schedule: {args.num_checkpoints} ckpts over "
              f"{total_steps} total steps → save+eval every {interval} steps")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"  train: {train_set.num_groups:,}/{train_full:,} groups "
          f"({len(train_set):,} pairs)")
    print(f"  valid: {valid_set.num_groups:,}/{valid_full:,} groups "
          f"({len(valid_set):,} pairs)")

    collate = partial(grpo_collate, pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        bf16=(args.dtype == "bf16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.best_metric}",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["none"],
        seed=args.seed,
        optim="adamw_torch_fused",
    )

    # ---- Reference model ----
    ref_path = args.ref_model_path or args.model_path
    print(f"Loading frozen reference model from {ref_path} ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": "cuda:0"},
        attn_implementation=attn_impl,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Pre-compute ref scores → cache → free ref_model.
    pre_bs = args.per_device_batch_size * 2
    cache_dir = (
        Path(args.ref_cache_dir) if args.ref_cache_dir
        else Path(args.train_parquet).parent / "_ref_cache"
    )
    use_cache = not args.no_ref_cache
    print(f"\nRef scores (batch={pre_bs}, "
          f"{'cache: ' + str(cache_dir) if use_cache else 'no cache'}):")
    train_set.ref_scores = load_or_compute_ref_scores(
        ref_model, train_set, args.train_parquet, "train",
        n_groups=args.max_train_groups, subsample_seed=12345,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
        ref_model_path=ref_path,
        batch_size=pre_bs, pad_token_id=tokenizer.pad_token_id,
        cache_dir=cache_dir, use_cache=use_cache,
    )
    valid_set.ref_scores = load_or_compute_ref_scores(
        ref_model, valid_set, args.valid_parquet, "valid",
        n_groups=args.max_eval_groups, subsample_seed=67890,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
        ref_model_path=ref_path,
        batch_size=pre_bs, pad_token_id=tokenizer.pad_token_id,
        cache_dir=cache_dir, use_cache=use_cache,
    )
    print("Freeing ref_model (cached scores cover all downstream needs).")
    del ref_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    trainer = DPOFromSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        dpo_beta=args.dpo_beta,
        group_norm_eps=args.group_norm_eps,
        group_norm_warmup=args.group_norm_warmup,
        G_per_group=args.G,
        sampler_seed=args.seed,
        ref_model=None,  # using cached ref scores
        callbacks=[TimingCallback()],
    )

    print("\n===== Training =====")
    print(f"  loss = L_dpo_grpo (β={args.dpo_beta}, eps={args.group_norm_eps}, "
          f"warmup={args.group_norm_warmup})    [no SFT, no KL]")
    print(f"  ref  = {ref_path}")
    print(f"  best metric: eval_{args.best_metric} (greater is better)")
    trainer.train()

    print("\n===== Final eval =====")
    metrics = trainer.evaluate()

    def _safe(v):
        if isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    print(json.dumps({k: _safe(v) for k, v in metrics.items()}, indent=2))

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nSaved LoRA adapter to {adapter_dir}")

    if args.merge_and_save:
        print("Merging LoRA into base weights ...")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        # Merge against the SAME base used for training (= Stage 1 merged
        # checkpoint, passed as --model_path). NOT the original OneRec base.
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map={"": "cuda:0"},
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        merged_dir = out_dir / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Saved merged checkpoint to {merged_dir}")


if __name__ == "__main__":
    main()