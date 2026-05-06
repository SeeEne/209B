"""
train_contrastive_dpo_g_normalize.py

DPO + SFT anchor + GRPO-style group normalization, on contrastive_dataset_v1_grpo.

Loss:
    L = L_dpo_grpo  +  sft_weight × scale × L_sft  [+ kl_weight × KL]

  - L_dpo_grpo: DPO with ref-baseline margin, divided by per-group std
    (group-norm eps capped at 1/eps; warmup steps use plain mean).
        margin = β × [(c_θ − c_ref) − (r_θ − r_ref)]
        l_pair = softplus(−margin)
        L_dpo_grpo = (l_pair / std_g).mean()      # group-normalized
  - L_sft = -chosen_θ.mean(): pushes absolute P(chosen) up — without it,
    contrastive can be minimized by lowering both sides.
  - scale: when --sft_scale_mode match_dpo (default), scale = mean(1/std_g),
    so sft_weight is in the same units as L_dpo (sft_weight=1.0 means SFT
    contributes ~as much as DPO). When --sft_scale_mode raw, scale = 1.0
    (legacy behavior; sft_weight then under-weights SFT by ~10-30×).
  - KL: optional explicit token-level KL (default off — DPO has implicit KL).

DESIGN NOTES (post-2026-05-04 fix; see archive/EXPERIMENTS.md):
  The first DPO smoke run had eval_pref_acc rising (0.607 → 0.642) while
  recall_chosen DROPPED (0.0063 → 0.0043). Root cause: SFT was effectively
  weightless (1-3% of total loss) because GRPO amplifies L_dpo by ~10-30×.
  Three fixes here:
    1. group_norm_eps: 1e-3 → 0.05 (cap at 20× instead of 1000×; fixes
       step-0 explosion of l_dpo from 0.69 → 693).
    2. group_norm_warmup=50: skip group-norm during cold start when std≈0.
    3. sft_scale_mode=match_dpo: rescale L_sft by 1/std_g so sft_weight
       has direct semantic meaning relative to DPO contribution.
    Also: best_metric default switched from pref_acc → chosen_score
    (chosen_score correlates with recall_chosen; pref_acc was decoupled).

Run on a single CUDA GPU.

Usage (smoke, ~5h on RTX 6000 Pro):
    python train/train_contrastive_dpo_g_normalize.py \
        --model_path model/OneRec-1.7B \
        --template model/qwen3_soft_switch.jinja2 \
        --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
        --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
        --output_dir runs/dpo_grpo_smoke \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --eval_steps 200 --save_steps 200 --logging_steps 25 \
        --per_device_batch_size 12 --grad_accum 1 \
        --lr 5e-5 --dpo_beta 0.1 --sft_weight 1.0 --kl_weight 0 \
        --merge_and_save
"""

import argparse
import hashlib
import json
import time
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from dataset import GroupedSampler, PairedSIDDataset, paired_collate
from utils import resolve_template

# Default group size — overridable via --G. Must match the build_contrastive_
# dataset_GRPO.py setting that produced --train_parquet / --valid_parquet.
DEFAULT_G = 3


# ===========================================================================
# Callback
# ===========================================================================


class TimingCallback(TrainerCallback):
    """Adds elapsed wall-clock and per-step time to each Trainer log line."""

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


# Dataset (PairedSIDDataset), collate (paired_collate) and GroupedSampler are
# imported from dataset.py — shared with train_dpo_from_sft.py and train_orpo.py.


# ===========================================================================
# Score / loss
# ===========================================================================


def compute_sid_logprobs(last_hidden, lm_head, input_ids, prompt_lens,
                         num_sid_tokens=3, return_full_log_probs=False):
    """Length=1 score: 3 SID tokens per (chosen|rejected) row.

    Returns (chosen_score, rejected_score, slice_log_probs):
        scores         : (B,) chosen + (B,) rejected
        slice_log_probs: (2B, K, V) log-probs at SID positions, OR None.

    `return_full_log_probs` controls whether the full (2B, K, V) fp32
    log-softmax is materialized. False (default) saves ~44 MB per call by
    using `log P(target) = logits[target] - logsumexp(logits)` which only
    needs the per-position normalizer scalar. Full log_probs are only
    required when explicit token-level KL is enabled (kl_weight > 0).
    """
    bsz = input_ids.size(0)
    half = bsz // 2
    H = last_hidden.size(-1)
    # prompt_lens may arrive on CPU (precompute_ref_scores uses a raw
    # DataLoader; HF Trainer normally moves it via _prepare_inputs).
    # Align to last_hidden's device so the broadcast below works in both paths.
    prompt_lens = prompt_lens.to(last_hidden.device, non_blocking=True)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)
    ks = torch.arange(num_sid_tokens, device=last_hidden.device)
    pred_pos = plens.unsqueeze(1) - 1 + ks
    token_pos = plens.unsqueeze(1) + ks

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)
    slice_logits = lm_head(slice_hidden)             # (2B, K, V) bf16
    slice_logits_fp32 = slice_logits.float()         # (2B, K, V) fp32 — needed for stable softmax over 152k vocab

    target_ids = input_ids.gather(1, token_pos)
    if return_full_log_probs:
        log_probs = F.log_softmax(slice_logits_fp32, dim=-1)
        token_logp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    else:
        # log P(target) = logits[target] − logsumexp(logits)
        # Equivalent to log_softmax + gather, but skips the (2B,K,V) tensor.
        log_probs = None
        token_logits = slice_logits_fp32.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        log_norm = torch.logsumexp(slice_logits_fp32, dim=-1)
        token_logp = token_logits - log_norm

    scores = token_logp.mean(dim=1)
    return scores[:half], scores[half:], log_probs


def compute_ref_outputs(ref_model, input_ids, attention_mask, prompt_lens,
                        num_sid_tokens=3, return_full_log_probs=False):
    """
    Frozen ref forward with lm_head bypass. Returns ref's per-sequence
    scores AND (optionally) full slice log_probs (only needed when KL on).

    All outputs are detached — no grad through ref.
    """
    transformer = ref_model.model
    lm_head = ref_model.lm_head
    with torch.no_grad():
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_hidden = out.last_hidden_state
        ref_chosen, ref_rejected, ref_log_probs = compute_sid_logprobs(
            last_hidden, lm_head, input_ids, prompt_lens, num_sid_tokens,
            return_full_log_probs=return_full_log_probs,
        )
    ref_log_probs = ref_log_probs.detach() if ref_log_probs is not None else None
    return ref_chosen.detach(), ref_rejected.detach(), ref_log_probs


def compute_dpo_grpo_loss(trained_chosen, trained_rejected,
                          ref_chosen, ref_rejected,
                          dpo_beta, G_per_group,
                          eps=0.05, apply_group_norm=True):
    """
    DPO loss with optional GRPO-style per-group std normalization.

    DPO margin (per pair):
        margin_p = β × [(chosen_θ - chosen_ref) - (rejected_θ - rejected_ref)]
        L_pair_p = softplus(-margin_p)

    GRPO add-on (when apply_group_norm=True): divide each pair's L by per-group
    std (detached as baseline), then mean over all pairs. eps caps the
    1/std amplification at 1/eps (default 0.05 → cap at 20×; legacy 1e-3 →
    cap at 1000× which causes step-0 explosion when trained=ref → std≈0).

    When apply_group_norm=False (warmup): uses plain mean(L_pair). This
    avoids the cold-start pathology where std≈0 makes the first gradient
    update dominated by 1/eps.

    Returns (l_grpo, l_pair_mean, mean_margin, inv_std_mean):
      - l_grpo: the loss term to add to total
      - l_pair_mean, mean_margin: monitoring only
      - inv_std_mean: detached mean of 1/std_g (= 1.0 when no group-norm).
        Used by trainer to rescale L_sft so it matches L_dpo magnitude.
    """
    margin = dpo_beta * (
        (trained_chosen - ref_chosen) - (trained_rejected - ref_rejected)
    )
    l_pair = F.softplus(-margin)        # (B*G,)

    n = l_pair.size(0)
    assert n % G_per_group == 0, (
        f"batch pairs {n} not divisible by G={G_per_group}"
    )
    B = n // G_per_group
    l_grouped = l_pair.view(B, G_per_group)

    if apply_group_norm:
        std_g = l_grouped.std(dim=-1, keepdim=True, unbiased=False).detach() + eps
        l_grpo = (l_grouped / std_g).mean()
        inv_std_mean = (1.0 / std_g).mean().detach()
    else:
        l_grpo = l_grouped.mean()
        inv_std_mean = torch.tensor(1.0, device=l_pair.device)

    return l_grpo, l_pair.mean(), margin.mean(), inv_std_mean


# ===========================================================================
# Pre-compute ref scores (run-once, then ref_model can be freed)
# ===========================================================================


def precompute_ref_scores(ref_model, dataset, pad_token_id, batch_size,
                          device="cuda:0"):
    """
    One pass over `dataset` with the frozen ref_model to cache per-pair
    (ref_chosen_score, ref_rejected_score). Returned as a (N, 2) fp32 CPU
    tensor with row order matching dataset.df.

    Saves one ref forward per training step downstream — ref_model can be
    freed after this call when kl_weight == 0.
    """
    collate = partial(paired_collate, pad_token_id=pad_token_id)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate, drop_last=False,
    )
    out = torch.empty((len(dataset), 2), dtype=torch.float32)
    pos = 0
    ref_model.eval()
    for batch in tqdm(loader, desc=f"precompute ref ({len(dataset)} pairs)"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompt_lens = batch["prompt_lens"]
        bsz = prompt_lens.size(0)
        ref_c, ref_r, _ = compute_ref_outputs(
            ref_model, input_ids, attention_mask, prompt_lens,
            return_full_log_probs=False,
        )
        out[pos:pos + bsz, 0] = ref_c.float().cpu()
        out[pos:pos + bsz, 1] = ref_r.float().cpu()
        pos += bsz
    assert pos == len(dataset), f"precompute saw {pos} pairs, expected {len(dataset)}"
    return out


def _ref_cache_path(cache_dir, split_name, parquet_path,
                    n_groups_requested, subsample_seed,
                    max_hist, max_total_len, ref_model_path):
    """
    Compute cache filename. The hash covers EVERY param that affects ref
    scores: parquet identity (path/size/mtime), subsample (seed + n_groups),
    preprocessing (max_hist/max_total_len), and ref_model_path. Any of these
    changes → different filename → automatic cache miss + recompute.
    """
    parquet_path = Path(parquet_path)
    psize = parquet_path.stat().st_size if parquet_path.exists() else 0
    pmtime = int(parquet_path.stat().st_mtime) if parquet_path.exists() else 0
    key = "|".join(map(str, [
        parquet_path.resolve(), psize, pmtime,
        n_groups_requested, subsample_seed,
        max_hist, max_total_len,
        ref_model_path,
    ]))
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"ref_{split_name}_n{n_groups_requested}_{h}.pt"


def load_or_compute_ref_scores(ref_model, dataset, parquet_path, split_name,
                               n_groups_requested, subsample_seed,
                               max_hist, max_total_len, ref_model_path,
                               batch_size, pad_token_id,
                               cache_dir, use_cache=True):
    """
    Disk-cached wrapper around precompute_ref_scores. Cache hit → 1 sec
    load instead of ~30 min recompute. Cache key encodes all params that
    affect the result, so changing any of them auto-invalidates.
    """
    cache_path = _ref_cache_path(
        cache_dir, split_name, parquet_path,
        n_groups_requested, subsample_seed,
        max_hist, max_total_len, ref_model_path,
    )

    if use_cache and cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu",
                                 weights_only=False)
            scores = payload["scores"]
            if scores.shape == (len(dataset), 2):
                print(f"  [cache HIT] {split_name}: loaded {scores.shape[0]} "
                      f"ref scores from {cache_path.name}")
                return scores
            print(f"  [cache mismatch] {split_name}: shape "
                  f"{tuple(scores.shape)} vs ({len(dataset)}, 2) — recomputing")
        except Exception as e:
            print(f"  [cache load failed] {split_name}: "
                  f"{type(e).__name__}: {e} — recomputing")

    print(f"  [cache MISS] {split_name}: computing ref scores fresh "
          f"({len(dataset)} pairs) ...")
    scores = precompute_ref_scores(
        ref_model, dataset, pad_token_id, batch_size,
    )

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "scores": scores,
            "n_pairs": len(dataset),
            "n_groups_requested": n_groups_requested,
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


class ContrastiveTrainerDPOGRPO(Trainer):
    def __init__(self, *args,
                 dpo_beta=0.1, sft_weight=1.0, kl_weight=0.0,
                 sft_scale_mode="match_dpo",
                 group_norm_eps=0.05, group_norm_warmup=50,
                 ref_model=None, G_per_group=DEFAULT_G, sampler_seed=42,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # ref_model may be None when ref scores are pre-cached on the dataset
        # (precompute_ref_scores). KL > 0 still needs live ref forward for
        # full log_probs, so it can't run with ref_model=None.
        if ref_model is None and kl_weight > 0:
            raise ValueError(
                "kl_weight > 0 requires a live ref_model (can't pre-cache "
                "the full (B,K,V) log_probs). Either set kl_weight=0 or "
                "skip pre-compute (--precompute_ref off)."
            )
        assert sft_scale_mode in ("raw", "match_dpo"), (
            f"unknown sft_scale_mode: {sft_scale_mode}"
        )
        self.dpo_beta = dpo_beta
        self.sft_weight = sft_weight
        self.kl_weight = kl_weight
        self.sft_scale_mode = sft_scale_mode
        self.group_norm_eps = group_norm_eps
        self.group_norm_warmup = group_norm_warmup
        self.ref_model = ref_model
        self.G_per_group = G_per_group
        self.sampler_seed = sampler_seed

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        return GroupedSampler(
            num_samples=len(ds),
            G_per_group=self.G_per_group,
            shuffle=True,
            seed=self.sampler_seed,
        )

    def _get_eval_sampler(self, eval_dataset):
        return GroupedSampler(
            num_samples=len(eval_dataset),
            G_per_group=self.G_per_group,
            shuffle=False,
            seed=self.sampler_seed,
        )

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        prompt_lens = inputs.pop("prompt_lens")
        group_ids = inputs.pop("group_ids", None)
        ref_chosen_cached = inputs.pop("ref_chosen_scores", None)
        ref_rejected_cached = inputs.pop("ref_rejected_scores", None)
        need_log_probs = self.kl_weight > 0  # full (B,K,V) only needed for KL

        # ---- Trained forward (with grad) — lm_head bypass to skip full logits ----
        causal_lm = (
            model.get_base_model() if hasattr(model, "get_base_model") else model
        )
        transformer = causal_lm.model
        lm_head = causal_lm.lm_head

        transformer_out = transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        last_hidden = transformer_out.last_hidden_state
        trained_chosen, trained_rejected, trained_log_probs = compute_sid_logprobs(
            last_hidden, lm_head, inputs["input_ids"], prompt_lens,
            return_full_log_probs=need_log_probs,
        )

        # ---- Reference scores: cached (preferred) or live forward ----
        if ref_chosen_cached is not None and not need_log_probs:
            ref_chosen = ref_chosen_cached.to(trained_chosen.device,
                                              non_blocking=True)
            ref_rejected = ref_rejected_cached.to(trained_chosen.device,
                                                  non_blocking=True)
            ref_log_probs = None
        else:
            assert self.ref_model is not None, (
                "ref_model is None and batch lacks cached ref scores — "
                "pre-compute step missed this dataset?"
            )
            ref_chosen, ref_rejected, ref_log_probs = compute_ref_outputs(
                self.ref_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                prompt_lens,
                return_full_log_probs=need_log_probs,
            )

        # ---- DPO + GRPO normalization (group-norm skipped during warmup) ----
        in_warmup = self.state.global_step < self.group_norm_warmup
        l_dpo, l_pair_mean, mean_margin, inv_std_mean = compute_dpo_grpo_loss(
            trained_chosen, trained_rejected,
            ref_chosen, ref_rejected,
            self.dpo_beta, self.G_per_group,
            eps=self.group_norm_eps,
            apply_group_norm=not in_warmup,
        )
        loss = l_dpo

        # ---- SFT anchor on chosen — pushes |chosen_θ| up in absolute terms ----
        # In match_dpo mode, multiply by current 1/std mean so sft_weight=1.0
        # truly means "SFT contribution ≈ DPO contribution". Without this,
        # GRPO amplifies L_dpo by ~10-30× and SFT becomes a 1-3% nuisance term.
        l_sft = -trained_chosen.mean()
        if self.sft_weight > 0:
            if self.sft_scale_mode == "match_dpo" and not in_warmup:
                sft_term = self.sft_weight * inv_std_mean * l_sft
            else:
                sft_term = self.sft_weight * l_sft
            loss = loss + sft_term

        # ---- Optional explicit KL (default 0; DPO has implicit KL via margin) ----
        l_kl = None
        if self.kl_weight > 0:
            trained_probs = trained_log_probs.exp()
            l_kl = (trained_probs * (trained_log_probs - ref_log_probs)) \
                .sum(dim=-1).mean()
            loss = loss + self.kl_weight * l_kl

        # Re-attach for prediction_step (compute_loss is called from there
        # with the same dict — keep popped keys consistent across calls).
        inputs["prompt_lens"] = prompt_lens
        if group_ids is not None:
            inputs["group_ids"] = group_ids
        if ref_chosen_cached is not None:
            inputs["ref_chosen_scores"] = ref_chosen_cached
            inputs["ref_rejected_scores"] = ref_rejected_cached

        if return_outputs:
            scores = torch.stack([trained_chosen, trained_rejected], dim=-1)
            return loss, {"scores": scores}

        if self.state.global_step % 100 == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            extra_kl = f" l_kl={l_kl.item():.3f}" if l_kl is not None else ""
            warmup_tag = " [warmup]" if in_warmup else ""
            print(f"[mem] step={self.state.global_step}{warmup_tag} "
                  f"alloc={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G "
                  f"l_dpo={l_dpo.item():.3f} l_pair={l_pair_mean.item():.3f} "
                  f"margin={mean_margin.item():.3f} l_sft={l_sft.item():.3f} "
                  f"inv_std={inv_std_mean.item():.2f}"
                  f"{extra_kl}")

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
    """
    Reports four metrics. The model-selection one is `chosen_score` —
    higher = chosen items have higher absolute log-prob, which is the
    proxy correlated with `recall_chosen` at eval time.

    Why not pref_acc: pref_acc only checks "c_θ > r_θ on these specific
    pairs", which can climb (0.607 → 0.642) while absolute chosen recall
    DROPS (0.0063 → 0.0043) because the model lowers BOTH and just
    widens the gap. We saw exactly this in the first DPO smoke run.
    """
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
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--template", default=None)
    parser.add_argument("--train_parquet", required=True,
                        help="v1_grpo train.parquet (built by "
                             "build_contrastive_dataset_GRPO.py)")
    parser.add_argument("--valid_parquet", required=True)
    parser.add_argument("--output_dir", required=True)

    # GRPO group size
    parser.add_argument(
        "--G", type=int, default=DEFAULT_G,
        help=f"Pairs per group. MUST match the G used by "
             f"build_contrastive_dataset_GRPO.py for --train_parquet. "
             f"Default {DEFAULT_G}.",
    )

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--per_device_batch_size", type=int, default=12,
        help="Pairs per micro-batch. MUST be a multiple of --G.",
    )
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # ---- Loss weights ----
    parser.add_argument(
        "--dpo_beta", type=float, default=0.1,
        help="DPO scaling factor. Standard value 0.1 (DeepSeek/Llama-3 use this). "
             "Higher β = more aggressive preference push, also harder for "
             "trained to drift from ref before sigmoid saturates.",
    )
    parser.add_argument(
        "--sft_weight", type=float, default=1.0,
        help="Weight on SFT anchor L_sft = -chosen.mean(). With "
             "--sft_scale_mode match_dpo (default), this is the TRUE relative "
             "magnitude of SFT vs DPO contributions (1.0 = SFT comparable to "
             "DPO; 0.5 = half; etc.). Set 0 to test pure DPO. "
             "[Pre-2026-05-04 default was 0.1, which was effectively ~1-3% "
             "due to GRPO amplifying L_dpo by ~10-30×.]",
    )
    parser.add_argument(
        "--sft_scale_mode", choices=["raw", "match_dpo"], default="match_dpo",
        help="raw   : loss += sft_weight * L_sft (legacy; sft_weight is "
             "semantically meaningless because GRPO amplifies L_dpo). "
             "match_dpo (default): loss += sft_weight * (1/std_g).mean() * "
             "L_sft, so sft_weight is in the same units as L_dpo. "
             "During warmup steps, falls back to raw regardless.",
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.0,
        help="Weight on explicit token-level KL(trained || ref). Default 0 "
             "because DPO already has *implicit* KL via the ref baseline in "
             "the margin (sigmoid saturates when trained drifts far from ref). "
             "Set > 0 only if you want extra constraint.",
    )

    # ---- GRPO normalization knobs ----
    parser.add_argument(
        "--group_norm_eps", type=float, default=0.05,
        help="Cap on 1/std amplification. Default 0.05 → cap at 20×. "
             "[Pre-2026-05-04 was 1e-3 = 1000× cap, which made step 0 "
             "explode: l_pair=0.69 → l_dpo=693 because trained=ref → std≈0.]",
    )
    parser.add_argument(
        "--group_norm_warmup", type=int, default=50,
        help="Steps at the start of training where group-norm is DISABLED "
             "(use plain mean(L_pair) and raw SFT). Avoids cold-start "
             "explosion from std≈0 dominating the first few weight updates.",
    )

    parser.add_argument("--ref_model_path", default=None,
                        help="Default = --model_path (pre-training base).")
    parser.add_argument(
        "--precompute_ref", choices=["auto", "force", "off"], default="auto",
        help="auto (default): pre-compute ref scores once and free ref_model "
             "when kl_weight=0; live ref forward when kl_weight>0. "
             "force: always pre-compute (errors if kl_weight>0). "
             "off:   always live ref forward (legacy, ~×1.4 slower). "
             "Pre-compute saves ONE ref forward per training step "
             "(~30-40% wall-clock speedup) and frees ~3.4 GB ref VRAM.",
    )
    parser.add_argument(
        "--ref_cache_dir", default="",
        help="Where to cache pre-computed ref scores on disk. Empty (default) "
             "→ auto-derive as <train_parquet's parent>/_ref_cache/. Same "
             "(parquet, ref_model, subsample, max_hist, max_total_len) → "
             "cache hit on subsequent runs (1 sec load vs ~30 min recompute). "
             "Cache filename includes a hash of all identity params, so "
             "changing any of them auto-invalidates.",
    )
    parser.add_argument(
        "--no_ref_cache", action="store_true",
        help="Disable disk caching of ref scores (always recompute). Useful "
             "for debugging or one-off configs you don't want to cache.",
    )
    parser.add_argument("--max_steps", type=int, default=-1)

    # Data
    parser.add_argument("--max_hist", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=3072)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_train_groups", type=int, default=-1,
        help="Subsample train to this many GROUPS. -1 = full set. "
             "Smoke: 5000.",
    )
    parser.add_argument(
        "--max_eval_groups", type=int, default=2000,
    )

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

    # Logging
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument(
        "--num_checkpoints", type=int, default=5,
        help="If > 0 (default 5), schedule exactly N checkpoints + evals "
             "evenly spaced across training. Overrides --save_steps, "
             "--eval_steps, --save_total_limit. Convenient for trend "
             "analysis (diagnose/checkpoint_recall_trend.py expects equal "
             "spacing). Set 0 to fall back to explicit --save_steps / "
             "--eval_steps below.",
    )
    parser.add_argument("--eval_steps", type=int, default=1500,
                        help="Ignored when --num_checkpoints > 0.")
    parser.add_argument("--save_steps", type=int, default=1500,
                        help="Ignored when --num_checkpoints > 0.")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Ignored when --num_checkpoints > 0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument(
        "--best_metric",
        choices=["chosen_score", "pref_acc", "margin"],
        default="chosen_score",
        help="Metric used by load_best_model_at_end. chosen_score (default) "
             "is correlated with eval-time recall_chosen — selects the ckpt "
             "where chosen items have highest absolute log-prob. pref_acc "
             "(legacy) only measures pair-wise ordering and was DECOUPLED "
             "from recall_chosen in the first DPO smoke (pref_acc ↑ while "
             "chosen recall ↓). All three are greater-is-better.",
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
    # NB: config.json is written LATER (after auto-schedule mutates
    # save_steps / eval_steps) so the recorded values match what actually ran.

    assert torch.cuda.is_available(), "CUDA not available"

    print(f"Loading tokenizer + base model from {args.model_path} ...")
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
    train_set = PairedSIDDataset(
        args.train_parquet, tokenizer,
        G_per_group=args.G,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
    )
    valid_set = PairedSIDDataset(
        args.valid_parquet, tokenizer,
        G_per_group=args.G,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
    )

    train_full_groups = train_set.num_groups
    valid_full_groups = valid_set.num_groups

    if args.max_train_groups > 0:
        train_set = train_set.subsample_groups(args.max_train_groups, seed=12345)
    if args.max_eval_groups > 0:
        valid_set = valid_set.subsample_groups(args.max_eval_groups, seed=67890)

    def _fmt(now, full):
        return f"{now:,}" + (f" / {full:,}" if now < full else "")

    # ---- Auto-schedule N evenly-spaced checkpoints if requested ----
    # Compute total_steps from FINAL (post-subsample) train set so the
    # interval reflects what will actually run.
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
        print(f"  schedule: {args.num_checkpoints} checkpoints over "
              f"{total_steps} total steps → save+eval every {interval} steps")

    # Now that args reflect the actual running config, persist it.
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"  train: {_fmt(train_set.num_groups, train_full_groups)} groups  "
          f"({len(train_set):,} pairs)")
    print(f"  valid: {_fmt(valid_set.num_groups, valid_full_groups)} groups  "
          f"({len(valid_set):,} pairs)")

    collate = partial(paired_collate, pad_token_id=tokenizer.pad_token_id)

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

    # ---- Reference model: load, then optionally pre-compute scores + free ----
    ref_path = args.ref_model_path or args.model_path
    print(f"Loading frozen reference model from {ref_path} (DPO requires ref) ...")
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

    # Decide on pre-compute strategy from --precompute_ref:
    #   auto:  pre-compute when kl=0 (free ref); live forward when kl>0
    #   force: always pre-compute (errors if kl>0 because we'd still need ref live)
    #   off:   never pre-compute (legacy live-ref behavior every step)
    do_precompute = (
        (args.precompute_ref == "force") or
        (args.precompute_ref == "auto" and args.kl_weight == 0)
    )
    if args.precompute_ref == "force" and args.kl_weight > 0:
        raise ValueError(
            "--precompute_ref force is incompatible with --kl_weight > 0 "
            "(KL needs full per-step ref log_probs)."
        )

    if do_precompute:
        # Bigger batch is fine: ref forward only, no backward, no LoRA, no
        # gradient checkpointing recompute. 2× the train batch is safe on
        # any card that already fits training.
        pre_bs = args.per_device_batch_size * 2
        cache_dir = (
            Path(args.ref_cache_dir) if args.ref_cache_dir
            else Path(args.train_parquet).parent / "_ref_cache"
        )
        use_cache = not args.no_ref_cache
        ref_path_for_cache = args.ref_model_path or args.model_path
        print(f"\nRef scores (batch={pre_bs}, "
              f"{'cache: ' + str(cache_dir) if use_cache else 'no cache'}):")
        train_set.ref_scores = load_or_compute_ref_scores(
            ref_model, train_set, args.train_parquet, "train",
            n_groups_requested=args.max_train_groups,
            subsample_seed=12345,
            max_hist=args.max_hist, max_total_len=args.max_total_len,
            ref_model_path=ref_path_for_cache,
            batch_size=pre_bs, pad_token_id=tokenizer.pad_token_id,
            cache_dir=cache_dir, use_cache=use_cache,
        )
        valid_set.ref_scores = load_or_compute_ref_scores(
            ref_model, valid_set, args.valid_parquet, "valid",
            n_groups_requested=args.max_eval_groups,
            subsample_seed=67890,
            max_hist=args.max_hist, max_total_len=args.max_total_len,
            ref_model_path=ref_path_for_cache,
            batch_size=pre_bs, pad_token_id=tokenizer.pad_token_id,
            cache_dir=cache_dir, use_cache=use_cache,
        )
        if args.kl_weight == 0:
            print("Freeing ref_model (kl_weight=0; cached scores cover all "
                  "downstream needs).")
            del ref_model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            ref_model = None
        else:
            print("Keeping ref_model live (kl_weight>0 needs per-step "
                  "log_probs even with cached scalars).")

    trainer = ContrastiveTrainerDPOGRPO(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        dpo_beta=args.dpo_beta,
        sft_weight=args.sft_weight,
        kl_weight=args.kl_weight,
        sft_scale_mode=args.sft_scale_mode,
        group_norm_eps=args.group_norm_eps,
        group_norm_warmup=args.group_norm_warmup,
        ref_model=ref_model,
        G_per_group=args.G,
        sampler_seed=args.seed,
        callbacks=[TimingCallback()],
    )

    print("\n===== Training =====")
    sft_scale_str = (
        f"{args.sft_weight} × (1/std).mean × L_sft"
        if args.sft_scale_mode == "match_dpo"
        else f"{args.sft_weight} × L_sft  [raw]"
    )
    print(f"  loss = L_dpo_grpo (β={args.dpo_beta}, eps={args.group_norm_eps}, "
          f"warmup={args.group_norm_warmup}) + {sft_scale_str}"
          + (f" + {args.kl_weight} × KL" if args.kl_weight > 0 else ""))
    print(f"  best metric: eval_{args.best_metric} (greater is better)")
    trainer.train()

    print("\n===== Final eval =====")
    metrics = trainer.evaluate()

    def _safe_serialize(v):
        if isinstance(v, str):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    print(json.dumps({k: _safe_serialize(v) for k, v in metrics.items()},
                     indent=2))

    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nSaved LoRA adapter to {adapter_dir}")

    if args.merge_and_save:
        print("Merging LoRA into base weights ...")
        del trainer, model
        # Free ref_model too if it's still around (kl_weight>0 path).
        if ref_model is not None:
            del ref_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

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
