"""
train_orpo.py

Single-stage ORPO trainer (Hong et al. 2024, "ORPO: Monolithic Preference
Optimization without Reference Model").

    log_odds_θ(y|x) = log P_θ(y|x) − log(1 − P_θ(y|x))
    ratio           = log_odds_θ(c|x) − log_odds_θ(r|x)
    L_NLL           = −mean( log P_θ(c|x) )
    L_OR            = −mean( log σ(ratio) )
    L_ORPO          = L_NLL + λ × L_OR

Where P_θ(y|x) is interpreted as the geometric mean per token over the 3 SID
tokens, i.e. log P_θ(y|x) = (1/3) Σ log p_θ(y_t | x, y_<t). This matches the
chosen_score / rejected_score convention used everywhere else in this repo
(SFT trainer, DPO trainer, evaluate_engaged.py).

Why ORPO is interesting for this project:
  - No reference model → no ref-cache machinery → ~50% less peak VRAM than
    DPO+SFT joint, and no precompute pass at startup.
  - Single stage from base → directly comparable to "SFT-only" arm at the
    same data scale (no SFT init confound).
  - The L_OR term is bounded and only sharpens *near* P(c) ≈ P(r); when
    chosen is already well above rejected, gradient flows mostly through
    L_NLL — naturally preventing the GRPO-only failure mode where margin
    grows by lowering both sides.

Design choices specific to this project (vs the paper):
  - We score the 3 SID tokens only (not the full chat-template envelope),
    consistent with all other trainers/evaluators here.
  - L_NLL is the same mean-log-p we use in the SFT trainer. For ABLATION
    PARITY with the SFT-only run, we expose --nll_loss_scale (default 1.0
    here, since ORPO papers fold the relative weighting into λ; the SFT
    trainer's 16× scale was a different ablation goal).
  - Group structure is preserved in the *dataset subsampling* (--G 3 keeps
    us reading the same 5000 users as the SFT and DPO trainers), but is
    NOT used in the loss — vanilla ORPO has no group normalization.
  - Default λ = 0.1 (paper / OPUS implementation).

Numerical stability of log_odds:
  P close to 1 → log(1 − P) → −∞. We use log1mexp(log_p) with the standard
  branched implementation (paper "Accurately Computing log(1 − exp(−|a|))",
  Mächler 2012) so the gradient stays finite even if a chosen probability
  saturates near 1. In practice on this task log_p ≈ −4.84 nats, far from
  saturation, but the guard is essentially free.

Usage (5k smoke):
    python train/train_orpo.py \\
        --model_path model/OneRec-1.7B \\
        --template model/qwen3_soft_switch.jinja2 \\
        --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \\
        --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \\
        --output_dir runs/orpo_5k \\
        --max_train_groups 5000 --max_eval_groups 1000 \\
        --num_checkpoints 5 --logging_steps 25 \\
        --per_device_batch_size 24 \\
        --lr 5e-5 \\
        --lambda_or 0.1 \\
        --merge_and_save
"""

import argparse
import json
import math
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

from dataset import PairedSIDDataset, paired_collate
from utils import resolve_template


class TimingCallback(TrainerCallback):
    """Adds wall-clock and per-step time to each Trainer log line."""

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
# Score / loss
# ===========================================================================


def compute_sid_logprobs(last_hidden, lm_head, input_ids, prompt_lens,
                         num_sid_tokens=3):
    """Returns (chosen_score, rejected_score) given (2B, L) packed input.
    Each score is the per-sequence MEAN log-prob over the 3 SID tokens."""
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


_LOG_HALF = math.log(0.5)  # ≈ -0.6931


def log1mexp(x):
    """Numerically stable log(1 - exp(x)) for x < 0.

    Mächler (2012) "Accurately Computing log(1 − exp(−|a|))".
    Two regimes:
      x > log(0.5)  →  log(-expm1(x))      (1 - exp(x) is small → use expm1)
      x ≤ log(0.5)  →  log1p(-exp(x))      (1 - exp(x) is close to 1)

    Caller MUST clamp x strictly < 0 (we clamp to ≤ -EPS just below).
    """
    return torch.where(
        x > _LOG_HALF,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def compute_orpo_loss(chosen_score, rejected_score, lambda_or,
                      nll_loss_scale=1.0, log_clamp=-1e-6):
    """
    chosen_score, rejected_score: (B,) mean log-prob per sequence.
    Returns (loss, components_dict).

    log_clamp guards log1mexp against log P → 0 (would yield -inf log_odds).
    -1e-6 ≈ P up to 0.999999 → log_odds up to 13.8 nats; safe and tight.
    """
    # NLL = -mean log P(chosen | x)
    l_nll_raw = -chosen_score.mean()
    l_nll = nll_loss_scale * l_nll_raw

    # Clamp to strictly negative for log1mexp.
    c_clamped = torch.clamp(chosen_score, max=log_clamp)
    r_clamped = torch.clamp(rejected_score, max=log_clamp)

    log1m_c = log1mexp(c_clamped)
    log1m_r = log1mexp(r_clamped)
    log_odds_c = chosen_score - log1m_c
    log_odds_r = rejected_score - log1m_r
    ratio = log_odds_c - log_odds_r

    # L_OR = -mean log σ(ratio)
    l_or = -F.logsigmoid(ratio).mean()

    loss = l_nll + lambda_or * l_or

    return loss, {
        "l_nll_raw": l_nll_raw.detach(),
        "l_nll_scaled": l_nll.detach(),
        "l_or": l_or.detach(),
        "log_odds_margin": ratio.mean().detach(),
        "log_odds_c": log_odds_c.mean().detach(),
        "log_odds_r": log_odds_r.mean().detach(),
    }


# ===========================================================================
# Trainer
# ===========================================================================


class ORPOTrainer(Trainer):
    """Single-stage ORPO. No ref model, no group norm, no KL.

    L = nll_loss_scale × (-mean log P(c)) + lambda_or × (-mean log σ(log_odds(c) - log_odds(r)))
    """

    def __init__(self, *args, lambda_or=0.1, nll_loss_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_or = lambda_or
        self.nll_loss_scale = nll_loss_scale

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        prompt_lens = inputs.pop("prompt_lens")
        # PairedSIDDataset always emits group_id and the shared paired_collate
        # forwards group_ids; ORPO doesn't consume groups but pop here so
        # nothing extra travels into the transformer call.
        group_ids = inputs.pop("group_ids", None)

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
        chosen_score, rejected_score = compute_sid_logprobs(
            out.last_hidden_state, lm_head,
            inputs["input_ids"], prompt_lens,
        )
        loss, components = compute_orpo_loss(
            chosen_score, rejected_score,
            lambda_or=self.lambda_or,
            nll_loss_scale=self.nll_loss_scale,
        )

        # Re-attach for prediction_step.
        inputs["prompt_lens"] = prompt_lens
        if group_ids is not None:
            inputs["group_ids"] = group_ids

        if return_outputs:
            scores = torch.stack([chosen_score, rejected_score], dim=-1)
            return loss, {"scores": scores}

        if self.state.global_step % 100 == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[mem] step={self.state.global_step} "
                  f"alloc={alloc:.2f}G peak={peak:.2f}G "
                  f"loss={loss.item():.3f} "
                  f"l_nll={components['l_nll_scaled'].item():.3f} "
                  f"l_or={components['l_or'].item():.3f} "
                  f"log_odds_margin={components['log_odds_margin'].item():.3f} "
                  f"chosen={chosen_score.mean().item():.3f} "
                  f"rejected={rejected_score.mean().item():.3f}")
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


_LOG_HALF_NP = math.log(0.5)


def _log1mexp_np(x: np.ndarray, log_clamp: float = -1e-6) -> np.ndarray:
    """Numpy port of log1mexp for compute_metrics. Same Mächler 2012 split as
    the torch version above; ``log_clamp`` keeps log_odds finite at P→1."""
    x = np.minimum(x, log_clamp)
    return np.where(
        x > _LOG_HALF_NP,
        np.log(-np.expm1(x)),
        np.log1p(-np.exp(x)),
    )


def compute_metrics(eval_pred):
    """Returns chosen / rejected absolute log-prob, the linear margin,
    pref_acc, AND ``log_odds_margin`` (= what L_OR actually optimizes).

    log_odds_margin is the ORPO-specific signal: pref_acc and linear margin
    can climb while log_odds_margin saturates, because log_odds compresses
    near P→1. Worth tracking separately when interpreting OR-term behavior.
    """
    scores = eval_pred.predictions
    if isinstance(scores, tuple):
        scores = scores[0]
    chosen = scores[:, 0].astype(np.float64)
    rejected = scores[:, 1].astype(np.float64)

    log_odds_c = chosen - _log1mexp_np(chosen)
    log_odds_r = rejected - _log1mexp_np(rejected)

    return {
        "chosen_score": float(chosen.mean()),
        "rejected_score": float(rejected.mean()),
        "margin": float((chosen - rejected).mean()),
        "pref_acc": float((chosen > rejected).mean()),
        "log_odds_margin": float((log_odds_c - log_odds_r).mean()),
    }


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B",
                        help="Base model (ORPO is single-stage from base; "
                             "do NOT pass an SFT checkpoint here).")
    parser.add_argument("--template", default=None)
    parser.add_argument("--train_parquet", required=True,
                        help="v1_grpo train.parquet (chosen + rejected used).")
    parser.add_argument("--valid_parquet", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--G", type=int, default=3,
        help="Pairs-per-group of v1_grpo. Used ONLY for whole-group "
             "subsampling so step-counts/users align with SFT and DPO arms. "
             "The ORPO loss does NOT use group structure. Default 3.",
    )

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=24,
                        help="Pairs per micro-batch (each pair = 2 sequences "
                             "stacked → 48 forward sequences). A100-80GB at "
                             "seq~640 fits 24 comfortably; 32 plausible.")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Same as SFT/DPO arms for clean ablation.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=-1)

    # ORPO
    parser.add_argument(
        "--lambda_or", type=float, default=0.3,
        help="Weight on the odds-ratio term. Paper default is 0.1 under the "
             "*sum*-log-p convention (log P(y|x) = sum over tokens). This repo "
             "scores SIDs with the *mean*-log-p convention (chosen_score = "
             "(1/3) * sum log p_t) for consistency with all other trainers, "
             "which scales the OR-term gradient by 1/3 vs the paper. Default "
             "0.3 here ≈ paper's 0.1 in equivalent gradient strength. Drop to "
             "0.1 for a literal-paper recipe; raise to 1.0 for stronger "
             "discriminative pressure.",
    )
    parser.add_argument(
        "--nll_loss_scale", type=float, default=1.0,
        help="Multiplier on the NLL term (= the SFT-style loss on chosen). "
             "Default 1.0 keeps the paper's formulation. Setting this to "
             "16.0 would match the SFT-only ablation's match_dpo scaling — "
             "do that ONLY if you also rescale lambda_or to keep the OR/NLL "
             "ratio you want.",
    )

    # Data
    parser.add_argument("--max_hist", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=3072)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_groups", type=int, default=-1,
                        help="Subsample to this many groups (G rows each). "
                             "-1 = full set. Smoke: 5000.")
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
        help="Default chosen_score — same protection rationale as DPO arm: "
             "pref_acc / margin can grow even if chosen log-prob drops "
             "(both sides go down, rejected faster). chosen_score directly "
             "correlates with recall_chosen on the engagement valid.",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    # PairedSIDDataset is shared across DPO joint / DPO-from-SFT / ORPO. ORPO
    # ignores the forwarded group_id (paired_collate emits it; ORPOTrainer pops
    # and discards) — kept as one impl so a fix lands once for every trainer.
    train_set = PairedSIDDataset(args.train_parquet, tokenizer,
                                 G_per_group=args.G,
                                 max_hist=args.max_hist,
                                 max_total_len=args.max_total_len)
    valid_set = PairedSIDDataset(args.valid_parquet, tokenizer,
                                 G_per_group=args.G,
                                 max_hist=args.max_hist,
                                 max_total_len=args.max_total_len)
    train_full = train_set.num_groups
    valid_full = valid_set.num_groups
    if args.max_train_groups > 0:
        train_set = train_set.subsample_groups(args.max_train_groups,
                                               seed=12345)
    if args.max_eval_groups > 0:
        valid_set = valid_set.subsample_groups(args.max_eval_groups,
                                               seed=67890)

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

    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        lambda_or=args.lambda_or,
        nll_loss_scale=args.nll_loss_scale,
        callbacks=[TimingCallback()],
    )

    print("\n===== Training =====")
    print(f"  loss = {args.nll_loss_scale} × L_NLL(chosen)  "
          f"+ {args.lambda_or} × L_OR(chosen, rejected)")
    print(f"  L_OR = -log σ(log_odds(c) - log_odds(r))   [no ref, no groups]")
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
