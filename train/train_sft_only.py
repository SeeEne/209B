"""
train_sft_only.py

Stage 1 of the sequential SFT → DPO ablation.

Pure SFT on chosen items only. NO rejected, NO ref model, NO group
normalization, NO KL.

    L_sft_raw   = -mean( log P_θ(chosen_sid | history) )
    loss        = SFT_LOSS_SCALE × L_sft_raw       # default scale = 16.0

ABLATION PARITY — why the 16× scale matters:
  In the 2026-05-04 joint DPO+SFT smoke, the SFT term entered total_loss as
      sft_term = sft_weight × inv_std_mean × L_sft_raw
               =    1.0     ×    ~16       × L_sft_raw   (match_dpo mode)
  with inv_std_mean observed at 14.82–16.57 across all logged steps (stable
  ≈16). So the SFT contribution to joint training was a 75-magnitude term,
  not 4.7. To attribute the SFT-only ablation result back to "what the SFT
  term in joint training was doing", we need the same effective magnitude.
  max_grad_norm=1.0 hides the difference whenever grad_norm > 1, but in
  warmup and late-training low-grad regimes, an unscaled SFT-only would get
  16× less effective LR than the SFT term in joint training, breaking the
  ablation. Default SFT_LOSS_SCALE=16.0 closes this gap.

Why this stage exists:
  In the joint DPO+SFT smoke (sft_weight=1.0 match_dpo), eval_pref_acc was
  only 0.546 (~random) even though chosen_recall went up. The hypothesis is
  that SFT (with match_dpo scaling) was getting ~87% of the gradient and
  swamping DPO's discriminative signal. By splitting into Stage 1 (SFT) and
  Stage 2 (DPO from SFT), each stage gets 100% of the gradient on a single
  objective.

  This file produces TWO useful artifacts:
    1. The Stage 1 checkpoint, which feeds train_dpo_from_sft.py as both
       --model_path AND --ref_model_path (standard SFT→DPO post-training).
    2. A clean SFT-only ablation arm — eval this checkpoint with
       evaluate_engaged.py to answer: "does SFT alone push chosen up AND
       rejected down, or does it lift both?"

Data: reads contrastive_dataset_v1_grpo (G=3, same as DPO trainer). The
3 cyclic-shift-1 pairs per group give 3 distinct chosens per user, which
is the SFT signal here. rejected_sid is in the parquet but ignored.

Usage (5k smoke):
    python train/train_sft_only.py \
        --model_path model/OneRec-1.7B \
        --template model/qwen3_soft_switch.jinja2 \
        --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
        --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
        --output_dir runs/sft_only_5k \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --num_checkpoints 5 --logging_steps 25 \
        --per_device_batch_size 24 \
        --lr 5e-5 \
        --merge_and_save
"""

import argparse
import json
import time
import warnings
from functools import partial
from pathlib import Path

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

from dataset import ChosenSIDDataset, chosen_collate
from utils import resolve_template


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


# Dataset (ChosenSIDDataset) and collate (chosen_collate) are imported from
# dataset.py — same v1_grpo prompt construction as the DPO/ORPO trainers,
# only the rejected column is ignored.


# ===========================================================================
# Score / loss
# ===========================================================================


def compute_chosen_score(last_hidden, lm_head, input_ids, prompt_lens,
                         num_sid_tokens=3):
    """
    Per-sequence mean log P(SID token | prefix) over the 3 SID tokens.
    Returns (B,) fp32 scores. Uses logits[target] - logsumexp(logits) so we
    skip materializing the full (B, K, V) softmax.
    """
    B, _, H = last_hidden.shape
    prompt_lens = prompt_lens.to(last_hidden.device, non_blocking=True)
    ks = torch.arange(num_sid_tokens, device=last_hidden.device)
    pred_pos = prompt_lens.unsqueeze(1) - 1 + ks            # (B, K)
    token_pos = prompt_lens.unsqueeze(1) + ks               # (B, K)

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)
    slice_logits = lm_head(slice_hidden).float()            # (B, K, V) fp32

    target_ids = input_ids.gather(1, token_pos)
    token_logits = slice_logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    log_norm = torch.logsumexp(slice_logits, dim=-1)
    token_logp = token_logits - log_norm                    # (B, K)
    return token_logp.mean(dim=1)                           # (B,)


# ===========================================================================
# Trainer
# ===========================================================================


class SFTOnlyTrainer(Trainer):
    """L = sft_loss_scale × -mean( chosen_score ).

    Default sft_loss_scale=16.0 reproduces the joint trainer's match_dpo SFT
    term (where sft_weight=1.0 and inv_std_mean stabilized at ~16). See
    module docstring for ablation-parity rationale.
    """

    def __init__(self, *args, sft_loss_scale=16.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sft_loss_scale = sft_loss_scale

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        prompt_lens = inputs.pop("prompt_lens")

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
        chosen_score = compute_chosen_score(
            out.last_hidden_state, lm_head,
            inputs["input_ids"], prompt_lens,
        )
        l_sft_raw = -chosen_score.mean()
        loss = self.sft_loss_scale * l_sft_raw

        # Re-attach so prediction_step (which calls compute_loss again with
        # the same dict) sees consistent keys.
        inputs["prompt_lens"] = prompt_lens

        if return_outputs:
            return loss, {"scores": chosen_score}

        if self.state.global_step % 100 == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"[mem] step={self.state.global_step} "
                  f"alloc={alloc:.2f}G peak={peak:.2f}G "
                  f"l_sft_raw={l_sft_raw.item():.3f} "
                  f"loss={loss.item():.3f}  "
                  f"(scale={self.sft_loss_scale}) "
                  f"chosen_score={chosen_score.mean().item():.3f}")
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
    """Only chosen_score is meaningful here (no rejected forward). Use the
    engagement-aware evaluator post-training for chosen_recall / Δrecall."""
    scores = eval_pred.predictions
    if isinstance(scores, tuple):
        scores = scores[0]
    return {"chosen_score": float(scores.mean())}


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--template", default=None)
    parser.add_argument("--train_parquet", required=True,
                        help="v1_grpo train.parquet (chosen column is used; "
                             "rejected is ignored by this trainer).")
    parser.add_argument("--valid_parquet", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--G", type=int, default=3,
        help="Pairs-per-group of the source v1_grpo dataset. Used ONLY for "
             "subsampling whole groups so step-counts stay aligned with the "
             "DPO trainer. Default 3.",
    )

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=24,
                        help="No 2× chosen+rejected stack here (chosen only), "
                             "so ~2× the DPO trainer's batch fits at the same "
                             "VRAM. Default 24.")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=-1)

    # Data
    parser.add_argument("--max_hist", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=3072)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_groups", type=int, default=-1,
                        help="Subsample to this many GROUPS (each = G rows). "
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
    parser.add_argument(
        "--num_checkpoints", type=int, default=5,
        help="If > 0 (default 5), schedule N evenly-spaced ckpts + evals "
             "across training. Overrides --eval_steps / --save_steps.",
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
        "--sft_loss_scale", type=float, default=16.0,
        help="Multiply L_sft = -chosen.mean() by this constant. Default "
             "16.0 reproduces the joint trainer's match_dpo SFT term (where "
             "inv_std_mean stabilized at 14.82–16.57 over the 2026-05-04 "
             "smoke; mean ≈ 16). Keep 16.0 for ablation parity with joint "
             "smoke. Setting 1.0 gives raw SFT loss — equivalent at large "
             "grad_norm (clip dominates) but diverges in warmup / late "
             "training where grad_norm < 1.",
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

    print(f"Loading datasets (chosen-only view of v1_grpo) ...")
    train_set = ChosenSIDDataset(
        args.train_parquet, tokenizer,
        G_per_group=args.G,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
    )
    valid_set = ChosenSIDDataset(
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

    train_groups = train_set.num_groups
    valid_groups = valid_set.num_groups

    # Auto-schedule N evenly-spaced checkpoints if requested.
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

    print(f"  train: {train_groups:,}/{train_full_groups:,} groups  "
          f"({len(train_set):,} chosen examples)")
    print(f"  valid: {valid_groups:,}/{valid_full_groups:,} groups  "
          f"({len(valid_set):,} chosen examples)")

    collate = partial(chosen_collate, pad_token_id=tokenizer.pad_token_id)

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
        metric_for_best_model="eval_chosen_score",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["none"],
        seed=args.seed,
        optim="adamw_torch_fused",
    )

    trainer = SFTOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        sft_loss_scale=args.sft_loss_scale,
        callbacks=[TimingCallback()],
    )

    print("\n===== Training =====")
    print(f"  loss = {args.sft_loss_scale} × -mean( log P(chosen_sid | history) )"
          f"   (no DPO, no ref)")
    print(f"  scale {args.sft_loss_scale} matches joint trainer's match_dpo "
          f"SFT term (inv_std_mean ≈ 16 in 2026-05-04 smoke)")
    print(f"  best metric: eval_chosen_score (greater is better)")
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