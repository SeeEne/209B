"""
train_contrastive.py

LoRA contrastive fine-tuning for OneRec-1.7B on contrastive_dataset_v0,
using HuggingFace Trainer + PEFT.

For each (history, chosen_item, rejected_item) example:

    score(item | history) = (1/3) * sum_t log P(item_token_t | history, item_<t)

    L = -log_softmax([score+/τ, score-/τ])[0]
      = log(1 + exp(-(score+ - score-)/τ))

Run on a single CUDA GPU.

Usage:
    python train/train_contrastive.py \
        --model_path /path/to/OneRec-1.7B \
        --template /path/to/qwen3_soft_switch.jinja2 \
        --train_parquet data/contrastive_dataset_v0/train.parquet \
        --valid_parquet data/contrastive_dataset_v0/valid.parquet \
        --output_dir runs/contrastive_v0_lora \
        --epochs 2 --per_device_batch_size 2 --grad_accum 8 --lr 2e-4 --temperature 0.1
"""

import argparse
import json
import time
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from dataset import ContrastiveDataset, contrastive_collate
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
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

# ---------------------------------------------------------------------------
# Score / loss
# ---------------------------------------------------------------------------


def compute_sid_logprobs(last_hidden, lm_head, input_ids, prompt_lens,
                         num_sid_tokens=3):
    """
    last_hidden : (2B, L, H)  — transformer output, BEFORE lm_head
    lm_head     : nn.Linear   — maps H -> V
    input_ids   : (2B, L)     — first B = chosen, last B = rejected
    prompt_lens : (B,)        — shared prompt length

    Returns (chosen_score, rejected_score, slice_log_probs):
        scores      : (B,) chosen + (B,) rejected — mean log P over K SID tokens
        slice_log_probs : (2B, K, V) full log-prob distribution at predictor
                         positions, kept for the KL regularizer.

    Memory note: applying lm_head to the full (2B, L, H) would materialize
    a (2B, L, V) tensor (~3 GB bf16 for Qwen3 V=152k at L=2600). We gather
    hidden states at the K predictor positions FIRST, then apply lm_head to
    just (2B, K, H) — turning a multi-GB allocation into a few MB.
    """
    bsz = input_ids.size(0)
    half = bsz // 2
    H = last_hidden.size(-1)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)              # (2B,)
    ks = torch.arange(num_sid_tokens, device=last_hidden.device)      # (K,)
    pred_pos = plens.unsqueeze(1) - 1 + ks                            # (2B, K)
    token_pos = plens.unsqueeze(1) + ks                               # (2B, K)

    # Gather hidden states at the K predictor positions: (2B, K, H).
    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)                         # (2B, K, H)
    # Apply lm_head ONLY to the K-row slice — output (2B, K, V), only ~MB.
    slice_logits = lm_head(slice_hidden)
    log_probs = F.log_softmax(slice_logits.float(), dim=-1)           # (2B, K, V)

    target_ids = input_ids.gather(1, token_pos)                       # (2B, K)
    token_logp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (2B, K)

    scores = token_logp.mean(dim=1)                                   # (2B,)
    return scores[:half], scores[half:], log_probs


def ref_log_probs_at_sid_positions(logits_full, prompt_lens, num_sid_tokens=3):
    """
    Reference path: take full (2B, L, V) logits from the frozen ref model,
    slice to (2B, K, V) at the K SID predictor positions, log_softmax in
    fp32. Used inside torch.no_grad(), so the (2B, L, V) allocation is
    transient.
    """
    V = logits_full.size(-1)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)
    ks = torch.arange(num_sid_tokens, device=logits_full.device)
    pred_pos = plens.unsqueeze(1) - 1 + ks                            # (2B, K)
    idx = pred_pos.unsqueeze(-1).expand(-1, -1, V)
    slice_logits = logits_full.gather(1, idx)                         # (2B, K, V)
    return F.log_softmax(slice_logits.float(), dim=-1)                # (2B, K, V)


def contrastive_loss(chosen_score, rejected_score, temperature):
    margin = (chosen_score - rejected_score) / temperature
    return F.softplus(-margin).mean()


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------


class ContrastiveTrainer(Trainer):
    """
    Overrides compute_loss to apply our pairwise contrastive objective.
    Overrides prediction_step so eval reports preference accuracy.
    """

    def __init__(self, *args, temperature=0.1, ref_model=None, kl_weight=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.ref_model = ref_model
        self.kl_weight = kl_weight

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        prompt_lens = inputs.pop("prompt_lens")

        # Skip the lm_head on the full sequence: it would produce a
        # (2B, L, V) ~3 GB bf16 tensor (plus its gradient in backward).
        # Walk the wrapping to grab the inner Qwen3Model + lm_head separately,
        # then apply lm_head only to the K positions we score.
        # PEFT wrapping: PeftModel -> LoraModel -> Qwen3ForCausalLM
        causal_lm = (
            model.get_base_model() if hasattr(model, "get_base_model") else model
        )
        transformer = causal_lm.model     # Qwen3Model (LoRA layers injected)
        lm_head = causal_lm.lm_head       # Linear, no LoRA target

        transformer_out = transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )
        last_hidden = transformer_out.last_hidden_state   # (2B, L, H) bf16

        chosen_s, rejected_s, trained_log_probs = compute_sid_logprobs(
            last_hidden,
            lm_head,
            inputs["input_ids"],
            prompt_lens,
        )

        # Pair-wise contrastive: keep chosen_score > rejected_score.
        l_contrast = contrastive_loss(chosen_s, rejected_s, self.temperature)
        loss = l_contrast

        # KL regularization against frozen reference model — anchors the full
        # token distribution at SID positions to ref's, preventing the
        # mode-collapse failure mode (all users → same popular SIDs) that
        # pure contrastive admits.
        l_kl = None
        if self.ref_model is not None and self.kl_weight > 0:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )
                ref_log_probs = ref_log_probs_at_sid_positions(
                    ref_out.logits, prompt_lens
                )
            # Reverse KL: KL(trained || ref). Trained is encouraged to keep
            # mass where ref does; mode collapse → high KL.
            trained_probs = trained_log_probs.exp()
            l_kl = (trained_probs * (trained_log_probs - ref_log_probs)) \
                .sum(dim=-1).mean()
            loss = loss + self.kl_weight * l_kl

        # Re-attach so prediction_step / metrics can see prompt_lens if needed.
        inputs["prompt_lens"] = prompt_lens

        if return_outputs:
            scores = torch.stack([chosen_s, rejected_s], dim=-1)  # (B, 2)
            return loss, {"scores": scores}

        if self.state.global_step % 100 == 0:
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            extra = f" l_kl={l_kl.item():.3f}" if l_kl is not None else ""
            print(f"[mem] step={self.state.global_step} "
                  f"alloc={alloc:.2f}G reserved={reserved:.2f}G peak={peak:.2f}G "
                  f"l_contrast={l_contrast.item():.3f}{extra}")

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, extras = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        # Pretend scores are "predictions"; compute_metrics turns them into pref_acc.
        # Provide dummy labels so HF eval loop is happy.
        labels = torch.zeros(
            extras["scores"].size(0), dtype=torch.long, device=extras["scores"].device
        )
        return (loss.detach(), extras["scores"].detach(), labels)


def compute_metrics(eval_pred):
    """eval_pred.predictions: (N, 2) of [chosen, rejected] scores."""
    scores = eval_pred.predictions
    if isinstance(scores, tuple):
        scores = scores[0]
    pref_acc = float((scores[:, 0] > scores[:, 1]).mean())
    margin = float((scores[:, 0] - scores[:, 1]).mean())
    return {"pref_acc": pref_acc, "margin": margin}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="OpenOneRec/OneRec-1.7B",
        help="HF repo id or local checkpoint dir. "
        "Default pulls from the Hub on first run.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Optional path to qwen3_soft_switch.jinja2. "
        "If omitted, looks under <project>/oneRec/, "
        "then ~/.cache/onerec_template/, then downloads "
        "from the OpenOneRec GitHub repo.",
    )
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--valid_parquet", required=True)
    parser.add_argument("--output_dir", required=True)

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="LoRA typically needs higher LR than full FT (~1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.1,
        help="Weight on KL(trained || ref) regularization at SID positions. "
             "0 disables (pure contrastive — risk of mode collapse). "
             "0.05–0.5 is the typical useful range.",
    )
    parser.add_argument(
        "--ref_model_path",
        default=None,
        help="Path to frozen reference model for KL regularization. "
             "Defaults to --model_path (the original base, before any "
             "training). Ignored if --kl_weight 0.",
    )
    parser.add_argument("--max_steps", type=int, default=-1)

    # Data
    parser.add_argument("--max_hist", type=int, default=512,
                        help="History items per example. OneRec's release is "
                             "already capped at 512, so 512 = full history "
                             "(matches the eval-time prompt distribution).")
    parser.add_argument("--max_total_len", type=int, default=3072,
                        help="Token-length safety cap. ~2600 tokens needed "
                             "for max_hist=512; leave headroom.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=-1,
        help="Subsample train.parquet to this many pairs (deterministic, "
             "seeded). -1 = use the full set. Useful for smoke tests, e.g. "
             "5000 finishes one short epoch in ~30min on a 5090.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=2000,
        help="Subsample valid.parquet to this many pairs (deterministic, "
             "seeded). -1 = use full valid set. 2000 gives ±1%% std error "
             "on pref_acc and is ~7x faster than the full 13.9k.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument(
        "--merge_and_save",
        action="store_true",
        help="After training, merge LoRA into base weights and "
        "save a full model under <output_dir>/merged. "
        "Lets evaluate_origin.py load the checkpoint with "
        "no extra flags.",
    )

    # Logging
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=1500)
    parser.add_argument("--save_steps", type=int, default=1500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    assert torch.cuda.is_available(), "CUDA not available"

    # ---- Tokenizer / model ----
    print(f"Loading tokenizer + base model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    template_path = resolve_template(args.template)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # Pick the fastest attention impl available. SDPA on PyTorch 2.x already
    # dispatches to FlashAttention-2 kernels for bf16 on Ampere+ — explicit
    # "flash_attention_2" only helps if flash-attn is installed natively
    # (often a pain on Windows).
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
        model.config.use_cache = False  # required with grad checkpointing

    # ---- Apply LoRA ----
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
    print("┌─ LoRA config ─────────────────────────────────")
    print(f"│  rank (r)        : {args.lora_r}")
    print(
        f"│  alpha           : {args.lora_alpha}"
        f"     (scaling = alpha/r = {args.lora_alpha / args.lora_r:.2f})"
    )
    print(f"│  dropout         : {args.lora_dropout}")
    print(f"│  target modules  : {args.lora_target_modules}")
    print("├─ Parameter counts ────────────────────────────")
    print(f"│  trainable       : {n_trainable:>13,}")
    print(f"│  total           : {n_total:>13,}")
    print(f"│  trainable %     : {100 * n_trainable / n_total:>13.4f} %")
    print("└───────────────────────────────────────────────")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed when grad checkpoint + frozen base

    # ---- Data ----
    print("Loading datasets ...")
    train_set = ContrastiveDataset(
        args.train_parquet,
        tokenizer,
        max_hist=args.max_hist,
        max_total_len=args.max_total_len,
    )
    valid_set = ContrastiveDataset(
        args.valid_parquet,
        tokenizer,
        max_hist=args.max_hist,
        max_total_len=args.max_total_len,
    )

    # Optional deterministic subsampling. We seed independently of args.seed
    # so the same subset is selected regardless of training seed — this lets
    # numbers from a smoke run and a full run be compared on the same valid
    # subset.
    def _subsample(ds, n, sampling_seed):
        if n is None or n <= 0 or n >= len(ds):
            return ds
        rng = torch.Generator().manual_seed(sampling_seed)
        indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
        return torch.utils.data.Subset(ds, indices)

    train_full = len(train_set)
    valid_full = len(valid_set)
    train_set = _subsample(train_set, args.max_train_samples, sampling_seed=12345)
    valid_set = _subsample(valid_set, args.max_eval_samples, sampling_seed=67890)

    def _fmt(now, full):
        return f"{now:,}" + (f" / {full:,}" if now < full else "")

    print(f"  train: {_fmt(len(train_set), train_full)} pairs  |  "
          f"valid: {_fmt(len(valid_set), valid_full)} pairs")

    collate = partial(contrastive_collate, pad_token_id=tokenizer.pad_token_id)

    # ---- TrainingArguments ----
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
        metric_for_best_model="eval_pref_acc",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=["none"],
        seed=args.seed,
        optim="adamw_torch_fused",
    )

    # ---- Reference model for KL regularization ----
    ref_model = None
    if args.kl_weight > 0:
        ref_path = args.ref_model_path or args.model_path
        print(f"Loading frozen reference model from {ref_path} "
              f"(kl_weight={args.kl_weight}) ...")
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

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        temperature=args.temperature,
        ref_model=ref_model,
        kl_weight=args.kl_weight,
        callbacks=[TimingCallback()],
    )

    # ---- Train ----
    print("\n===== Training =====")
    trainer.train()

    # ---- Final eval ----
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

    # ---- Save adapter ----
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nSaved LoRA adapter to {adapter_dir}")

    # ---- Optionally merge for downstream eval ----
    if args.merge_and_save:
        print("Merging LoRA into base weights ...")
        # Free the trained model first so we don't hold two copies in memory
        # while building the merged checkpoint.
        del trainer, model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Reload base in the desired dtype, attach adapter, merge.
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map={"": "cuda:0"},
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()
        merged_dir = out_dir / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Saved merged checkpoint to {merged_dir}")
        print(
            f"evaluate_origin.py can load this directly with --model_path {merged_dir}"
        )


if __name__ == "__main__":
    main()
