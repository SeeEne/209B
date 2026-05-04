"""
train_contrastive_length3.py

Length=3 variant of train_contrastive.py. Each (history, chosen, rejected)
pair has 3 chosen items and 3 rejected items.

Score: mean log P over 9 SID tokens (3 items × 3 tokens each).
This is "equal weight per item" by construction (each item contributes
the same number of tokens, so mean over tokens == mean over items).

Layout per row after the prompt (which ends with <|sid_begin|>):
    pos 0..2  : item 1 SID tokens (<s_a><s_b><s_c>)
    pos 3     : <|sid_end|>
    pos 4     : <|sid_begin|>
    pos 5..7  : item 2 SID tokens
    pos 8     : <|sid_end|>
    pos 9     : <|sid_begin|>
    pos 10..12: item 3 SID tokens

Run on a single CUDA GPU.

Usage:
    python train/train_contrastive_length3.py \
        --model_path model/OneRec-1.7B \
        --template model/qwen3_soft_switch.jinja2 \
        --train_parquet data/contrastive_dataset_v1/train.parquet \
        --valid_parquet data/contrastive_dataset_v1/valid.parquet \
        --output_dir runs/contrastive_v1_smoke \
        --max_train_samples 5000 --max_eval_samples 1000 \
        --eval_steps 50 --save_steps 50 --logging_steps 10 \
        --per_device_batch_size 16 --grad_accum 1 \
        --lr 5e-5 --temperature 0.5 --kl_weight 0.5
"""

import argparse
import json
import time
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from dataset import (
    SYSTEM_PROMPT,
    build_history_text,
    contrastive_collate,
    sid_to_core_text,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
from utils import resolve_template

# 3 items × 3 SID tokens + 2 separators × 2 tokens between items = 13
NUM_ITEMS = 3
TOKENS_PER_ITEM = 3
ANSWER_TOKEN_LEN = NUM_ITEMS * TOKENS_PER_ITEM + (NUM_ITEMS - 1) * 2  # 13

# SID token offsets relative to prompt_len, for the 9 scored tokens.
SID_OFFSETS = [0, 1, 2, 5, 6, 7, 10, 11, 12]


# ---------------------------------------------------------------------------
# Callback: per-step timing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_answer_text(items):
    """3 items joined by <|sid_end|><|sid_begin|>; no trailing <|sid_end|>."""
    parts = []
    for i, sid in enumerate(items):
        if i > 0:
            parts.append("<|sid_end|><|sid_begin|>")
        parts.append(sid_to_core_text(sid))
    return "".join(parts)


class ContrastiveDatasetLength3(Dataset):
    """Length-3 variant: each row carries 3 chosen + 3 rejected items."""

    def __init__(self, parquet_path, tokenizer, max_hist=512, max_total_len=3072):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_hist = max_hist
        self.max_total_len = max_total_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hist_sids = row["hist_sids"]
        chosen_sids = row["chosen_sids"]
        rejected_sids = row["rejected_sids"]

        assert len(chosen_sids) == NUM_ITEMS, (
            f"need {NUM_ITEMS} chosen items, got {len(chosen_sids)} "
            f"(did you build the dataset with --length 3?)"
        )
        assert len(rejected_sids) == NUM_ITEMS, (
            f"need {NUM_ITEMS} rejected items, got {len(rejected_sids)}"
        )

        hist_text = build_history_text(hist_sids, max_hist=self.max_hist)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": hist_text},
        ]
        prompt_with_begin = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<|sid_begin|>"

        chosen_text = prompt_with_begin + build_answer_text(chosen_sids)
        rejected_text = prompt_with_begin + build_answer_text(rejected_sids)

        prompt_ids = self.tokenizer(prompt_with_begin, add_special_tokens=True)["input_ids"]
        chosen_ids = self.tokenizer(chosen_text, add_special_tokens=True)["input_ids"]
        rejected_ids = self.tokenizer(rejected_text, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        # Each special token (<s_a_*>, <s_b_*>, <s_c_*>, <|sid_begin|>,
        # <|sid_end|>) tokenizes to exactly 1 token. Layout = prompt + 13.
        assert len(chosen_ids) == prompt_len + ANSWER_TOKEN_LEN, (
            f"chosen tokens != prompt+{ANSWER_TOKEN_LEN}: "
            f"{len(chosen_ids)} vs {prompt_len + ANSWER_TOKEN_LEN}"
        )
        assert len(rejected_ids) == prompt_len + ANSWER_TOKEN_LEN, (
            f"rejected tokens != prompt+{ANSWER_TOKEN_LEN}: "
            f"{len(rejected_ids)} vs {prompt_len + ANSWER_TOKEN_LEN}"
        )

        # Truncate from the LEFT of the prompt if too long (keep the answer tail).
        if self.max_total_len is not None and len(chosen_ids) > self.max_total_len:
            cut = len(chosen_ids) - self.max_total_len
            chosen_ids = chosen_ids[cut:]
            rejected_ids = rejected_ids[cut:]
            prompt_len = prompt_len - cut

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt_len": prompt_len,
        }


# ---------------------------------------------------------------------------
# Score / loss
# ---------------------------------------------------------------------------


def compute_sid_logprobs_length3(last_hidden, lm_head, input_ids, prompt_lens):
    """
    Score 9 SID tokens (3 items × 3 tokens) per row, equal-weighted.

    Returns (chosen_score, rejected_score, slice_log_probs):
        scores      : (B,) chosen + (B,) rejected — mean log P over 9 tokens
        slice_log_probs : (2B, 9, V) for the KL regularizer
    """
    bsz = input_ids.size(0)
    half = bsz // 2
    H = last_hidden.size(-1)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)             # (2B,)
    token_offsets = torch.tensor(SID_OFFSETS, device=last_hidden.device)  # (9,)
    pred_offsets = token_offsets - 1

    pred_pos = plens.unsqueeze(1) + pred_offsets                     # (2B, 9)
    token_pos = plens.unsqueeze(1) + token_offsets                   # (2B, 9)

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)                        # (2B, 9, H)
    slice_logits = lm_head(slice_hidden)                             # (2B, 9, V)
    log_probs = F.log_softmax(slice_logits.float(), dim=-1)          # (2B, 9, V)

    target_ids = input_ids.gather(1, token_pos)                      # (2B, 9)
    token_logp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (2B, 9)

    # Equal weight per token == equal weight per item (3 tokens each).
    scores = token_logp.mean(dim=1)                                  # (2B,)
    return scores[:half], scores[half:], log_probs


def ref_log_probs_at_sid_positions_length3(logits_full, prompt_lens):
    """Same gather pattern, on the full (2B, L, V) logits from the ref model."""
    V = logits_full.size(-1)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)
    token_offsets = torch.tensor(SID_OFFSETS, device=logits_full.device)
    pred_offsets = token_offsets - 1
    pred_pos = plens.unsqueeze(1) + pred_offsets

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, V)
    slice_logits = logits_full.gather(1, idx)
    return F.log_softmax(slice_logits.float(), dim=-1)


def contrastive_loss(chosen_score, rejected_score, temperature):
    margin = (chosen_score - rejected_score) / temperature
    return F.softplus(-margin).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, temperature=0.1, ref_model=None, kl_weight=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.ref_model = ref_model
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        prompt_lens = inputs.pop("prompt_lens")

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

        chosen_s, rejected_s, trained_log_probs = compute_sid_logprobs_length3(
            last_hidden, lm_head, inputs["input_ids"], prompt_lens
        )

        l_contrast = contrastive_loss(chosen_s, rejected_s, self.temperature)
        loss = l_contrast

        l_kl = None
        if self.ref_model is not None and self.kl_weight > 0:
            with torch.no_grad():
                ref_out = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    use_cache=False,
                )
                ref_log_probs = ref_log_probs_at_sid_positions_length3(
                    ref_out.logits, prompt_lens
                )
            trained_probs = trained_log_probs.exp()
            l_kl = (trained_probs * (trained_log_probs - ref_log_probs)) \
                .sum(dim=-1).mean()
            loss = loss + self.kl_weight * l_kl

        inputs["prompt_lens"] = prompt_lens

        if return_outputs:
            scores = torch.stack([chosen_s, rejected_s], dim=-1)
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
    return {"pref_acc": pref_acc, "margin": margin}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenOneRec/OneRec-1.7B")
    parser.add_argument("--template", default=None)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--valid_parquet", required=True)
    parser.add_argument("--output_dir", required=True)

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--kl_weight", type=float, default=0.5)
    parser.add_argument("--ref_model_path", default=None)
    parser.add_argument("--max_steps", type=int, default=-1)

    # Data
    parser.add_argument("--max_hist", type=int, default=512)
    parser.add_argument("--max_total_len", type=int, default=3072)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=2000)

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

    print(f"Loading tokenizer + base model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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

    print("Loading datasets ...")
    train_set = ContrastiveDatasetLength3(
        args.train_parquet, tokenizer,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
    )
    valid_set = ContrastiveDatasetLength3(
        args.valid_parquet, tokenizer,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
    )

    def _subsample(ds, n, seed):
        if n is None or n <= 0 or n >= len(ds):
            return ds
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(ds), generator=rng)[:n].tolist()
        return torch.utils.data.Subset(ds, indices)

    train_full = len(train_set)
    valid_full = len(valid_set)
    train_set = _subsample(train_set, args.max_train_samples, 12345)
    valid_set = _subsample(valid_set, args.max_eval_samples, 67890)

    def _fmt(now, full):
        return f"{now:,}" + (f" / {full:,}" if now < full else "")

    print(f"  train: {_fmt(len(train_set), train_full)} pairs  |  "
          f"valid: {_fmt(len(valid_set), valid_full)} pairs")

    collate = partial(contrastive_collate, pad_token_id=tokenizer.pad_token_id)

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

    print("\n===== Training =====")
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
