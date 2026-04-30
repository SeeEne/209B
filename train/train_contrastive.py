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
    TrainingArguments,
)
from transformers.utils import logging as hf_logging
from utils import resolve_template

# ---------------------------------------------------------------------------
# Score / loss
# ---------------------------------------------------------------------------


def compute_sid_logprobs(logits, input_ids, prompt_lens, num_sid_tokens=3):
    """
    logits   : (2B, L, V)
    input_ids: (2B, L)  — first B = chosen, last B = rejected (same prompt prefix)
    prompt_lens: (B,)   — shared prompt length

    Returns (chosen_score, rejected_score), each shape (B,).
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    bsz = input_ids.size(0)
    half = bsz // 2
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)  # (2B,)

    scores = torch.zeros(bsz, device=logits.device, dtype=torch.float32)
    for k in range(num_sid_tokens):
        pred_pos = plens - 1 + k  # logits index
        token_pos = plens + k  # token index
        idx = pred_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, log_probs.size(-1))
        gathered = log_probs.gather(1, idx).squeeze(1)  # (2B, V)
        target = input_ids.gather(1, token_pos.unsqueeze(1)).squeeze(1)
        scores = scores + gathered.gather(1, target.unsqueeze(1)).squeeze(1)

    scores = scores / num_sid_tokens
    return scores[:half], scores[half:]


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

    def __init__(self, *args, temperature=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        prompt_lens = inputs.pop("prompt_lens")
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        chosen_s, rejected_s = compute_sid_logprobs(
            out.logits,
            inputs["input_ids"],
            prompt_lens,
        )
        loss = contrastive_loss(chosen_s, rejected_s, self.temperature)
        # Re-attach so prediction_step / metrics can see prompt_lens if needed.
        inputs["prompt_lens"] = prompt_lens

        if return_outputs:
            scores = torch.stack([chosen_s, rejected_s], dim=-1)  # (B, 2)
            return loss, {"scores": scores}
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
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="LoRA typically needs higher LR than full FT (~1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
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
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": "cuda:0"},
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
    print(f"  train: {len(train_set):,} pairs  |  valid: {len(valid_set):,} pairs")

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
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        temperature=args.temperature,
    )

    # ---- Train ----
    print("\n===== Training =====")
    trainer.train()

    # ---- Final eval ----
    print("\n===== Final eval =====")
    metrics = trainer.evaluate()
    print(json.dumps({k: float(v) for k, v in metrics.items()}, indent=2))

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
