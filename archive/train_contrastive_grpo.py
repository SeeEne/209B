"""
train_contrastive_grpo.py

GRPO-style contrastive fine-tuning on contrastive_dataset_v1_grpo.

Each user contributes G=3 single-item (chosen, rejected) pairs sharing a
group_id. Within each group, per-pair contrastive losses are normalized by
the group's std (detached) — this attacks the per-prompt difficulty
variance that pure contrastive can't reduce by adding more data.

Per-pair loss (length=1 contrastive on 3 SID tokens):
    margin_g = (chosen_score_g - rejected_score_g) / temperature
    L_g      = softplus(-margin_g)
GRPO normalization (within each group of G pairs):
    L_grpo   = mean over groups of mean over g of L_g / (std(L_g..L_G) + eps)
    The std is detached (treated as a baseline scalar), so gradients still
    flow through L_g, just scaled per group.
Plus standard KL anchor against frozen reference (same as length3).

Critical batching constraint: every micro-batch must contain whole groups
(all G rows of a group together) so std-normalization has the right
denominator. Enforced by GroupedSampler + per_device_batch_size % G == 0.

Run on a single CUDA GPU.

Usage (smoke, ~5h on RTX 6000 Pro):
    python train/train_contrastive_grpo.py \
        --model_path model/OneRec-1.7B \
        --template model/qwen3_soft_switch.jinja2 \
        --train_parquet data/contrastive_dataset_v1_grpo/train.parquet \
        --valid_parquet data/contrastive_dataset_v1_grpo/valid.parquet \
        --output_dir runs/contrastive_grpo_smoke \
        --max_train_groups 5000 --max_eval_groups 1000 \
        --eval_steps 200 --save_steps 200 --logging_steps 25 \
        --per_device_batch_size 12 --grad_accum 1 \
        --lr 5e-5 --temperature 0.5 --kl_weight 0.5 \
        --merge_and_save \
        2>&1 | tee runs/grpo_smoke.log
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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset, Sampler
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


# ===========================================================================
# Dataset
# ===========================================================================


class ContrastiveDatasetGRPO(Dataset):
    """
    Each row = one (chosen, rejected) pair with single SID per side.
    Rows are kept sorted by (group_id, g) so that any G_per_group consecutive
    rows are exactly one group — required by GroupedSampler.
    """

    def __init__(self, parquet_path, tokenizer, G_per_group=DEFAULT_G,
                 max_hist=512, max_total_len=3072):
        df = pd.read_parquet(parquet_path)
        df = df.sort_values(["group_id", "g"]).reset_index(drop=True)
        assert len(df) % G_per_group == 0, (
            f"len={len(df)} not divisible by G={G_per_group}; "
            f"dataset / G mismatch"
        )
        # Spot-check: first 3 groups have G_per_group unique g values
        for i in range(min(3, len(df) // G_per_group)):
            block = df.iloc[i * G_per_group:(i + 1) * G_per_group]
            assert block["group_id"].nunique() == 1
            assert sorted(block["g"].tolist()) == list(range(G_per_group))

        self.df = df
        self.tokenizer = tokenizer
        self.G_per_group = G_per_group
        self.max_hist = max_hist
        self.max_total_len = max_total_len

    def __len__(self):
        return len(self.df)

    @property
    def num_groups(self):
        return len(self.df) // self.G_per_group

    def subsample_groups(self, n_groups, seed):
        """Return a new dataset narrowed to n_groups groups (G rows each)."""
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

        new_ds = object.__new__(ContrastiveDatasetGRPO)
        new_ds.df = new_df
        new_ds.tokenizer = self.tokenizer
        new_ds.G_per_group = self.G_per_group
        new_ds.max_hist = self.max_hist
        new_ds.max_total_len = self.max_total_len
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
        prompt_with_begin = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<|sid_begin|>"

        chosen_text = prompt_with_begin + sid_to_core_text(chosen_sid)
        rejected_text = prompt_with_begin + sid_to_core_text(rejected_sid)

        prompt_ids = self.tokenizer(prompt_with_begin, add_special_tokens=True)["input_ids"]
        chosen_ids = self.tokenizer(chosen_text, add_special_tokens=True)["input_ids"]
        rejected_ids = self.tokenizer(rejected_text, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        # Each special token tokenizes to 1 token; layout = prompt + 3.
        assert len(chosen_ids) == prompt_len + 3, (
            f"chosen tokens != prompt+3: {len(chosen_ids)} vs {prompt_len + 3}"
        )
        assert len(rejected_ids) == prompt_len + 3, (
            f"rejected tokens != prompt+3: {len(rejected_ids)} vs {prompt_len + 3}"
        )

        if self.max_total_len is not None and len(chosen_ids) > self.max_total_len:
            cut = len(chosen_ids) - self.max_total_len
            chosen_ids = chosen_ids[cut:]
            rejected_ids = rejected_ids[cut:]
            prompt_len = prompt_len - cut

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt_len": prompt_len,
            "group_id": group_id,
        }


# ===========================================================================
# Collate
# ===========================================================================


def grpo_collate(batch, pad_token_id):
    """
    (B pairs in batch) -> single (2B, L) tensor of chosen+rejected.
    Tracks group_ids alongside prompt_lens.
    """
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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens,
        "group_ids": group_ids,
    }


# ===========================================================================
# Sampler — yields G consecutive indices per group, shuffles group order
# ===========================================================================


class GroupedSampler(Sampler):
    """
    Yields indices in groups of G consecutive rows. Group order is
    shuffled per epoch (deterministic via seed + epoch). Pair order
    within a group is fixed (g=0, 1, ..., G-1) so the loss can rely
    on group structure.
    """

    def __init__(self, num_samples, G_per_group=3, shuffle=True, seed=42):
        assert num_samples % G_per_group == 0, (
            f"num_samples={num_samples} must be divisible by G={G_per_group}"
        )
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
    """Length=1 score: 3 SID tokens per (chosen|rejected) row."""
    bsz = input_ids.size(0)
    half = bsz // 2
    H = last_hidden.size(-1)
    plens = torch.cat([prompt_lens, prompt_lens], dim=0)
    ks = torch.arange(num_sid_tokens, device=last_hidden.device)
    pred_pos = plens.unsqueeze(1) - 1 + ks
    token_pos = plens.unsqueeze(1) + ks

    idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
    slice_hidden = last_hidden.gather(1, idx)
    slice_logits = lm_head(slice_hidden)
    log_probs = F.log_softmax(slice_logits.float(), dim=-1)

    target_ids = input_ids.gather(1, token_pos)
    token_logp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    scores = token_logp.mean(dim=1)
    return scores[:half], scores[half:], log_probs


def compute_ref_log_probs(ref_model, input_ids, attention_mask, prompt_lens,
                          num_sid_tokens=3):
    """
    Reference path with the same lm_head bypass as compute_sid_logprobs.

    Calling ref_model(...) directly would materialize a (2B, L, V) bf16
    logits tensor (~18 GB at bsz=12, scales linearly with bsz). Instead,
    walk into ref_model.model (Qwen3Model), gather hidden states at the K
    SID positions, and apply lm_head only to that (2B, K, H) slice — the
    full (2B, L, V) logits are never materialized.
    """
    transformer = ref_model.model      # Qwen3Model
    lm_head = ref_model.lm_head        # Linear
    with torch.no_grad():
        out = transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        last_hidden = out.last_hidden_state               # (2B, L, H)
        H = last_hidden.size(-1)
        plens = torch.cat([prompt_lens, prompt_lens], dim=0)
        ks = torch.arange(num_sid_tokens, device=last_hidden.device)
        pred_pos = plens.unsqueeze(1) - 1 + ks
        idx = pred_pos.unsqueeze(-1).expand(-1, -1, H)
        slice_hidden = last_hidden.gather(1, idx)         # (2B, K, H)
        slice_logits = lm_head(slice_hidden)              # (2B, K, V)
        return F.log_softmax(slice_logits.float(), dim=-1)


def compute_grpo_contrast_loss(chosen_s, rejected_s, temperature,
                               G_per_group, eps=1e-3):
    """
    Per-pair contrastive loss, then per-group divide by std (detached).

    Returns (l_grpo, l_pair_mean):
        l_grpo      : scalar loss to minimize
        l_pair_mean : raw per-pair mean loss, for monitoring only
    """
    margin = (chosen_s - rejected_s) / temperature
    l_pair = F.softplus(-margin)            # (B*G,)

    n = l_pair.size(0)
    assert n % G_per_group == 0, (
        f"batch pairs {n} not divisible by G={G_per_group} — collate / "
        f"sampler likely misaligned (per_device_batch_size must be a "
        f"multiple of G)."
    )
    B = n // G_per_group
    l_grouped = l_pair.view(B, G_per_group)

    std_g = l_grouped.std(dim=-1, keepdim=True, unbiased=False).detach() + eps
    l_grpo = (l_grouped / std_g).mean()

    return l_grpo, l_pair.mean()


# ===========================================================================
# Trainer
# ===========================================================================


class ContrastiveTrainerGRPO(Trainer):
    def __init__(self, *args, temperature=0.5, ref_model=None, kl_weight=0.5,
                 G_per_group=DEFAULT_G, sampler_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.ref_model = ref_model
        self.kl_weight = kl_weight
        self.G_per_group = G_per_group
        self.sampler_seed = sampler_seed

    # Override samplers so train + eval batches always contain whole groups.
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

        chosen_s, rejected_s, trained_log_probs = compute_sid_logprobs(
            last_hidden, lm_head, inputs["input_ids"], prompt_lens
        )

        l_contrast, l_pair_mean = compute_grpo_contrast_loss(
            chosen_s, rejected_s, self.temperature,
            G_per_group=self.G_per_group,
        )
        loss = l_contrast

        l_kl = None
        if self.ref_model is not None and self.kl_weight > 0:
            ref_log_probs = compute_ref_log_probs(
                self.ref_model,
                inputs["input_ids"],
                inputs["attention_mask"],
                prompt_lens,
            )
            trained_probs = trained_log_probs.exp()
            l_kl = (trained_probs * (trained_log_probs - ref_log_probs)) \
                .sum(dim=-1).mean()
            loss = loss + self.kl_weight * l_kl

        # Re-attach so prediction_step / metrics work as expected
        inputs["prompt_lens"] = prompt_lens
        if group_ids is not None:
            inputs["group_ids"] = group_ids

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
                  f"l_grpo={l_contrast.item():.3f} "
                  f"l_pair={l_pair_mean.item():.3f}{extra}")

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

    # GRPO group size (must match what built --train_parquet)
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
        help="Pairs per micro-batch. MUST be a multiple of --G. "
             "e.g. for G=3, 12 = 4 users × 3; for G=5, 15 = 3 users × 5.",
    )
    parser.add_argument("--grad_accum", type=int, default=1)
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
    parser.add_argument(
        "--max_train_groups", type=int, default=-1,
        help="Subsample train to this many GROUPS (each = G pairs). "
             "-1 = use full set. Smoke test: 5000.",
    )
    parser.add_argument(
        "--max_eval_groups", type=int, default=2000,
        help="Subsample eval to this many GROUPS (G pairs each).",
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
    parser.add_argument("--eval_steps", type=int, default=1500)
    parser.add_argument("--save_steps", type=int, default=1500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    assert args.per_device_batch_size % args.G == 0, (
        f"--per_device_batch_size ({args.per_device_batch_size}) must be a "
        f"multiple of --G ({args.G}). Try "
        f"{(args.per_device_batch_size // args.G + 1) * args.G}."
    )

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

    print(f"Loading datasets (G={args.G}) ...")
    train_set = ContrastiveDatasetGRPO(
        args.train_parquet, tokenizer,
        G_per_group=args.G,
        max_hist=args.max_hist, max_total_len=args.max_total_len,
    )
    valid_set = ContrastiveDatasetGRPO(
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

    print(f"  train: {_fmt(train_set.num_groups, train_full_groups)} groups  "
          f"({len(train_set):,} pairs)")
    print(f"  valid: {_fmt(valid_set.num_groups, valid_full_groups)} groups  "
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

    trainer = ContrastiveTrainerGRPO(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate,
        compute_metrics=compute_metrics,
        temperature=args.temperature,
        ref_model=ref_model,
        kl_weight=args.kl_weight,
        G_per_group=args.G,
        sampler_seed=args.seed,
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
