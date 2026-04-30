"""
dataset.py

ContrastiveDataset for OneRec contrastive fine-tuning on contrastive_dataset_v0.

Each example yields tokenized chosen/rejected sequences sharing a common prompt
prefix (system + user-history + <|sid_begin|>). The chosen and rejected SID
tokens (3 tokens each) are appended for next-token-likelihood scoring.

Output per example:
    {
        "chosen_input_ids":   LongTensor[L_chosen],
        "rejected_input_ids": LongTensor[L_rejected],
        "prompt_len":         int   # shared prompt length in tokens
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

# OneRec system prompt — same as official benchmark for video task
SYSTEM_PROMPT = (
    "你是一位视频推荐系统专家，擅长捕捉用户的兴趣演变。"
    "请根据历史序列推荐后续视频。"
)


def sid_to_text(sid) -> str:
    """Convert a 3-int SID array/list to its <|sid_begin|>...<|sid_end|> string."""
    a, b, c = int(sid[0]), int(sid[1]), int(sid[2])
    return f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>"


def sid_to_core_text(sid) -> str:
    """Convert a 3-int SID to its core 3-token string (no sid_begin/end wrapping)."""
    a, b, c = int(sid[0]), int(sid[1]), int(sid[2])
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def build_history_text(hist_sids, max_hist: Optional[int] = None) -> str:
    """Concatenate history SIDs into the user-message text expected by OneRec."""
    if max_hist is not None and len(hist_sids) > max_hist:
        hist_sids = hist_sids[-max_hist:]
    return "".join(sid_to_text(s) for s in hist_sids)


class ContrastiveDataset(Dataset):
    """
    Loads contrastive_dataset_v0 train/valid parquet and tokenizes on-the-fly.

    Args:
        parquet_path:   path to train.parquet or valid.parquet
        tokenizer:      a OneRec tokenizer with chat_template already set
        max_hist:       optional cap on history length (in items). OneRec
                        already truncates at 512, so the default 512 is a
                        no-op for clean rows. Use a smaller value only for
                        memory-constrained CPU smoke tests.
        max_total_len:  safety cap on total token length per sequence.
                        512 items × 5 tokens + chat template + 3 SID tokens
                        ≈ 2600 tokens; 3072 leaves headroom.
    """

    def __init__(self, parquet_path, tokenizer, max_hist: Optional[int] = 512,
                 max_total_len: Optional[int] = 3072):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_hist = max_hist
        self.max_total_len = max_total_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        hist_sids = row["hist_sids"]
        chosen_sids = row["chosen_sids"]    # list of [a,b,c]
        rejected_sids = row["rejected_sids"]

        hist_text = build_history_text(hist_sids, max_hist=self.max_hist)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": hist_text},
        ]

        # Build prompt (ends right before the SID we want to score)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "<|sid_begin|>"

        # Build full sequences = prompt + 3 SID tokens (chosen / rejected).
        # We score only the 3 SID tokens — <|sid_end|> is deterministic, omit it.
        chosen_text = prompt + sid_to_core_text(chosen_sids[0])
        rejected_text = prompt + sid_to_core_text(rejected_sids[0])

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        chosen_ids = self.tokenizer(chosen_text, add_special_tokens=True)["input_ids"]
        rejected_ids = self.tokenizer(rejected_text, add_special_tokens=True)["input_ids"]

        prompt_len = len(prompt_ids)

        # Sanity: chosen/rejected should be exactly prompt_len + 3 tokens.
        # If tokenizer merges differently, surface this so we can investigate.
        assert len(chosen_ids) == prompt_len + 3, (
            f"chosen tokens != prompt+3: {len(chosen_ids)} vs {prompt_len + 3}"
        )
        assert len(rejected_ids) == prompt_len + 3, (
            f"rejected tokens != prompt+3: {len(rejected_ids)} vs {prompt_len + 3}"
        )

        # Optional truncation from the LEFT of the prompt (keep the SID tokens
        # at the end). This preserves recent history, the chat template tail,
        # and the 3 scored tokens.
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


def contrastive_collate(batch, pad_token_id: int):
    """
    Right-pads chosen and rejected sequences to a common length.
    Returns a single tensor of shape (2*B, L) so chosen and rejected are
    processed in one forward pass.

    The first B rows are chosen, the last B are rejected. prompt_lens has
    the per-example prompt length (same for chosen and rejected since they
    share the prefix).
    """
    B = len(batch)
    chosen = [b["chosen_input_ids"] for b in batch]
    rejected = [b["rejected_input_ids"] for b in batch]
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)

    all_seqs = chosen + rejected
    max_len = max(s.size(0) for s in all_seqs)

    input_ids = torch.full((2 * B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((2 * B, max_len), dtype=torch.long)
    for i, s in enumerate(all_seqs):
        input_ids[i, : s.size(0)] = s
        attention_mask[i, : s.size(0)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens,   # length B (shared by chosen+rejected)
    }
