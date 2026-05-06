"""
dataset.py

Prompt helpers + shared Dataset / collate / sampler classes for the active
SFT, DPO, and ORPO trainers. Also keeps the legacy ``ContrastiveDataset`` /
``contrastive_collate`` used by ``archive/`` scripts.

Active exports (used by all four trainers in ``train/``):
    * ``SYSTEM_PROMPT``, ``sid_to_text``, ``sid_to_core_text``,
      ``build_history_text``       — prompt construction helpers.
    * ``PairedSIDDataset``         — chosen + rejected pairs from v1_grpo;
                                     optional ref-score forwarding.
    * ``ChosenSIDDataset``         — chosen-only view (SFT).
    * ``paired_collate`` /
      ``chosen_collate``           — pad + pack helpers.
    * ``GroupedSampler``           — emits G consecutive rows per group.

Each item from these datasets shares the same prompt construction:
    system + user(history) + chat_template(...) + "<|sid_begin|>"
followed by the 3 SID core tokens being scored. Truncation, group-aware
subsampling, and the prompt-length invariant (``len(full) == prompt_len + 3``)
are factored into a base class so a fix lands once for every trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

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


# ===========================================================================
# Shared primitives for the active trainers (DPO joint, DPO-from-SFT, ORPO,
# SFT-only). All four read v1_grpo per-pair parquet and emit prompts of the
# same shape; the differences are which sides are scored, what loss runs, and
# whether group_id / ref_scores are forwarded.
# ===========================================================================


def _build_prompt(tokenizer, hist_sids, max_hist):
    hist_text = build_history_text(hist_sids, max_hist=max_hist)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": hist_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    ) + "<|sid_begin|>"


def _tokenize_with_sids(tokenizer, prompt: str,
                        sids: Sequence) -> tuple[List[List[int]], int]:
    """Tokenize prompt + each SID. Asserts every full sequence == prompt_len+3.

    Returns ``(list_of_full_ids, prompt_len)``.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)
    full_ids: List[List[int]] = []
    for sid in sids:
        text = prompt + sid_to_core_text(sid)
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        assert len(ids) == prompt_len + 3, (
            f"tokens != prompt+3: {len(ids)} vs {prompt_len + 3}"
        )
        full_ids.append(ids)
    return full_ids, prompt_len


def _truncate_left(full_ids_list: List[List[int]], prompt_len: int,
                   max_total_len: Optional[int]) -> tuple[List[List[int]], int]:
    """Left-truncate the prompt prefix so each sequence fits ``max_total_len``.

    Asserts the new prompt_len stays positive — pathological rows where the
    chat-template + history alone exceed ``max_total_len - 3`` would otherwise
    push prompt_len negative and silently corrupt SID position computation.
    """
    if max_total_len is None:
        return full_ids_list, prompt_len
    cur_max = max(len(ids) for ids in full_ids_list)
    if cur_max <= max_total_len:
        return full_ids_list, prompt_len
    cut = cur_max - max_total_len
    new_prompt_len = prompt_len - cut
    assert new_prompt_len > 0, (
        f"left-truncation would push prompt_len <= 0 "
        f"(prompt_len={prompt_len}, cut={cut}). Increase max_total_len "
        f"or shrink max_hist."
    )
    return [ids[cut:] for ids in full_ids_list], new_prompt_len


class _GroupedSIDDataset(Dataset):
    """v1_grpo-backed base class. Sorts rows by ``(group_id, g)`` so any G
    consecutive rows form one group — required for group-aware samplers and
    losses. Subsampling preserves whole groups so ``len(df) % G == 0`` always
    holds."""

    def __init__(self, parquet_path, tokenizer, G_per_group: int = 3,
                 max_hist: int = 512, max_total_len: int = 3072):
        df = pd.read_parquet(parquet_path)
        if "group_id" in df.columns and "g" in df.columns:
            df = df.sort_values(["group_id", "g"])
        self.df = df.reset_index(drop=True)
        assert len(self.df) % G_per_group == 0, (
            f"len={len(self.df)} not divisible by G={G_per_group}; "
            f"dataset / G mismatch"
        )
        if "group_id" in self.df.columns and "g" in self.df.columns:
            for i in range(min(3, len(self.df) // G_per_group)):
                block = self.df.iloc[i * G_per_group:(i + 1) * G_per_group]
                assert block["group_id"].nunique() == 1, (
                    f"row block {i} not in a single group_id"
                )
                assert sorted(block["g"].tolist()) == list(range(G_per_group)), (
                    f"g values in row block {i} are not 0..G-1"
                )
        self.tokenizer = tokenizer
        self.G_per_group = G_per_group
        self.max_hist = max_hist
        self.max_total_len = max_total_len

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_groups(self) -> int:
        return len(self.df) // self.G_per_group

    def _clone_with_df(self, new_df: pd.DataFrame) -> "_GroupedSIDDataset":
        new_ds = object.__new__(type(self))
        new_ds.df = new_df
        new_ds.tokenizer = self.tokenizer
        new_ds.G_per_group = self.G_per_group
        new_ds.max_hist = self.max_hist
        new_ds.max_total_len = self.max_total_len
        return new_ds

    def subsample_groups(self, n_groups: int, seed: int) -> "_GroupedSIDDataset":
        """Return a new dataset keeping ``n_groups`` whole groups (G rows each).
        Always subsample BEFORE attaching ref scores — the per-row ref index
        cannot survive arbitrary row permutation."""
        if n_groups >= self.num_groups:
            return self
        rng = torch.Generator().manual_seed(seed)
        all_groups = torch.randperm(self.num_groups, generator=rng).tolist()
        keep_gids = sorted(all_groups[:n_groups])
        keep_indices: List[int] = []
        for gid in keep_gids:
            for g in range(self.G_per_group):
                keep_indices.append(gid * self.G_per_group + g)
        new_df = self.df.iloc[keep_indices].reset_index(drop=True)
        return self._clone_with_df(new_df)


class PairedSIDDataset(_GroupedSIDDataset):
    """Per-pair v1_grpo view. Yields chosen + rejected for each row, plus
    ``group_id`` (always present in v1_grpo). When ``self.ref_scores`` is set
    to a precomputed ``(N, 2)`` tensor, each item also forwards
    ``ref_chosen_score`` / ``ref_rejected_score`` for the DPO trainers."""

    def __init__(self, parquet_path, tokenizer, G_per_group: int = 3,
                 max_hist: int = 512, max_total_len: int = 3072):
        super().__init__(parquet_path, tokenizer, G_per_group,
                         max_hist, max_total_len)
        self.ref_scores: Optional[torch.Tensor] = None

    def _clone_with_df(self, new_df: pd.DataFrame) -> "PairedSIDDataset":
        new_ds = super()._clone_with_df(new_df)
        new_ds.ref_scores = None  # invalidates on subsample (indices change)
        return new_ds

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        prompt = _build_prompt(self.tokenizer, row["hist_sids"], self.max_hist)
        (chosen_ids, rejected_ids), prompt_len = _tokenize_with_sids(
            self.tokenizer, prompt, [row["chosen_sid"], row["rejected_sid"]],
        )
        (chosen_ids, rejected_ids), prompt_len = _truncate_left(
            [chosen_ids, rejected_ids], prompt_len, self.max_total_len,
        )
        item = {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt_len": prompt_len,
            "group_id": int(row["group_id"]),
        }
        if self.ref_scores is not None:
            item["ref_chosen_score"] = float(self.ref_scores[idx, 0].item())
            item["ref_rejected_score"] = float(self.ref_scores[idx, 1].item())
        return item


class ChosenSIDDataset(_GroupedSIDDataset):
    """Chosen-only view of v1_grpo (rejected column ignored). Used by the
    SFT-only trainer. Group structure is preserved only so subsampling stays
    aligned with the DPO/ORPO arms — the SFT loss does not consume groups."""

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        prompt = _build_prompt(self.tokenizer, row["hist_sids"], self.max_hist)
        (chosen_ids,), prompt_len = _tokenize_with_sids(
            self.tokenizer, prompt, [row["chosen_sid"]],
        )
        (chosen_ids,), prompt_len = _truncate_left(
            [chosen_ids], prompt_len, self.max_total_len,
        )
        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "prompt_len": prompt_len,
        }


def paired_collate(batch, pad_token_id: int):
    """Pack chosen + rejected into ``(2B, L)`` (first B = chosen, last B =
    rejected). Forwards ``group_ids`` whenever the dataset emits ``group_id``,
    and ``ref_chosen_scores`` / ``ref_rejected_scores`` when ref scores are
    attached. Trainers that don't consume those keys can simply ignore them."""
    B = len(batch)
    chosen = [b["chosen_input_ids"] for b in batch]
    rejected = [b["rejected_input_ids"] for b in batch]
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)

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
    }
    if "group_id" in batch[0]:
        out["group_ids"] = torch.tensor(
            [b["group_id"] for b in batch], dtype=torch.long,
        )
    if "ref_chosen_score" in batch[0]:
        out["ref_chosen_scores"] = torch.tensor(
            [b["ref_chosen_score"] for b in batch], dtype=torch.float32,
        )
        out["ref_rejected_scores"] = torch.tensor(
            [b["ref_rejected_score"] for b in batch], dtype=torch.float32,
        )
    return out


def chosen_collate(batch, pad_token_id: int):
    """Pack chosen-only into ``(B, L)``. For the SFT-only trainer."""
    B = len(batch)
    seqs = [b["chosen_input_ids"] for b in batch]
    prompt_lens = torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long)
    max_len = max(s.size(0) for s in seqs)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :s.size(0)] = s
        attention_mask[i, :s.size(0)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lens": prompt_lens,
    }


class GroupedSampler(Sampler):
    """Yields G consecutive rows per group. With ``shuffle=True`` the GROUP
    order is randomized but the G rows within each group stay contiguous —
    required so ``view(B, G)`` in the loss aligns with one user's pairs."""

    def __init__(self, num_samples: int, G_per_group: int = 3,
                 shuffle: bool = True, seed: int = 42):
        assert num_samples % G_per_group == 0, (
            f"num_samples={num_samples} must be divisible by G={G_per_group}"
        )
        self.num_samples = num_samples
        self.G = G_per_group
        self.num_groups = num_samples // G_per_group
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
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

    def __len__(self) -> int:
        return self.num_samples
