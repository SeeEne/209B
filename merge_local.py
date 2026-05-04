"""Merge LoRA adapter into a LOCAL base model checkpoint (offline server).

Differences from merge.py:
  - Base model is a local directory path, not a HF Hub repo id.
  - Adapter / output paths are CLI args so checkpoint number is not
    hardcoded (smoke / full runs end at different step numbers).

Example:
    python merge_local.py \
        --base model/OneRec-1.7B \
        --adapter runs/contrastive_v0_smoke/checkpoint-156 \
        --out runs/contrastive_v0_smoke/merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True,
                    help="Local path to OneRec-1.7B model dir "
                         "(e.g. model/OneRec-1.7B).")
parser.add_argument("--adapter", required=True,
                    help="Path to LoRA adapter / training checkpoint dir.")
parser.add_argument("--out", required=True,
                    help="Where to write the merged model.")
args = parser.parse_args()

print(f"Loading base from {args.base} ...")
base = AutoModelForCausalLM.from_pretrained(
    args.base,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
)

print(f"Attaching adapter from {args.adapter} ...")
merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

print(f"Saving merged model to {args.out} ...")
merged.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
print("done")
