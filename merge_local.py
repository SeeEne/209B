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
                    help="Local path to base model dir (e.g. model/OneRec-1.7B). "
                         "For DPO Stage 2 adapters, point this to the SFT merged "
                         "output (runs/sft_only_50k/merged or model/OneRec-1.7B-sft50k/merged).")
parser.add_argument("--adapter", required=True,
                    help="Path to LoRA adapter / training checkpoint dir.")
parser.add_argument("--out", required=True,
                    help="Where to write the merged model.")
parser.add_argument("--device", default=None,
                    choices=["cuda", "cpu", "mps"],
                    help="Device for the merge. Defaults to cuda if available, "
                         "else cpu (mps is experimental and bf16 is unstable there).")
args = parser.parse_args()

if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

# bf16 throughout: matches training dtype on GPU and keeps disk footprint
# small on CPU (~3.4 GB per 1.7B-param checkpoint). Torch 2.1+ supports
# bf16 matmul on CPU; merging is mostly weight-add so even older builds work.
print(f"Loading base from {args.base} on {args.device} (bf16) ...")
base = AutoModelForCausalLM.from_pretrained(
    args.base,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": args.device} if args.device == "cuda" else None,
)
if args.device != "cuda":
    base = base.to(args.device)

print(f"Attaching adapter from {args.adapter} ...")
merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

print(f"Saving merged model to {args.out} ...")
merged.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.out)
print("done")
