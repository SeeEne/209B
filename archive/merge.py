
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ck = 'runs/contrastive_v0_smoke/checkpoint-313'
base = AutoModelForCausalLM.from_pretrained('OpenOneRec/OneRec-1.7B', trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'})
merged = PeftModel.from_pretrained(base, ck).merge_and_unload()
merged.save_pretrained('runs/contrastive_v0_smoke/merged')
AutoTokenizer.from_pretrained('OpenOneRec/OneRec-1.7B', trust_remote_code=True).save_pretrained('runs/contrastive_v0_smoke/merged')
print('done')

