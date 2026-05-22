"""ai-model/merge_and_convert.py — Shard-단위 LoRA 머지 (peak RAM ~2GB).

Commit limit이 모델 전체 크기보다 작은 머신용. 32GB RAM 가정.
1. LoRA 어댑터 텐서 80개 RAM 적재 (~25MB)
2. base safetensors shard 28개를 1개씩 순회 — qkv는 W + scaling*B@A 머지, 나머지는 그대로 복사
3. tokenizer + config 복사
"""
import json
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

BASE_DIR     = Path(r"E:\hf-cache\hub\models--EleutherAI--polyglot-ko-12.8b\snapshots\09dfc839067bf44e7f52976eca8adbc17f04e1b0")
ADAPTER_DIR  = Path("koalpaca_save")
MERGED_DIR   = Path("koalpaca_merged")
MERGED_DIR.mkdir(exist_ok=True)

# adapter_config.json: r=8, lora_alpha=32 → scaling = 32/8 = 4
ADAPTER_CFG = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
SCALING = ADAPTER_CFG["lora_alpha"] / ADAPTER_CFG["r"]
print(f"[cfg] LoRA scaling = alpha({ADAPTER_CFG['lora_alpha']}) / r({ADAPTER_CFG['r']}) = {SCALING}")

print("[1/4] Loading LoRA adapter (80 tensors, ~25MB)...")
loras = {}
with safe_open(ADAPTER_DIR / "adapter_model.safetensors", framework="pt") as f:
    for k in f.keys():
        # k = "base_model.model.gpt_neox.layers.{i}.attention.query_key_value.lora_X.weight"
        base_key = k.replace("base_model.model.", "")
        # base_key = "gpt_neox.layers.{i}.attention.query_key_value.lora_X.weight"
        loras[base_key] = f.get_tensor(k)
print(f"       loaded {len(loras)} LoRA tensors")

# Index per-layer: target_key -> (A, B)
lora_pairs = {}
for k, v in loras.items():
    if ".lora_A.weight" in k:
        target = k.replace(".lora_A.weight", ".weight")
        lora_pairs.setdefault(target, {})["A"] = v
    elif ".lora_B.weight" in k:
        target = k.replace(".lora_B.weight", ".weight")
        lora_pairs.setdefault(target, {})["B"] = v
print(f"       {len(lora_pairs)} layers will be merged")

print("[2/4] Iterating 28 base shards...")
INDEX = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
shards = sorted(set(INDEX["weight_map"].values()))
merged_count = 0
for shard_name in shards:
    src = BASE_DIR / shard_name
    dst = MERGED_DIR / shard_name
    new_tensors = {}
    with safe_open(src, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            if k in lora_pairs:
                A = lora_pairs[k]["A"]  # [r, in]
                B = lora_pairs[k]["B"]  # [out, r]
                delta = (SCALING * (B.float() @ A.float())).to(t.dtype)
                t = t + delta
                merged_count += 1
                print(f"       merged {k} (shard {shard_name})")
            new_tensors[k] = t.contiguous()
    save_file(new_tensors, dst, metadata={"format": "pt"})
    del new_tensors
print(f"       merged {merged_count}/{len(lora_pairs)} target weights into {len(shards)} shards")
assert merged_count == len(lora_pairs), f"merged {merged_count} != expected {len(lora_pairs)}"

print("[3/4] Copying config/tokenizer files...")
# From base: config.json, generation_config.json, model.safetensors.index.json
for fn in ["config.json", "generation_config.json", "model.safetensors.index.json"]:
    src = BASE_DIR / fn
    if src.exists():
        shutil.copy2(src, MERGED_DIR / fn)
        print(f"       copied {fn}")
# From adapter: tokenizer files
for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
    src = ADAPTER_DIR / fn
    if src.exists():
        shutil.copy2(src, MERGED_DIR / fn)
        print(f"       copied {fn}")

print(f"[4/4] Done: {MERGED_DIR.resolve()}")
