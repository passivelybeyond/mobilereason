"""
Export MobileReason-MoE → GGUF (4-bit quantized)
For inference on GTX 1650 (4GB) or mobile via llama.cpp / MLC-LLM

Steps:
1. Load trained PyTorch checkpoint
2. Convert weights to llama.cpp-compatible format (safetensors)
3. Run llama.cpp's quantization tool (Q4_K_M recommended)

Requirements:
  pip install safetensors transformers
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make
"""

import sys
import json
import struct
import torch
import numpy as np
from pathlib import Path
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from model.architecture import MobileReasonMoE, MobileReasonConfig


# ──────────────────────────────────────────────
# 1. Load checkpoint
# ──────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu") -> MobileReasonMoE:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = MobileReasonConfig(**ckpt["model_config"])
    model = MobileReasonMoE(model_cfg)
    state = ckpt["model_state"]
    # Strip DDP prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from step {ckpt['step']}, loss={ckpt['loss']:.4f}")
    return model, model_cfg


# ──────────────────────────────────────────────
# 2. Save as safetensors (llama.cpp convert script needs this)
# ──────────────────────────────────────────────

def save_safetensors(model: MobileReasonMoE, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg

    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(out_dir / "model.safetensors"))

    # Save HuggingFace-style config.json for llama.cpp convert.py compatibility
    hf_config = {
        "architectures": ["MobileReasonMoE"],
        "model_type": "mobilereason_moe",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_dim,
        "num_hidden_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_heads,
        "num_key_value_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.ffn_hidden_dim,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.rms_norm_eps,
        # MoE fields
        "num_local_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_active_experts,
        "moe_every_n_layers": cfg.moe_every_n_layers,
        "expert_intermediate_size": cfg.expert_hidden_dim,
        "shared_expert_intermediate_size": cfg.shared_expert_hidden_dim,
        "torch_dtype": "float16",
        "tie_word_embeddings": cfg.tie_embeddings,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"Saved safetensors to {out_dir}/model.safetensors")
    print(f"Saved config to {out_dir}/config.json")
    return out_dir


# ──────────────────────────────────────────────
# 3. Write minimal GGUF file directly
#    (for models llama.cpp doesn't natively support yet)
# ──────────────────────────────────────────────

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

def write_gguf_string(f, s: str):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)

def write_gguf_value(f, type_id: int, value):
    f.write(struct.pack("<I", type_id))
    if type_id == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    elif type_id == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", value))
    elif type_id == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", value))

def write_tensor_info(f, name: str, shape, dtype_enum: int, offset: int):
    write_gguf_string(f, name)
    f.write(struct.pack("<I", len(shape)))
    for s in shape:
        f.write(struct.pack("<Q", s))
    f.write(struct.pack("<I", dtype_enum))
    f.write(struct.pack("<Q", offset))

def export_gguf(model: MobileReasonMoE, out_path: str, quantize_q4: bool = True):
    """
    Write a minimal GGUF file. For production use, run llama.cpp's convert.py
    on the safetensors output, then run quantize binary.
    This is a simplified direct writer for reference.
    """
    print(f"Exporting GGUF to {out_path}...")
    cfg = model.cfg
    state = model.state_dict()

    # For Q4_K_M: use llama.cpp CLI after saving safetensors
    # This function saves fp16 GGUF as a starting point
    GGUF_F16 = 1

    metadata = [
        ("general.architecture", GGUF_TYPE_STRING, "mobilereason"),
        ("general.name", GGUF_TYPE_STRING, "MobileReason-MoE"),
        ("mobilereason.vocab_size", GGUF_TYPE_UINT32, cfg.vocab_size),
        ("mobilereason.hidden_size", GGUF_TYPE_UINT32, cfg.hidden_dim),
        ("mobilereason.num_layers", GGUF_TYPE_UINT32, cfg.num_layers),
        ("mobilereason.num_attention_heads", GGUF_TYPE_UINT32, cfg.num_heads),
        ("mobilereason.num_kv_heads", GGUF_TYPE_UINT32, cfg.num_kv_heads),
        ("mobilereason.max_seq_len", GGUF_TYPE_UINT32, cfg.max_seq_len),
        ("mobilereason.num_experts", GGUF_TYPE_UINT32, cfg.num_experts),
        ("mobilereason.num_active_experts", GGUF_TYPE_UINT32, cfg.num_active_experts),
    ]

    # Prepare tensors as fp16 numpy arrays
    tensors = {}
    for k, v in state.items():
        tensors[k] = v.to(torch.float16).numpy()

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        for key, type_id, value in metadata:
            write_gguf_string(f, key)
            write_gguf_value(f, type_id, value)

        # Alignment
        alignment = 32
        offset = 0
        tensor_offsets = []
        for name, arr in tensors.items():
            tensor_offsets.append((name, arr.shape, offset))
            offset += arr.nbytes
            offset = (offset + alignment - 1) & ~(alignment - 1)

        for name, shape, off in tensor_offsets:
            write_tensor_info(f, name, list(shape), GGUF_F16, off)

        # Tensor data
        for (name, shape, off), (_, arr) in zip(tensor_offsets, tensors.items()):
            f.write(arr.tobytes())
            pad = ((arr.nbytes + alignment - 1) & ~(alignment - 1)) - arr.nbytes
            f.write(b"\x00" * pad)

    size_gb = Path(out_path).stat().st_size / 1e9
    print(f"GGUF saved: {size_gb:.2f} GB (fp16)")
    print(f"\nTo quantize to Q4_K_M (for GTX 1650):")
    print(f"  ./llama.cpp/quantize {out_path} model_q4.gguf Q4_K_M")
    print(f"  Expected size after Q4: ~{size_gb * 0.28:.2f} GB")


# ──────────────────────────────────────────────
# 4. Recommended llama.cpp workflow
# ──────────────────────────────────────────────

LLAMA_CPP_INSTRUCTIONS = """
=== RECOMMENDED EXPORT WORKFLOW ===

# Step 1: Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1   # compile with CUDA for GTX 1650

# Step 2: Convert safetensors → GGUF (fp16)
python convert_hf_to_gguf.py /path/to/safetensors/dir \\
    --outfile model_fp16.gguf \\
    --outtype f16

# Step 3: Quantize to Q4_K_M
./quantize model_fp16.gguf model_q4km.gguf Q4_K_M

# Step 4: Run inference on GTX 1650
./llama-cli -m model_q4km.gguf \\
    -n 200 \\
    --gpu-layers 32 \\
    -p "Once upon a time"

# Expected performance on GTX 1650:
#   Model size after Q4_K_M: ~500MB–800MB
#   Tokens/sec: ~15–30 tok/s (fits in 4GB VRAM with --gpu-layers 20-24)
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="export")
    parser.add_argument("--format", choices=["safetensors", "gguf", "both"], default="both")
    args = parser.parse_args()

    model, model_cfg = load_model_from_checkpoint(args.checkpoint)

    if args.format in ("safetensors", "both"):
        save_safetensors(model, args.output_dir)

    if args.format in ("gguf", "both"):
        export_gguf(model, str(Path(args.output_dir) / "model_fp16.gguf"))

    print(LLAMA_CPP_INSTRUCTIONS)
