# MobileReason-MoE

A from-scratch nano-scale language model trained on human-generated text.

## Architecture
- **GQA** (Grouped Query Attention, 16Q / 4KV heads) → smaller KV cache
- **RoPE** positional embeddings → length generalization
- **RMSNorm** (Pre-LN) → stable training
- **Sparse MoE** (8 experts, top-2, every 2nd layer) + shared expert → high capacity at low compute
- **SwiGLU** FFN for dense layers

## Model Size
| | Value |
|---|---|
| Total params | ~600M–1.2B |
| Active params per forward | ~160–300M |
| fp16 size | ~1.2–2.4 GB |
| Q4_K_M GGUF size | ~400–700 MB |

## Dataset Mix
| Source | Weight | Content |
|---|---|---|
| FineWeb-Edu | 50% | High-quality web text |
| BookCorpus | 30% | Books & literature |
| OpenWebText | 20% | Reddit / social text |

## Files
```
mobilereason/
├── model/
│   └── architecture.py       # Full model: GQA, RoPE, MoE, generate()
├── data/
│   ├── tokenizer_train.py    # BPE tokenizer (32K vocab)
│   └── dataset.py            # Streaming multi-source dataset
├── training/
│   └── train.py              # DDP training loop
├── scripts/
│   └── export_gguf.py        # Export to GGUF for llama.cpp
└── kaggle_notebook.ipynb     # All-in-one Kaggle notebook
```

## How to Train (Kaggle 2×T4)

1. Upload all files to Kaggle
2. Open `kaggle_notebook.ipynb`
3. Enable **GPU T4×2** accelerator
4. Run cells 1–5 in order
5. On session expiry, re-run from the **Resume Training** cell

Expected throughput: ~50K–80K tokens/sec → ~35–55 hours for 10B tokens

## Inference on GTX 1650

After training:
```bash
# Export
python scripts/export_gguf.py --checkpoint checkpoints/step_XXXXXXX.pt --format both

# Quantize (llama.cpp)
./llama.cpp/quantize export/model_fp16.gguf export/model_q4km.gguf Q4_K_M

# Run
./llama.cpp/llama-cli -m export/model_q4km.gguf \
    --gpu-layers 20 \
    -n 200 \
    -p "Once upon a time"
```

Expected: ~15–30 tok/s on GTX 1650 with Q4_K_M

## Next Steps (Phase 2)
- Fine-tune on longer books and narrative text
- Add latent reasoning tokens (`<think>` / `</think>`)
- RLVR training on verifiable tasks
- MLC-LLM export for Android/iOS
