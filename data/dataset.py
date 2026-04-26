"""
Dataset Pipeline
Streaming multi-source dataset mixer:
- FineWeb-Edu (50%) — general web text
- BookCorpus / PG19 (30%) — books and literature  
- OpenWebText / Reddit-like (20%) — social/chat text

All sources streamed to avoid downloading 100s of GBs.
Packed into fixed-length sequences for efficient training.
"""

import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from typing import Iterator


# ──────────────────────────────────────────────
# Source configs
# ──────────────────────────────────────────────

DATASET_SOURCES = [
    {
        "name": "fineweb",
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "hf_name": "sample-10BT",
        "split": "train",
        "field": "text",
        "weight": 0.5,
    },
    {
        "name": "books",
        "hf_path": "bookcorpus",
        "hf_name": None,
        "split": "train",
        "field": "text",
        "weight": 0.3,
    },
    {
        "name": "reddit",
        "hf_path": "Skylion007/openwebtext",
        "hf_name": None,
        "split": "train",
        "field": "text",
        "weight": 0.2,
    },
]


# ──────────────────────────────────────────────
# Streaming dataset per source
# ──────────────────────────────────────────────

def stream_source(source: dict, tokenizer, min_tokens: int = 64) -> Iterator[list[int]]:
    """Yields tokenized documents from a single source."""
    kwargs = {"streaming": True, "split": source["split"]}
    if source["hf_name"]:
        kwargs["name"] = source["hf_name"]
    ds = load_dataset(source["hf_path"], **kwargs)

    for item in ds:
        text = item[source["field"]]
        if not text or len(text.strip()) < 50:
            continue
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(tokens) < min_tokens:
            continue
        yield tokens


# ──────────────────────────────────────────────
# Sequence packing: fills context window efficiently
# ──────────────────────────────────────────────

class PackedSequenceDataset(IterableDataset):
    """
    Streams and packs tokenized documents into fixed-length sequences.
    Documents are concatenated end-to-end (with EOS boundary) and then
    cut into seq_len chunks. No padding waste.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        sources: list = DATASET_SOURCES,
        seed: int = 42,
        worker_id: int = 0,
        num_workers: int = 1,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sources = sources
        self.seed = seed
        self.worker_id = worker_id
        self.num_workers = num_workers

    def _weighted_stream(self):
        """Interleaves all sources according to their weights."""
        rng = random.Random(self.seed + self.worker_id)
        names = [s["name"] for s in self.sources]
        weights = [s["weight"] for s in self.sources]
        iters = {
            s["name"]: stream_source(s, self.tokenizer)
            for s in self.sources
        }

        while iters:
            active_names = list(iters.keys())
            active_weights = [weights[names.index(n)] for n in active_names]
            chosen = rng.choices(active_names, weights=active_weights)[0]
            try:
                yield next(iters[chosen])
            except StopIteration:
                del iters[chosen]

    def __iter__(self):
        buffer = []
        for doc_tokens in self._weighted_stream():
            buffer.extend(doc_tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def make_dataloader(
    tokenizer,
    seq_len: int = 2048,
    batch_size: int = 8,
    num_workers: int = 2,
    sources: list = DATASET_SOURCES,
    seed: int = 42,
) -> DataLoader:
    dataset = PackedSequenceDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        sources=sources,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ──────────────────────────────────────────────
# Token count estimator
# ──────────────────────────────────────────────

def estimate_dataset_tokens():
    """Rough estimate of available tokens per source."""
    estimates = {
        "fineweb-edu (10BT sample)": "~10B tokens",
        "bookcorpus": "~1B tokens",
        "openwebtext": "~8B tokens",
        "total weighted mix": "~10–12B tokens usable",
    }
    for k, v in estimates.items():
        print(f"  {k:35s}: {v}")
    print("\nTarget training: 10B tokens (~1 full pass through mix)")
    print("Kaggle 2×T4 throughput: ~50K–80K tokens/sec → ~35–55 hours")
    print("Tip: Use multiple Kaggle sessions with checkpointing!")


if __name__ == "__main__":
    print("Dataset token estimates:")
    estimate_dataset_tokens()
