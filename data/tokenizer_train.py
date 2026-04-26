"""
Tokenizer Training
- BPE via SentencePiece (32K vocab)
- Trained on a sample of our target dataset mix
- Handles: general text, books, social/chat text
"""

import os
import sentencepiece as spm
import json
from pathlib import Path


TOKENIZER_DIR = Path("tokenizer")
TOKENIZER_DIR.mkdir(exist_ok=True)

SAMPLE_TEXT_PATH = TOKENIZER_DIR / "tokenizer_sample.txt"
MODEL_PREFIX = str(TOKENIZER_DIR / "mobilereason")

# ──────────────────────────────────────────────
# 1. Collect sample text for tokenizer training
# ──────────────────────────────────────────────

def collect_sample_text(output_path: Path, max_chars: int = 50_000_000):
    """
    Streams a sample from HuggingFace datasets to train the tokenizer.
    Uses: FineWeb-Edu (web), BookCorpus (books), pushshift Reddit (social/chat)
    Run this ONCE on Kaggle to generate tokenizer_sample.txt
    """
    from datasets import load_dataset

    sources = []

    print("Loading FineWeb-Edu sample...")
    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    sources.append(("fineweb", fw, "text", 0.5))

    print("Loading books (pg19)...")
    # bookcorpus scripts are deprecated; use pg19 (Project Gutenberg books, HF-hosted parquet)
    books = load_dataset("deepmind/pg19", split="train", streaming=True)
    sources.append(("books", books, "text", 0.3))

    print("Loading Reddit/social (OpenWebText)...")
    reddit = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    sources.append(("reddit", reddit, "text", 0.2))

    total_chars = 0
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        iterators = {name: iter(ds) for name, ds, _, _ in sources}
        weights = {name: w for name, _, _, w in sources}
        fields = {name: field for name, _, field, _ in sources}

        import random
        names = list(iterators.keys())
        weight_vals = [weights[n] for n in names]

        while total_chars < max_chars:
            name = random.choices(names, weights=weight_vals)[0]
            try:
                item = next(iterators[name])
                text = item[fields[name]].strip()
                if len(text) < 50:
                    continue
                f.write(text + "\n")
                total_chars += len(text)
                written += 1
                if written % 10000 == 0:
                    print(f"  {total_chars/1e6:.1f}M chars written...")
            except StopIteration:
                names.remove(name)
                weight_vals = [weights[n] for n in names]
                if not names:
                    break

    print(f"Done: {total_chars/1e6:.1f}M chars, {written} documents")


# ──────────────────────────────────────────────
# 2. Train BPE tokenizer
# ──────────────────────────────────────────────

def train_tokenizer(
    input_file: str,
    model_prefix: str,
    vocab_size: int = 32000,
):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,  # broad unicode coverage
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        user_defined_symbols=[
            "[INST]", "[/INST]",      # future instruction tuning
            "<think>", "</think>",    # latent reasoning tokens
            "[SYS]", "[/SYS]",
        ],
        num_threads=os.cpu_count(),
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc_cf",  # unicode normalization
    )
    print(f"Tokenizer saved to {model_prefix}.model and {model_prefix}.vocab")


# ──────────────────────────────────────────────
# 3. Wrapper class
# ──────────────────────────────────────────────

class MobileReasonTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.pad_id = self.sp.PieceToId("[PAD]")
        self.unk_id = self.sp.PieceToId("[UNK]")
        self.bos_id = self.sp.PieceToId("[BOS]")
        self.eos_id = self.sp.PieceToId("[EOS]")
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True):
        ids = self.sp.Encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        # Filter special tokens before decode
        filtered = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.sp.Decode(filtered)

    def encode_batch(self, texts: list[str], max_len: int = 2048, pad: bool = True):
        encoded = [self.encode(t) for t in texts]
        if pad:
            max_l = min(max(len(e) for e in encoded), max_len)
            padded = []
            masks = []
            for e in encoded:
                e = e[:max_l]
                mask = [1] * len(e) + [0] * (max_l - len(e))
                e = e + [self.pad_id] * (max_l - len(e))
                padded.append(e)
                masks.append(mask)
            return padded, masks
        return encoded, None

    def save_config(self, path: str):
        config = {
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not SAMPLE_TEXT_PATH.exists():
        print("Step 1: Collecting sample text for tokenizer training...")
        collect_sample_text(SAMPLE_TEXT_PATH)
    else:
        print(f"Sample text already exists at {SAMPLE_TEXT_PATH}")

    print("Step 2: Training BPE tokenizer...")
    train_tokenizer(str(SAMPLE_TEXT_PATH), MODEL_PREFIX)

    print("Step 3: Testing tokenizer...")
    tok = MobileReasonTokenizer(MODEL_PREFIX + ".model")
    tok.save_config(str(TOKENIZER_DIR / "tokenizer_config.json"))

    test_texts = [
        "Hello world, this is a test of our tokenizer!",
        "lol ok fine whatever I guess",
        "The transformer architecture relies on self-attention mechanisms.",
    ]
    for t in test_texts:
        ids = tok.encode(t)
        decoded = tok.decode(ids)
        print(f"\nInput : {t}")
        print(f"Tokens: {ids[:10]}... ({len(ids)} total)")
        print(f"Decode: {decoded}")
