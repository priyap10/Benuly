"""
data/prepare.py

Character-level tokenizer. Reads your raw text file, builds a vocabulary
of unique characters, encodes the whole file as integers, and saves
train/val splits as tensors, plus the vocab mapping (needed later to
turn generated integers back into text).
"""

import os
import json
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RAW_PATH = config.DATA_PATH
OUT_DIR = "data"


def main():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters from {RAW_PATH}")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size} unique characters")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(config.TRAIN_SPLIT * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    torch.save(train_data, os.path.join(OUT_DIR, "train.pt"))
    torch.save(val_data, os.path.join(OUT_DIR, "val.pt"))

    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos, "vocab_size": vocab_size}, f, ensure_ascii=False, indent=2)

    print("Saved data/train.pt, data/val.pt, data/vocab.json")


if __name__ == "__main__":
    main()