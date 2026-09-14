
import argparse
import json
import torch

import config
from model.gpt import GPT

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
    model_cfg = checkpoint["config"]

    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        block_size=model_cfg["block_size"],
        dropout=model_cfg["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_vocab():
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    itos = {int(k): v for k, v in vocab["itos"].items()}
    stoi = vocab["stoi"]
    return stoi, itos


def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS)
    args = parser.parse_args()

    model = load_model()
    stoi, itos = load_vocab()

    if args.prompt:
        context_ids = encode(args.prompt, stoi)
        idx = torch.tensor([context_ids], dtype=torch.long, device=device)
    else:
        
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(idx, max_new_tokens=args.max_new_tokens)
    text = decode(generated[0].tolist(), itos)

    print(text)


if __name__ == "__main__":
    main()