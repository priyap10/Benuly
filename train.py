

import os
import json
import torch

import config
from model.gpt import GPT

torch.manual_seed(config.SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_data():
    train_data = torch.load("data/train.pt")
    val_data = torch.load("data/val.pt")
    with open("data/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return train_data, val_data, vocab["vocab_size"]


def get_batch(split, train_data, val_data):
    
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i + config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + config.BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            x, y = get_batch(split, train_data, val_data)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    train_data, val_data, vocab_size = load_data()
    print(f"Vocab size: {vocab_size}")

    model = GPT(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        block_size=config.BLOCK_SIZE,
        dropout=config.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    for iter in range(config.MAX_ITERS):
        if iter % config.EVAL_INTERVAL == 0 or iter == config.MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "config": {
            "d_model": config.D_MODEL,
            "n_heads": config.N_HEADS,
            "n_layers": config.N_LAYERS,
            "block_size": config.BLOCK_SIZE,
            "dropout": config.DROPOUT,
        },
    }, config.CHECKPOINT_PATH)
    print(f"Saved checkpoint to {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()