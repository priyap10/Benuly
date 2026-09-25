# Benuly

A small GPT-style language model built from scratch in PyTorch, no HuggingFace, no pre-built training loop. Built the attention mechanism, the transformer blocks, and the training loop, to actually learn how transformers work rather than just using one.

## What it does

Trains a character-level language model on a text file and generates new text in a similar style.

## Files

- `model/attention.py` — self-attention
- `model/transformer_block.py` — one transformer block (attention + feed-forward + residuals)
- `model/gpt.py` — the full model + text generation
- `data/prepare.py` — turns raw text into training data
- `train.py` — the training loop
- `generate.py` — generates text from the trained model

## Model

- Parameters: 2,799,643 
- vocab : 283 unique characters 
- train / val loss: started with 5.6678 / 5.6703
- train / val loss: final ~ 1.18 / 1.57

## How to run

python data/prepare.py

python train.py

python generate.py --prompt "hey"


*might generate half formed sentences but fully formed words for sure



## What I learned

- How self-attention actually works (Query/Key/Value, softmax, causal masking)
- Why residual connections matter in deep networks
- The difference between train loss and validation loss




