"""
config.py

All hyperparameters live here so every other file imports from one place.
Starting small on purpose — a tiny model trains fast on CPU and is much
easier to debug. Scale these up once everything works end-to-end.
"""

# --- Data ---
DATA_PATH = "data/raw/model_training.txt"      # point this at your renamed file
BLOCK_SIZE = 128                  # context length (tokens per training example)
TRAIN_SPLIT = 0.9                     # fraction of data used for training vs validation

# --- Model architecture ---
D_MODEL = 192        # embedding dimension
N_HEADS = 6          # number of attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS = 6          # number of transformer blocks stacked
DROPOUT = 0.1

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_ITERS = 6000
EVAL_INTERVAL = 300   # how often to check validation loss
EVAL_ITERS = 50         # how many batches to average for eval loss

# --- Generation ---
MAX_NEW_TOKENS = 300

# --- Misc ---
SEED = 1337
CHECKPOINT_PATH = "checkpoints/benuly.pt"