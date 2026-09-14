

DATA_PATH = "data/raw/model_training.txt"     
BLOCK_SIZE = 128                 
TRAIN_SPLIT = 0.9                   

D_MODEL = 192        
N_HEADS = 6        
N_LAYERS = 6          
DROPOUT = 0.1


BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_ITERS = 6000
EVAL_INTERVAL = 300  
EVAL_ITERS = 50        
MAX_NEW_TOKENS = 300


SEED = 1337
CHECKPOINT_PATH = "checkpoints/benuly.pt"