

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)          
        self.lm_head = nn.Linear(d_model, vocab_size)  
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
       
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                              
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb                                            

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        if targets is None:
            return logits, None

       
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
       
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] 
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat([idx, next_token], dim=1)
        return idx