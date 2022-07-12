import torch.nn as nn

class Waug(nn.Module):
    def __init__(self,config,dropout: float = 0.1):
        super(Waug, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.project = config.project
        if self.project:
            self.q_project = nn.Linear(config.hidden_dim,config.hidden_dim)
            self.k_project = nn.Linear(config.hidden_dim,config.hidden_dim)
            self.v_project = nn.Linear(config.hidden_dim,config.hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim = config.hidden_dim, 
            num_heads = config.waug_nheads, 
            dropout=0.1, 
            batch_first=True)
    def _sa_block(self, x):
        if self.project:
            x = self.self_attn(
                self.q_project(x), 
                self.k_project(x), 
                self.v_project(x),
                need_weights=False)[0]
        else:
            x = self.self_attn(x, x, x,need_weights=False)[0]
        return self.dropout(x)
    def forward(self,x):
        x = self.norm(x + self._sa_block(x))
        return x