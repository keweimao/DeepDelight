# Related Residual and Dropout Code in nanoGPT

https://github.com/karpathy/nanoGPT/blob/master/model.py

## Residual (Skip) Connections

An example residual (skip) implementation: 

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

Droput for attention and residual connections in : 

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        ...
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        ...
        self.dropout = config.dropout
    
    def forward(self, x):
        ...
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```


