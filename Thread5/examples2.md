# Examples for Residual and Dropouts in nanoGPT

Based on nanoGPT implementation: 
https://github.com/karpathy/nanoGPT/blob/master/model.py

With a focus on the `Block` class, here are a few examples on 
1. how to implement residual connections; 
2. followed by dropout on the residuals at the end. 

## 1. No Skip Connections First

An example with **NO** residual/skip connections: 

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        y = self.attn(self.ln_1(x))
        z = self.mlp(self.ln_2(y))
        return z
```

## 2. Rewrite Code for Better Clarity

I'd rather restructure the code to: 

```python
    def forward(self, x):
        x = self.ln_1(x)    # layer norm
        y = self.attn(x)
        y = self.ln_2(y)    # layer norm
        z = self.mlp(y)
        return z
```

If we ignore `ln_1` and `ln_2` layer norms, here we have 2 layers of real transformations/connections: 

```
     ------           -----
x --| attn |--> y -- | mlp |--> z
     ------           -----
```

## 3. Skip-1-layer Connections

To implement skip connections, we can revise the forward function to: 

```python
    def forward(self, x):
        x = self.ln_1(x)        
        y = x + self.attn(x)    # add skip 1 layer from x
        y = self.ln_2(y)        
        z = y + self.mlp(y)     # add skip 1 layer from y
        return z
```

Here we have two separate skip-1-layer residuals, visualized below: 

```
|--------------    |--------------
|             ||   |             ||
|    ------   \/   |    -----    \/
x --| attn |-----> y --| mlp |-------> z
     ------             -----
```

## 4. Skip-2-layer Connections

To skip 2 layers, we add first input `x` to the final output for `z`: 

```python
    def forward(self, x):
        x = self.ln_1(x)        
        y = x + self.attn(x)        # add skip 1 layer from x
        y = self.ln_2(y)        
        z = x + y + self.mlp(y)     # add skip 1 layer from y PLUS x, which skips two. 
        return z
```


## 5. Dropout on Skip Connections

Now putting everything togehter with the dropout: 

```python
class Block(nn.Module):
 
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.dropout1 = nn.Dropout(0.1)  # dropout1 for skip-1-layer connections
        self.dropout2 = nn.Dropout(0.2)  # dropout2 for skip-2-layer connections
 
    def forward(self, x):
        x = self.ln_1(x)        
        y = self.dropout1(x) + self.attn(x)                     # x skips 1 layer
        y = self.ln_2(y)        
        z = self.dropout2(x) + self.dropout1(y) + self.mlp(y)   # x skips 2 layers and y skips 1 layer
        return z
```
