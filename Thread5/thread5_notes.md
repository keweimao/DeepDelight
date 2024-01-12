

## Fall 2023 Week 4

Base Implementation of `Transformer`: 
* nanoGPT model: https://github.com/karpathy/nanoGPT/blob/master/model.py
    + Include residual projections and dropout for regularization
    + For now, it's a "fixed" parameter, e.g. constant dropout rate on each layer and on residuals too

TODO: 
* Experiment with variant dropout rates: 1) one dropout rate for an individual layer, 2) a different (greater) dropout rate for residual connections. 
* How does it impact optimization speed (convergence) and optimal results (validation cross-entropy loss in the end)? 
* Read Chapter 3 of: https://cdr.lib.unc.edu/concern/dissertations/3j333275x?locale=en


Related implementations: 
* ResNet (2023): https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
* ResNets (2023): https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5


## Fall 2023 - Break

Pilot Experiments on GPT training (tiny shakespeare): 
* With dropout rate for direct connections at 0.2, vary dropout for residual connections. 
* Results appear that: at 0.2 (range 0.1-0.3?), training (convergence) is optimal? Lower or higher rates degrade the results. 

TODOs: 

1. At (local) optimal (0.2) for residual, vary dropout rates for direct connections, e.g. 0, 0.05, 0.10, ..., 0.5.
2. Identify the optimal range for direct connection dropout, say 0.1-0.2.
3. Experiment with combinations of 1) direct dropout and 2) residual dropout
    * Direct dropout: 0.1, 0.125, 0.15, 0.175, 0.2 (assume optimal range 0.1-0.2)
    * Residual dropout: 0.1, 0.15, 0.2, 0.25, 0.3 (assume optimal range 0.1-0.3)
4. Report moving average Val Loss in the last 5 iterations


## Winter 2024 (Week 1)

1. Use the NanoGPT implementation here: https://github.com/karpathy/nanoGPT
2. Revise code in `https://github.com/karpathy/nanoGPT/blob/master/model.py`, and
3. Program the following two parameters: 

```
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
```

so you can have **different dropout rates** for `self.attn_dropout` and `self.resid_dropout`. 

4. Test the following combinations of parameters (identical dropouts): 

self.attn_dropout = self.resid_dropout = 0
self.attn_dropout = self.resid_dropout = 0.1
self.attn_dropout = self.resid_dropout = 0.2
self.attn_dropout = self.resid_dropout = 0.3
self.attn_dropout = self.resid_dropout = 0.4
self.attn_dropout = self.resid_dropout = 0.5
self.attn_dropout = self.resid_dropout = 0.6

5. Test additional combinations of parameters: 

```python
with self.resid_dropout = 0
    self.attn_dropout = 0
    self.attn_dropout = 0.1
    self.attn_dropout = 0.2
    self.attn_dropout = 0.3
    self.attn_dropout = 0.4
with self.resid_dropout = 0.1
    self.attn_dropout = 0
    self.attn_dropout = 0.1
    self.attn_dropout = 0.2
    self.attn_dropout = 0.3
    self.attn_dropout = 0.4
with self.resid_dropout = 0.2
    self.attn_dropout = 0
    self.attn_dropout = 0.1
    self.attn_dropout = 0.2
    self.attn_dropout = 0.3
    self.attn_dropout = 0.4
with self.resid_dropout = 0.3
    self.attn_dropout = 0
    self.attn_dropout = 0.1
    self.attn_dropout = 0.2
    self.attn_dropout = 0.3
    self.attn_dropout = 0.4
with self.resid_dropout = 0.4
    self.attn_dropout = 0
    self.attn_dropout = 0.1
    self.attn_dropout = 0.2
    self.attn_dropout = 0.3
    self.attn_dropout = 0.4
```

6. After an initial test, program this with a loop in a `.py` implementation so you can run it in a batch mode. 
7. Save all results to CSV file in the following format: 

```
attn_dropout    resid_dropout   iterations  train_loss  val_loss
0.0             0.0             5000        2.0         2.2
0.1             0.0             5000        1.9         2.0
..              ..              ..          ..          ..
```