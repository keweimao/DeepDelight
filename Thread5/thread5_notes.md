

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


