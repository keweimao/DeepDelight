# Thread 1: Delightful Loss

## Resources
* PyTorch loss functions: https://neptune.ai/blog/pytorch-loss-functions
* Implementation: https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1#:~:text=In%20PyTorch%2C%20custom%20loss%20functions,the%20value%20of%20the%20loss.
* Custom criterion: https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568


## Notes and TODOs

### 9/1/2023

Observation after 4999 (5000) steps: 
* Training with DLITE: 
    * DLITE train & val loss: 0.4652, 0.4906 (better)
    * CE train & val loss: 2.8259, 2.9784
* Training with CE: 
    * DLITE train & val loss: 0.4662, 0.4957
    * CE train & val loss: 1.6615, 1.8189 (better)

### Baseline Comparison (2023 Fall Week 2)

* More experiments: 
    * Increase number of iterations: 10k or 20k or until loss/error stop decreasing
        * Please also report `nn.L1Loss()` and `nn.L1Loss()` (two more columns)
        * Results based on DLITE? 
        * Results based on CE? 
    * Mixed iterations and report results
        1. First 5000 on DLITE, and continue 5000 on CE
        2. First 5000 on CE, and continue 5000 on DLITE
* Reading: https://link.springer.com/article/10.1007/s40745-020-00253-5
  * Focus on section 3.1 Loss Functions for Classification
  * Compare logarithmic function fig 1.c to DLITE fig. 2.a
* Refer to this for math notation and examples of softmax plus cross-entropy: https://gombru.github.io/2018/05/23/cross_entropy_loss/

### Review and Additional Implementations (2023 Fall Week 7)

1. Train with DLITE (5000x) then CE (5000x): 
    * These figures are interesting, and suggest something unusual with DLITE: [DLITE Over Time](figures/dl5k_ce5k_dlite_over_time.png), [KL Over Time](figures/dl5k_ce5k_kl_over_time.png), [CE Over Time](figures/dl5k_ce5k_ce_over_time.png)
    * That is, it is either an issue of the DLITE theory OR a bug in the implementation? 
2. One way to check and test (for a potential bug) is to **implement KL from scratch**: 
    * Instead of calling `F.kl_div()` in the `torch.nn` module
    * Implement our own `KLLoss()` based on simple revision (simplification) of `DLITELoss()`
    * The following is example code: 

Based on the sample zero-value masking, we can compute `KL` in the `forward` function: 

```python
def forward(self, logits, targets):
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)

        # One-hot encode the targets to get true probabilities
        true_probs = F.one_hot(targets, num_classes=probs.size(-1)).float()

        # Masks for non-zero elements of probs and true_probs
        mask_probs = probs > 0
        mask_true_probs = true_probs > 0

        # Calculate g function for non-zero elements using the mask
        kl_values = torch.zeros_like(probs)
        kl_values[mask_true_probs] = true_probs[mask_true_probs] * torch.log(true_probs[mask_true_probs]/probs[mask_true_probs])

        # Sum over all classes and average over the batch size
        loss = kl_values.sum(dim=-1).mean()

        return loss
```

The `KL` formula can be found at: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence, where $P$ is true probability distribution and $Q$ is the estimate (prediction). 

3. The result of the above `KL` implementation will confirm whether the probabilities have been used correctly: 
    * If this KL works well, the impelementation is good and the issue is with DLITE theory itself. 
    * If this doesn't, then review code and fix the issue until achieve the same good result with THIS `KL`. 
4. After the above correction, implement and test one more method $DLITE^{1/3}$, which is a metric distance: 
    * Use the final DLITE implementation, and compute its **Cube Root** as `loss`. 


### More on Implementation

* Besides the `forward` function (for loss), how can `backward` be implemnted (back propagate derivative)? 
* Derivative of DLITE given below--

Let $x$ be an output probability (after softmax), derivative of Least Information $lit$ can be obtained: 

$ lit'(x) = - \ln x $

Derivative of entropy discount $dh'(x)$ can be obtained by: 


Yes, the rule you're referring to is the **Quotient Rule** for differentiation. Given two differentiable functions \( u(x) \) and \( v(x) \), the derivative of their quotient is given by:

$ \frac{d}{dx} \left( \frac{u}{v} \right) = \frac{u'v - uv'}{v^2} $

$\frac{d}{dx} [u(x) \cdot v(x)] = u'(x) \cdot v(x) + u(x) \cdot v'(x) $


