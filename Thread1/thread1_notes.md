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

### More on Implementation

* Besides the `forward` function (for loss), how can `backward` be implemnted (back propagate derivative)? 
* Derivative of DLITE given below--

Let $x$ be an output probability (after softmax), derivative of Least Information $lit$ can be obtained: 

$ lit'(x) = - \ln x $

Derivative of entropy discount $dh'(x)$ can be obtained by: 


Yes, the rule you're referring to is the **Quotient Rule** for differentiation. Given two differentiable functions \( u(x) \) and \( v(x) \), the derivative of their quotient is given by:

$ \frac{d}{dx} \left( \frac{u}{v} \right) = \frac{u'v - uv'}{v^2} $

$\frac{d}{dx} [u(x) \cdot v(x)] = u'(x) \cdot v(x) + u(x) \cdot v'(x) $


