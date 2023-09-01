# Thread 1: Delightful Loss

## Resources

* PyTorch loss functions: https://neptune.ai/blog/pytorch-loss-functions

Example custom loss functions from the above reference: 

```python
def myCustomLoss(my_outputs, my_labels):
    #specifying the batch size
    my_batch_size = my_outputs.size()[0]
    #calculating the log of softmax values           
    my_outputs = F.log_softmax(my_outputs, dim=1)
    #selecting the values that correspond to labels
    my_outputs = my_outputs[range(my_batch_size), my_labels]
    #returning the results
    return -torch.sum(my_outputs)/number_examples
```


You can also create other advanced PyTorch custom loss functions. 

Creating custom loss function with a class definition
Let’s modify the Dice coefficient, which computes the similarity between two samples, to act as a loss function for binary classification problems:

```python
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
```


* Implementation: https://towardsdatascience.com/implementing-custom-loss-functions-in-pytorch-50739f9e0ee1#:~:text=In%20PyTorch%2C%20custom%20loss%20functions,the%20value%20of%20the%20loss.
* Custom criterion: https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568


## Notes

9/1/2023

Observation after 4999 (5000) steps: 
* Training with DLITE: 
    * DLITE train & val loss: 0.4652, 0.4906 (better)
    * CE train & val loss: 2.8259, 2.9784
* Training with CE: 
    * DLITE train & val loss: 0.4662, 0.4957
    * CE train & val loss: 1.6615, 1.8189 (better)

Next: 
* More experiments: 
    * Increase number of iterations: 10k or 20k or until loss/error stop decreasing
        * Please also report `nn.L1Loss()` and `nn.L1Loss()` (two more columns)
        * Results based on DLITE? 
        * Results based on CE? 
    * Mixed iterations and report results
        1. First 5000 on DLITE, and continue 5000 on CE
        2. First 5000 on CE, and continue 5000 on DLITE
* Reading: https://link.springer.com/article/10.1007/s40745-020-00253-5