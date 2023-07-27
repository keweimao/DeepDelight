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


