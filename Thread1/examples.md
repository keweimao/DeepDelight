

## Forward

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



## Backward

```python
import torch
from torch.autograd import Function

class SquareFunction(Function):

    @staticmethod
    def forward(ctx, input):
        # Save input for backward pass
        ctx.save_for_backward(input)
        return input * input

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved input
        input, = ctx.saved_tensors
        # Compute gradient
        grad_input = 2 * input * grad_output
        return grad_input
```

To use the custom function:

```python
input_tensor = torch.tensor([2.0, 3.0], requires_grad=True)
square = SquareFunction.apply
output_tensor = square(input_tensor)
output_tensor.backward(torch.ones_like(output_tensor))
print(input_tensor.grad)  # This should print "tensor([4., 6.])" since d(x^2)/dx = 2x
```

