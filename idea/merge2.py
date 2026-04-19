import numpy as np
import torch

# setup
# 2x3 channel 3x3 kernel -> 3x1 channel 1x1 kernel
conv1 = torch.nn.Conv2d(2, 3, 3, bias=False)
conv2 = torch.nn.Conv2d(3, 1, 1, bias=False)
conv12 = torch.nn.Conv2d(2, 1, 3, bias=False)

# merge
# note that, weight.data.shape = [channel out, channel in, kernel height, kernel width]
conv12.weight.data = (conv1.weight.data * conv2.weight.data.permute(1, 0, 2, 3)).sum(dim=0, keepdim=True)

# verify output
x = torch.randn(1, 2, 6, 6)
y = conv2(conv1(x))
y_hat = conv12(x)

print(y)
print(y_hat)

assert torch.allclose(y, y_hat)
print('passed')
