import numpy as np
import torch

# setup
# 1x1 channel 3x3 kernel -> 1x1 channel 1x1 kernel
conv1 = torch.nn.Conv2d(1, 1, 3, bias=False)
conv2 = torch.nn.Conv2d(1, 1, 1, bias=False)
conv12 = torch.nn.Conv2d(1, 1, 3, bias=False)

# merge
conv12.weight.data = conv1.weight.data * conv2.weight.data

# verify output
x = torch.randn(1, 1, 6, 6)
y = conv2(conv1(x))
y_hat = conv12(x)

print(y)
print(y_hat)

assert torch.allclose(y, y_hat)
print('passed')
