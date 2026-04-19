import torch.nn.functional as F
import numpy as np
import torch

# setup
# 2, 3, 3, 3 -> 3, 5, 2, 5
kernel_size_1 = np.array([3, 3])
kernel_size_2 = np.array([3, 5])
conv1 = torch.nn.Conv2d(2, 3, kernel_size_1, bias=False)
conv2 = torch.nn.Conv2d(3, 5, kernel_size_2, bias=False)

# merge
kernel_size_12 = kernel_size_1 + kernel_size_2 - 1
conv12 = torch.nn.Conv2d(2, 5, kernel_size_12, bias=False)
padding = [kernel_size_2[0] - 1, kernel_size_2[1] - 1]
conv12.weight.data = F.conv2d(conv1.weight.data.permute(1, 0, 2, 3),
                              conv2.weight.data.flip(-1, -2),
                              padding=padding).permute(1, 0, 2, 3)


# verify output
x = torch.randn(1, 2, 9, 9)
y = conv2(conv1(x))
y_hat = conv12(x)

print(y)
print(y_hat)

assert torch.allclose(y, y_hat)
print('passed')
