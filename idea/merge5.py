import torch.nn.functional as F
import numpy as np
import torch

# setup
# 2, 3, 3, 3 -> 3, 5, 2, 5
kernel_size_1 = np.array([3, 3])
kernel_size_2 = np.array([3, 5])
weight1 = torch.randn(2, 3, 3, 3)
weight2 = torch.randn(5, 3, 3, 5)
bias1 = torch.randn(3)
bias2 = torch.randn(5)

# merge
kernel_size_12 = kernel_size_1 + kernel_size_2 - 1
padding = [kernel_size_2[0] - 1, kernel_size_2[1] - 1]
weight12 = F.conv2d(weight1,
                    weight2.flip(-1, -2),
                    padding=padding).permute(1, 0, 2, 3)
_ = torch.ones(1, 3, *kernel_size_2) * bias1[None, :, None, None]
bias12 = F.conv2d(_, weight2, bias2).flatten()
# verify output
x = torch.randn(1, 2, 9, 9)
w1 = weight1.permute(1, 0, 2, 3)
y = F.conv2d(F.conv2d(x, w1, bias1), weight2, bias2)
y_hat = F.conv2d(x, weight12, bias12)

print(y)
print(y_hat)

assert torch.allclose(y, y_hat)
