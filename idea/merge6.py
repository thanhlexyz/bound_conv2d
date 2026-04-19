import torch.nn.functional as F
import numpy as np
import torch

# setup
# 2, 3, 3, 3 -> 3, 5, 2, 5
# stride 2, 3 -> stride 3, 4
kernel_size_1 = np.array([3, 3])
kernel_size_2 = np.array([3, 5])
stride1 = np.array([2, 3])
stride2 = np.array([3, 4])
weight1 = torch.randn(3, 2, 3, 3)
weight2 = torch.randn(5, 3, 3, 5)
bias1 = torch.randn(3)
bias2 = torch.randn(5)

# merge
kernel_size_12 = kernel_size_1 + kernel_size_2 - 1
stride12 = stride1 * stride2
padding = [(kernel_size_2[0] - 1) * stride1[0], (kernel_size_2[1] - 1) * stride1[1]]
weight12 = F.conv2d(weight1.permute(1, 0, 2, 3),
                    weight2.flip(-1, -2),
                    padding=padding,
                    dilation=tuple(stride1)).permute(1, 0, 2, 3)
_ = torch.ones(1, 3, *kernel_size_2) * bias1[None, :, None, None]
bias12 = F.conv2d(_, weight2, bias2).flatten()
# verify output
x = torch.randn(1, 2, 55, 55)
y = F.conv2d(F.conv2d(x, weight1, bias1, stride=tuple(stride1)), weight2, bias2, stride=tuple(stride2))
y_hat = F.conv2d(x, weight12, bias12, stride=tuple(stride12))
assert torch.allclose(y, y_hat, rtol=1e-4, atol=1e-5)
print('passed')
