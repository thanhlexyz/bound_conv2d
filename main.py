import torch.nn.functional as F
import torch

x = torch.randn(2, 3, 32, 32)

weight = torch.randn(16, 3, 3, 3)
bias = torch.randn(16)

y = F.conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)

print("input:", x.shape)   # [2, 3, 32, 32]
print("output:", y.shape)  # [2, 16, 32, 32] — same H,W because padding=1 for 3×3
