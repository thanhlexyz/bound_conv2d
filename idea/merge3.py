import numpy as np
import torch

# setup
# 2x3 channel 3x3 kernel -> 3x5 channel 1x1 kernel
conv1 = torch.nn.Conv2d(2, 3, 3, bias=False)
conv2 = torch.nn.Conv2d(3, 5, 1, bias=False)
conv12 = torch.nn.Conv2d(2, 5, 3, bias=False)

# merge
# note that, weight.data.shape = [channel out, channel in, kernel height, kernel width]
conv12.weight.data = (conv1.weight.data.unsqueeze(0) * conv2.weight.data.unsqueeze(-1)).sum(dim=1)
# 1 2 3 3 3 x 1 1 3 1 1
# 1 C1 C0 Kh1 Kw1 x C2 C1 Kh2 Kw2 1 (since Kh2 Kw2 = 1 can broadcast)
# -> 1 2 3 3 3 .sum(1) -> 1 3 3 3 (bs, C2, Kh2, Kw2)

# verify output
x = torch.randn(1, 2, 6, 6)
y = conv2(conv1(x))
y_hat = conv12(x)

print(y)
print(y_hat)

assert torch.allclose(y, y_hat)
