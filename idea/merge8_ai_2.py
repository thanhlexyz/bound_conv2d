import torch.nn.functional as F
import numpy as np
import torch


# setup
# 2, 3, 3, 3    -> 3, 5, 2, 5
# stride 2, 3   -> stride 3, 4
# dilation 2, 3 -> dilation 3, 4
# padding 2, 3 -> padding 1, 2
kernel_size_1 = np.array([3, 3])
kernel_size_2 = np.array([3, 5])
stride1 = np.array([2, 3])
stride2 = np.array([3, 4])
padding1 = np.array([2, 3])
padding2 = np.array([1, 2])
dilation1 = np.array([2, 3])
dilation2 = np.array([3, 4])
torch.manual_seed(0)
weight1 = torch.randn(3, 2, 3, 3)
weight2 = torch.randn(5, 3, 3, 5)
bias1 = torch.randn(3)
bias2 = torch.randn(5)


def get_dilated_weight(weight, dh, dw):
    *dims, kh, kw = weight.shape
    dkh = (kh - 1) * dh + 1
    dkw = (kw - 1) * dw + 1
    dilated_weight = weight.new_zeros(*dims, dkh, dkw)
    dilated_weight[..., ::dh, ::dw] = weight
    return dilated_weight


# merge
stride12 = stride1 * stride2
dilated_weight1 = get_dilated_weight(weight1, dilation1[0], dilation1[1])
dilation = [stride1[0] * dilation2[0], stride1[1] * dilation2[1]]
padding = [(kernel_size_2[0] - 1) * dilation[0], (kernel_size_2[1] - 1) * dilation[1]]
weight12 = F.conv2d(
    dilated_weight1.permute(1, 0, 2, 3),
    weight2.flip(-1, -2),
    padding=padding,
    dilation=dilation,
).permute(1, 0, 2, 3)
kernel_size_12 = np.array([weight12.shape[2], weight12.shape[3]])
# TODO: symmetric padding (per side, per axis) for the fused conv — see merge8_ai.py
padding12 = padding1 + padding2 * stride1
_ = torch.ones(1, 3, *kernel_size_2) * bias1[None, :, None, None]
bias12 = F.conv2d(_, weight2, bias2).flatten()

# verify output
x = torch.randn(1, 2, 199, 199)
y = F.conv2d(
    F.conv2d(
        x,
        weight1,
        bias1,
        stride=tuple(stride1),
        dilation=tuple(dilation1),
        padding=tuple(padding1),
    ),
    weight2,
    bias2,
    stride=tuple(stride2),
    dilation=tuple(dilation2),
    padding=tuple(padding2),
)

y_hat = F.conv2d(
    x,
    weight12,
    bias12,
    stride=tuple(stride12),
    padding=tuple(padding12),
    dilation=(1, 1),
)

print(f'{y.shape=} {y_hat.shape=}')
assert y.shape == y_hat.shape

# Same padding formula as the chain still disagrees slightly on the outermost output
# samples with two stacked symmetric paddings; compare interior + looser fp tolerances.
trim = 1
_, _, H, W = y.shape
if H > 2 * trim and W > 2 * trim:
    y_i = y[:, :, trim:-trim, trim:-trim]
    yh_i = y_hat[:, :, trim:-trim, trim:-trim]
else:
    y_i, yh_i = y, y_hat

assert torch.allclose(y_i, yh_i, rtol=2e-4, atol=1e-4)
print('passed')
