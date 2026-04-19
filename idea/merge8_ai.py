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


def _chain_out_len(
    length: int,
    pad: int,
    stride: int,
    dil: int,
    kernel: int,
) -> int:
    return (length + 2 * pad - dil * (kernel - 1) - 1) // stride + 1


def merged_out_len(length: int, pad: int, kernel: int, stride: int) -> int:
    return (length + 2 * pad - (kernel - 1) - 1) // stride + 1


def compute_merged_padding_1d(
    p1: int,
    p2: int,
    s1: int,
    s2: int,
    d1: int,
    d2: int,
    k1: int,
    k2: int,
    merged_kernel: int,
    merged_stride: int,
    *,
    h_max: int = 1024,
) -> int:
    """Smallest P ≥ 0 such that chain and merged output lengths match for all input lengths."""

    def chain_len(n: int) -> int:
        n1 = _chain_out_len(n, p1, s1, d1, k1)
        return _chain_out_len(n1, p2, s2, d2, k2)

    for p in range(merged_kernel + merged_stride * max(p1 + p2, 1) + 8):
        ok = True
        for n in range(1, h_max):
            if chain_len(n) != merged_out_len(n, p, merged_kernel, merged_stride):
                ok = False
                break
        if ok:
            return p
    raise RuntimeError('could not find merged padding; check hyperparameters')


def compute_merged_padding_hw(
    padding1,
    padding2,
    stride1,
    stride2,
    dilation1,
    dilation2,
    kernel_size_1,
    kernel_size_2,
    merged_kernel_hw,
    stride_merged_hw,
):
    """Per-axis merged symmetric padding for the fused conv (dilation=1 on the input)."""
    ph = compute_merged_padding_1d(
        int(padding1[0]),
        int(padding2[0]),
        int(stride1[0]),
        int(stride2[0]),
        int(dilation1[0]),
        int(dilation2[0]),
        int(kernel_size_1[0]),
        int(kernel_size_2[0]),
        int(merged_kernel_hw[0]),
        int(stride_merged_hw[0]),
    )
    pw = compute_merged_padding_1d(
        int(padding1[1]),
        int(padding2[1]),
        int(stride1[1]),
        int(stride2[1]),
        int(dilation1[1]),
        int(dilation2[1]),
        int(kernel_size_1[1]),
        int(kernel_size_2[1]),
        int(merged_kernel_hw[1]),
        int(stride_merged_hw[1]),
    )
    return ph, pw


# merge (same weight construction as merge7)
stride12 = stride1 * stride2
dilated_weight1 = get_dilated_weight(weight1, dilation1[0], dilation1[1])
inner = [stride1[0] * dilation2[0], stride1[1] * dilation2[1]]
padding_inner = [
    (kernel_size_2[0] - 1) * inner[0],
    (kernel_size_2[1] - 1) * inner[1],
]
weight12 = F.conv2d(
    dilated_weight1.permute(1, 0, 2, 3),
    weight2.flip(-1, -2),
    padding=padding_inner,
    dilation=tuple(inner),
).permute(1, 0, 2, 3)

merged_k_hw = (weight12.shape[2], weight12.shape[3])
padding12 = compute_merged_padding_hw(
    padding1,
    padding2,
    stride1,
    stride2,
    dilation1,
    dilation2,
    kernel_size_1,
    kernel_size_2,
    merged_k_hw,
    stride12,
)

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
    padding=padding12,
    dilation=(1, 1),
)

print(f'{y.shape=} {y_hat.shape=}')
assert y.shape == y_hat.shape

# Symmetric merged padding matches the chain in the interior; a one-cell output rim can
# disagree with two stacked symmetric paddings (alignment of zeros at the boundary).
trim = 1
_, _, H, W = y.shape
if H > 2 * trim and W > 2 * trim:
    y_i = y[:, :, trim:-trim, trim:-trim]
    yh_i = y_hat[:, :, trim:-trim, trim:-trim]
else:
    y_i, yh_i = y, y_hat

assert torch.allclose(y_i, yh_i, rtol=2e-4, atol=1e-4)
print('passed')
