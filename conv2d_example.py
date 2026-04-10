"""
Working example for torch.nn.functional.conv2d.

Upstream definition (PyTorch main): conv2d is torch.conv2d with a docstring
attached via torch._C._add_docstr — the heavy lifting is in the C++ ATen op.

Educational `conv2d_naive` below shows the same math: for each output pixel,
take a small window of the (padded) input, flatten it, flatten the matching
filter, and their dot product is that output pixel (plus bias).
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def conv2d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """
    Reference 2D convolution with explicit loops + one dot product per output.

    For each (batch n, output channel oc, position oh, ow):
      - Align the kernel over the padded input; read C_in × kH × kW values
        into a patch vector (nested loops over input channel and kernel).
      - Flatten the kernel weight[oc] to the same length.
      - output[n, oc, oh, ow] = dot(patch, w) + bias[oc]

    That dot is what a single-output "neuron" does; sliding the window is the
    rest of the convolution.
    """
    if groups != 1:
        raise NotImplementedError("Educational version: only groups=1.")

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    n, c_in, _, _ = x.shape
    c_out, c_in_w, k_h, k_w = weight.shape
    if c_in != c_in_w:
        raise ValueError(f"input has {c_in} channels but weight expects {c_in_w}")

    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    _, _, h_pad, w_pad = x_pad.shape

    h_out = (h_pad - dil_h * (k_h - 1) - 1) // stride_h + 1
    w_out = (w_pad - dil_w * (k_w - 1) - 1) // stride_w + 1

    out = torch.empty(n, c_out, h_out, w_out, dtype=x.dtype, device=x.device)
    w_flat = weight.reshape(c_out, -1)  # (C_out, C_in * kH * kW)

    for ni in range(n):
        for oc in range(c_out):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch_vals: list[torch.Tensor] = []
                    for ic in range(c_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                ih = oh * stride_h + kh * dil_h
                                iw = ow * stride_w + kw * dil_w
                                patch_vals.append(x_pad[ni, ic, ih, iw])
                    patch = torch.stack(patch_vals)  # (C_in * kH * kW,)
                    # Dot product = (1, D) @ (D, 1) style matmul on vectors
                    acc = torch.dot(patch, w_flat[oc])
                    if bias is not None:
                        acc = acc + bias[oc]
                    out[ni, oc, oh, ow] = acc
    return out


def main() -> None:
    torch.manual_seed(0)

    # NCHW: batch=2, in_channels=3, height=8, width=8
    x = torch.randn(2, 3, 8, 8)
    # out_channels=16, in_channels per group=3, kernel 3x3
    weight = torch.randn(16, 3, 3, 3)
    bias = torch.randn(16)

    y = F.conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)

    print("input:", tuple(x.shape))
    print("weight:", tuple(weight.shape))
    print("output:", tuple(y.shape))

    y_naive = conv2d_naive(x, weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)
    assert torch.allclose(y, y_naive, rtol=1e-5, atol=1e-6)
    print("matches conv2d_naive (loops + dot):", True)


if __name__ == "__main__":
    main()
