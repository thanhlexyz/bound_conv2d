# Bounded convolution (L∞ box propagation)

Small PyTorch utilities to propagate **axis-aligned L∞ bounds** through a **convolutional layer** and to **pull back** output-side linear operators to the input using a **wedge** (composed kernels).

## What is in this repo

| Piece | Role |
|--------|------|
| `tool/bound_linf.py` | `BoundLinf(L, U)` stores lower/upper tensors. Implements standard **interval / IBP-style** conv: midpoint and radius with `abs(weight)`. |
| `tool/bound_tensor.py` | `BoundTensor` wraps a tensor plus a perturbation object (`BoundLinf`). |
| `tool/conv2d_wedge.py` | `Conv2dWedge(W_L, b_L, W_U, b_U)` holds two-sided conv kernels and biases. `accumulate_weight(weight, bias)` **composes** the wedge with a conv layer (kernel composition via `conv_transpose2d`). `to_bound_tensor(F_conv, x)` **concretizes** output `[L_y, U_y]` from an input `BoundTensor`. |
| `tool/wedge.py` | Linear (matrix) wedge analogue (not conv-specific). |
| `main.py` | Demo: random box + conv + wedge + sampling checks. |

## Bound on one conv layer (minimal idea)

For input `x` in a box `L ≤ x ≤ U` (elementwise), write `mean = (L+U)/2` and `std = (U-L)/2`. For one conv with weight `W` and bias `b`, the **tightest axis-aligned L∞ output box** under this linear map uses the usual mid ± |W| rule:

- `mid_y = conv(mean, W, b)`
- `rad_y = conv(std, |W|, 0)` (no bias on the radius term)
- `L_y = mid_y - rad_y`, `U_y = mid_y + rad_y`

That is what `BoundLinf.conv(F_conv, x, weight, bias)` implements when `F_conv` is `F.conv2d` (or a partial with fixed stride, padding, dilation, groups).

## Wedge (why compose kernels?)

You start from an **identity** wedge at the conv output: per-channel 1×1 delta kernels so the bound at the output is expressed in output feature space. **`accumulate_weight(weight, bias)`** folds the conv into the wedge so the same output bound can be written as a **single equivalent conv** from the **input** of that layer: composed kernels `W'` and bias `b'` such that, in the interior, chaining `conv(·, weight) + bias` with the wedge matches `conv(·, W') + b'` (composition semantics; `compose_weight` assumes stride 1, dilation 1, groups 1, square kernels).

Then `to_bound_tensor(F_conv, x)` applies the mid / |W| / std recipe using the **composed** `W_L, W_U` and biases.

## Minimal example

```python
import torch
import torch.nn.functional as F
from functools import partial

import tool

# Input L∞ box
L = torch.zeros(1, 3, 8, 8)
U = L + 0.1
x = tool.BoundTensor(L, tool.BoundLinf(L, U))

# One conv: 3 -> 2 channels, 3x3 kernel
weight = torch.randn(2, 3, 3, 3)
bias = torch.randn(2)
F_conv = partial(F.conv2d, stride=1, padding=1, dilation=1, groups=1)

# Reference output box (direct interval conv)
p_y = x.perturbation.conv(F_conv, x, weight, bias)
y_ref = tool.BoundTensor(p_y.L, p_y)

# Same box via wedge: identity at output, then accumulate this conv
C = weight.shape[0]
W_I = torch.zeros(C, C, 1, 1)
W_I[torch.arange(C), torch.arange(C), 0, 0] = 1.0
b_I = torch.zeros(C, dtype=weight.dtype, device=weight.device)
wedge = tool.Conv2dWedge(W_I, b_I, W_I, b_I).accumulate_weight(weight, bias)
y_wedge = wedge.to_bound_tensor(F_conv, x)

assert torch.allclose(y_ref.concretize()[0], y_wedge.concretize()[0])
assert torch.allclose(y_ref.concretize()[1], y_wedge.concretize()[1])
```

## Running the demo

```bash
cd conv2d
python main.py
```

## Caveats

- **`compose_weight`** is documented for **stride 1, dilation 1, groups 1**, **square** kernels; mismatch with arbitrary `F_conv` stride/padding/dilation in `to_bound_tensor` can make the composed wedge inconsistent with the concrete conv you evaluate on samples.
- Output boxes are **sound** for `x ∈ [L, U]`; they are **not** the exact reachable set (that is generally a zonotope), only the **tight axis-aligned** hull per coordinate for this linear layer.
