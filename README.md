# conv2d: L∞ bounds through convolution

Propagates axis-aligned **L∞** input boxes through `conv2d` (interval / IBP: `mid = conv(mean,W,b)`, `rad = conv(std,|W|,0)`, `L_y = mid - rad`, `U_y = mid + rad`). **`Conv2dWedge`** pulls output-side kernels back to the input by **composing** conv kernels (`accumulate_weight`); **`to_bound_tensor`** applies the same mid/|W|/std rule on the composed weights.

**Modules:** `tool/bound_linf.py` (box + `conv`), `bound_tensor.py`, `conv2d_wedge.py` (wedge), `wedge.py` (linear), `main.py` (demo).

`compose_weight` assumes stride 1, dilation 1, groups 1, square kernels; keep `F_conv` consistent. Boxes are tight **per output coordinate** for this linear layer, not the full zonotope image.

## Minimal example

```python
import torch, torch.nn.functional as F
from functools import partial
import tool

L = torch.zeros(1, 3, 8, 8)
U = L + 0.1
x = tool.BoundTensor(L, tool.BoundLinf(L, U))
weight, bias = torch.randn(2, 3, 3, 3), torch.randn(2)
F_conv = partial(F.conv2d, stride=1, padding=1, dilation=1, groups=1)

p_y = x.perturbation.conv(F_conv, x, weight, bias)
y_ref = tool.BoundTensor(p_y.L, p_y)

C = weight.shape[0]
W_I = torch.zeros(C, C, 1, 1)
W_I[torch.arange(C), torch.arange(C), 0, 0] = 1.0
b_I = torch.zeros(C, dtype=weight.dtype, device=weight.device)
wedge = tool.Conv2dWedge(W_I, b_I, W_I, b_I).accumulate_weight(weight, bias)
y_wedge = wedge.to_bound_tensor(F_conv, x)

Lr, Ur = y_ref.concretize()
Lw, Uw = y_wedge.concretize()
assert torch.allclose(Lr, Lw) and torch.allclose(Ur, Uw)
```

```bash
cd conv2d && python main.py
```
