# Conv2d bounds with a wedge

This repo shows a small setup for **interval bounds** on a 2D convolution layer,
implemented via a paired-kernel structure called a **wedge**.

## Idea

A `Conv2dWedge` holds four tensors `(W_L, b_L, W_U, b_U)` and a dict `attr`
(stride, padding, dilation, groups).  It encodes a channel-wise bound on some
activation `z` in terms of a preceding activation `y`:

```
F.conv2d(y, W_L, b_L, **attr) <= z <= F.conv2d(y, W_U, b_U, **attr)
```

Shape of `W_L` / `W_U`: `(batch, start_channel, end_channel, kH, kW)`.  
Shape of `b_L` / `b_U`: `(batch, start_channel)`.

## Workflow

```
input x                       output z
  └──── conv(weight, bias, attr) ────┘
```

1. **`Conv2dWedge.init_identity(z_sample)`** — build an identity wedge at the
   output (each output channel bounded by itself, `1×1` kernel, zero bias).

2. **`wedge.accumulate_layer(weight, bias, attr)`** — fold one `conv2d` layer
   into the wedge.  The new wedge maps the layer *input* directly to the bound:

   - kernels are fused with `F.conv2d` (treating `weight.T` as input and the
     old wedge kernels as filters);
   - `attr` values compose as: `stride` and `dilation` multiply element-wise,
     `padding` adds element-wise;
   - `bias` contributes via `einsum('bsehw, be -> bs', W_*, bias)`.

3. **`wedge.to_bound_tensor(x)`** — concretize the wedge on a `BoundTensor`
   input.  Applies the standard IBP rule (same as `BoundInterval.conv`):

   ```
   L = F.conv2d(mean, W_L, b_L, **attr) - F.conv2d(std, |W_L|, None, **attr)
   U = F.conv2d(mean, W_U, b_U, **attr) + F.conv2d(std, |W_U|, None, **attr)
   ```

   where `mean = (L_in + U_in) / 2` and `std = (U_in - L_in) / 2`.

## Key classes

| Class | File | Role |
|---|---|---|
| `Conv2dWedge` | `tool/conv2d_wedge.py` | Paired-kernel bound for one conv layer |
| `Wedge` | `tool/wedge.py` | Base class; linear (fully-connected) version |
| `BoundTensor` | `tool/bound_tensor.py` | Wraps a concrete tensor + a bound object |
| `BoundInterval` | `tool/bound_interval.py` | `ℓ∞` interval `x0 ± eps` |

## Demo

```bash
python main.py
```

```
y0.shape=torch.Size([1, 2, 5, 5])
wedge_out=Conv2dWedge(shape=torch.Size([1, 2, 2, 1, 1]))
wedge_in=Conv2dWedge(shape=torch.Size([1, 2, 4, 3, 3]))
BoundTensor(Interval(-1.2567±0.3864, shape=torch.Size([1, 2, 5, 5])))
input in interval correct/n_sample=1.0
input not in interval correct/total=0.206
```

*input in interval* — all 1000 samples from the declared input box land inside
the output bound (soundness check).

*input not in interval* — ~20 % of samples drawn from a wider box (3× epsilon)
still fall inside the output bound, which shows the bound is not vacuously
large but also not perfectly tight.

## `Conv2dWedge.attr` composition rules

| parameter | rule across two layers |
|---|---|
| `stride` | element-wise product |
| `dilation` | element-wise product |
| `padding` | element-wise sum |
| `groups` | must both be `1` (only `groups=1` supported) |
