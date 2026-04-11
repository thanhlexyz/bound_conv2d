# Conv2d bounds with a wedge (upper / lower)

This repo shows a small setup for **interval bounds** on a convolution layer.

You start with a **wedge**: a pair of conv kernels and biases (`W_L`, `b_L`, `W_U`, `b_U`) that describe how bounds at the **output** of the layer relate to the activations right before that output. Initially the wedge can be the **identity** (each output channel passes through unchanged).

When you **fold in** a real conv layer (its `weight` and `bias`), you **compose** the wedge with that layer. After composition, the wedge talks about the **input** of the conv instead of its output: same story as multiplying matrices in the fully-connected case, but here kernels are combined (see `tool/conv2d_wedge.py`).

Finally you turn an input **`BoundLinf`** box (`L`, `U` per tensor entry) into an output box by applying the usual **mid ± |W|** (interval / IBP) rule with the composed kernels—either directly via `BoundLinf.conv`, or by building the wedge and calling `to_bound_tensor`.

Run the demo:

```bash
python main.py
```
