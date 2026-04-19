import torch.nn.functional as F
import numpy as np
import torch

# setup
# 2, 3, 3, 3    -> 3, 5, 2, 5
# stride 2, 3   -> stride 3, 4
# dilation 2, 3 -> dilation 3, 4
# padding (example): padding1=(3,3), padding2=(4,4)
kernel_size_1 = np.array([3, 3])
kernel_size_2 = np.array([3, 5])
stride1 = np.array([2, 3])
stride2 = np.array([3, 4])
padding1 = np.array([3, 3])
padding2 = np.array([4, 4])
dilation1 = np.array([2, 3])
dilation2 = np.array([3, 4])
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


def build_merged_kernel(w1, w2, dilation1, stride1, dilation2):
    dilated = get_dilated_weight(w1, dilation1[0], dilation1[1])
    dil = [stride1[0] * dilation2[0], stride1[1] * dilation2[1]]
    pad = [(w2.shape[2] - 1) * dil[0], (w2.shape[3] - 1) * dil[1]]
    return F.conv2d(dilated.permute(1, 0, 2, 3), w2.flip(-1, -2),
                    padding=pad, dilation=dil).permute(1, 0, 2, 3)


# merge
stride12 = stride1 * stride2
padding12 = tuple(int(v) for v in (padding1 + padding2 * stride1))
weight12 = build_merged_kernel(weight1, weight2, dilation1, stride1, dilation2)
_ = torch.ones(1, 3, *kernel_size_2) * bias1[None, :, None, None]
bias12 = F.conv2d(_, weight2, bias2).flatten()

# For the loose bound: the same merge formula with |w1| and |w2|.
weight12_abs = build_merged_kernel(weight1.abs(), weight2.abs(), dilation1, stride1, dilation2)
# |bias1 leakage| bounded per output channel.
bias_bound = (weight2.abs() * bias1.abs().view(1, -1, 1, 1)).sum(dim=(1, 2, 3))


def _border_mask(shape, p2, s2):
    bh = (int(p2[0]) + int(s2[0]) - 1) // int(s2[0]) if p2[0] > 0 else 0
    bw = (int(p2[1]) + int(s2[1]) - 1) // int(s2[1]) if p2[1] > 0 else 0
    mask = torch.zeros(shape)
    h_out, w_out = shape[2], shape[3]
    if bh > 0:
        mask[..., :min(bh, h_out), :] = 1
        mask[..., max(0, h_out - bh):, :] = 1
    if bw > 0:
        mask[..., :, :min(bw, w_out)] = 1
        mask[..., :, max(0, w_out - bw):] = 1
    return mask


def _tight_bound(x):
    # z_enlarged = conv1(x, pad = p1 + p2*s1). Its shape is exactly H_z_chain + 2*p2 in each
    # spatial axis. Running conv2 with pad=0 over z_enlarged would reproduce y_merge; the chain
    # value y_chain is the same conv2 over z_enlarged with the virtual (pad-p2) rim zeroed out.
    # So y_merge - y_chain = conv2(z_virtual, w2, pad=0) where z_virtual keeps only the rim.
    z = F.conv2d(
        x, weight1, bias1,
        stride=tuple(stride1), dilation=tuple(dilation1),
        padding=tuple(int(v) for v in (padding1 + padding2 * stride1)),
    )
    z_virtual = torch.zeros_like(z)
    p2h, p2w = int(padding2[0]), int(padding2[1])
    if p2h > 0:
        z_virtual[..., :p2h, :] = z[..., :p2h, :]
        z_virtual[..., -p2h:, :] = z[..., -p2h:, :]
    if p2w > 0:
        z_virtual[..., :, :p2w] = z[..., :, :p2w]
        z_virtual[..., :, -p2w:] = z[..., :, -p2w:]
    # Triangle inequality over k2 only (conv1 has already summed across k1 and c_in).
    return F.conv2d(z_virtual.abs(), weight2.abs(), None,
                    stride=tuple(stride2), dilation=tuple(dilation2), padding=0)


def _loose_bound(x):
    # Triangle inequality applied per (k1, k2, c_in) pair — looser but cheaper.
    bound_x = F.conv2d(x.abs(), weight12_abs, None,
                       stride=tuple(stride12), padding=padding12)
    return bound_x + bias_bound.view(1, -1, 1, 1)


def conv12(x, mode='exact'):
    """Merged conv with optional sound bounds at the border; never uses y_chain.

    modes:
      'exact'         -> raw merged conv; interior matches conv2(conv1(x)), border may not.
      'upper'         -> elementwise >= chain; TIGHT bound (uses conv1 + conv2 over the rim).
      'lower'         -> elementwise <= chain; TIGHT bound.
      'loose_upper'   -> elementwise >= chain; LOOSE bound (only uses |x| conv with |w12|).
      'loose_lower'   -> elementwise <= chain; LOOSE bound.
    """
    y_merge = F.conv2d(x, weight12, bias12, stride=tuple(stride12), padding=padding12)
    if mode == 'exact':
        return y_merge
    if mode in ('upper', 'lower'):
        bound = _tight_bound(x)  # already masked to the rim by construction (interior is 0)
    elif mode in ('loose_upper', 'loose_lower'):
        mask = _border_mask(y_merge.shape, padding2, stride2).to(y_merge)
        bound = mask * _loose_bound(x)
    else:
        raise ValueError(f'unknown mode {mode!r}')
    if mode in ('upper', 'loose_upper'):
        return y_merge + bound
    return y_merge - bound


# verify output
x = torch.randn(1, 2, 199, 199)
y = F.conv2d(
    F.conv2d(x, weight1, bias1,
             stride=tuple(stride1), dilation=tuple(dilation1), padding=tuple(padding1)),
    weight2, bias2,
    stride=tuple(stride2), dilation=tuple(dilation2), padding=tuple(padding2),
)

y_upper = conv12(x, mode='upper')
y_lower = conv12(x, mode='lower')
y_upper_loose = conv12(x, mode='loose_upper')
y_lower_loose = conv12(x, mode='loose_lower')

print(f'{y.shape=} {y_upper.shape=} {y_lower.shape=}')

assert torch.all(y_upper >= y - 1e-4), 'tight upper must dominate chain'
assert torch.all(y_lower <= y + 1e-4), 'tight lower must be dominated by chain'
assert torch.all(y_upper_loose >= y - 1e-4), 'loose upper must dominate chain'
assert torch.all(y_lower_loose <= y + 1e-4), 'loose lower must be dominated by chain'

mask = _border_mask(y.shape, padding2, stride2).bool()
assert torch.allclose(y_upper[~mask], y[~mask], rtol=1e-4, atol=1e-5)
assert torch.allclose(y_lower[~mask], y[~mask], rtol=1e-4, atol=1e-5)

gap_tight_u = (y_upper - y)[mask].abs().mean().item()
gap_tight_l = (y - y_lower)[mask].abs().mean().item()
gap_loose_u = (y_upper_loose - y)[mask].abs().mean().item()
gap_loose_l = (y - y_lower_loose)[mask].abs().mean().item()
print(f'tight avg border gap: upper={gap_tight_u:.3f} lower={gap_tight_l:.3f}')
print(f'loose avg border gap: upper={gap_loose_u:.3f} lower={gap_loose_l:.3f}')
print('passed')
