import torch.nn.functional as F
import torch

def conv2d(x, weight, bias=None, stride=1, padding=1, dilation=1):
    B, C_in, H_in, W_in = x.shape
    C_out, C_in_w, Kh, Kw = weight.shape
    assert C_in_w == C_in
    if isinstance(stride, int):
        Sh = Sw = stride
    else:
        Sh, Sw = int(stride[0]), int(stride[1])
    if isinstance(padding, int):
        Ph = Pw = padding
    else:
        Ph, Pw = int(padding[0]), int(padding[1])
    if isinstance(dilation, int):
        Dh = Dw = dilation
    else:
        Dh, Dw = int(dilation[0]), int(dilation[1])
    # padding
    x_pad = F.pad(x, (Pw, Pw, Ph, Ph))
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
    # init output
    H_out = (H_pad - Dh * (Kh - 1) - 1) // Sh + 1
    W_out = (W_pad - Dw * (Kw - 1) - 1) // Sw + 1
    out = torch.zeros(B, C_out, H_out, W_out, dtype=x.dtype, device=x.device)
    # convolve
    for b in range(B):
        for c_out in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    acc = torch.zeros((), dtype=x.dtype, device=x.device)
                    for c_in in range(C_in):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                ih = h_out * Sh + kh * Dh
                                iw = w_out * Sw + kw * Dw
                                acc += x_pad[b, c_in, ih, iw] * weight[c_out, c_in, kh, kw]
                    if bias is not None:
                        acc = acc + bias[c_out]
                    out[b, c_out, h_out, w_out] = acc
    return out

if __name__ == '__main__':
    x = torch.randn(2, 3, 10, 10)

    weight = torch.randn(2, 3, 5, 5)
    bias = torch.randn(2)

    y = F.conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)
    y_hat = conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1)

    print(y.shape)
    print(
        torch.allclose(y, y_hat, rtol=1e-4, atol=1e-5),
        (y - y_hat).abs().max().item(),
    )
