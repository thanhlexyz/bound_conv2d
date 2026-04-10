


if __name__ == '__main__':
    # import
    from .bound_tensor import BoundTensor
    from .bound_linf import BoundLinf

    # initialize x as bound tensor
    L = torch.randn(2, 3, 10, 10)
    U = L + 0.02
    p = BoundLinf(L, U)
    x = BoundTensor(L, p)
    # initialize weight and bias for conv2d
    weight = torch.randn(2, 3, 5, 5)
    bias = torch.randn(2)

    #




#     y = F.conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)
#     y_hat = conv2d(x, weight, bias=bias, stride=1, padding=1, dilation=1)
#
#     print(y.shape)
#     print(
#         torch.allclose(y, y_hat, rtol=1e-4, atol=1e-5),
#         (y - y_hat).abs().max().item(),
#     )
