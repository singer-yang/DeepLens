import torch


class DiffFloat(torch.autograd.Function):
    """Convert double precision tensor to float precision with gradient calculation.

    Args:
        input (tensor): Double precision tensor.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.double()
        return grad_input


def diff_float(input):
    return DiffFloat.apply(input)
