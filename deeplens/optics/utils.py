import torch


# ================================
# Autograd Function diff_float
# ================================
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


# ================================
# Autograd Function diff_quantize
# ================================
class DiffQuantize(torch.autograd.Function):
    """Quantize tensor to n levels with gradient calculation (Straight-Through Estimator).

    Args:
        input (tensor): Input tensor.
        levels (int): Number of quantization levels.
        interval (float): Total range to quantize over (default: 2*pi).
    """

    @staticmethod
    def forward(ctx, x, levels, interval=2 * torch.pi):
        step = interval / levels
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None


def diff_quantize(input, levels, interval=2 * torch.pi):
    return DiffQuantize.apply(input, levels, interval)
