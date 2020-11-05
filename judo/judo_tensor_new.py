from judo.judo_backend import Backend, torch
from judo.data_types import dtype, typing


def _new_torch_tensor(x, use_grad, device, *args, **kwargs):
    try:
        new_tensor = torch.tensor(x, *args, requires_grad=use_grad, device=device, **kwargs,)
    except Exception:
        new_tensor = torch.tensor(x, *args, requires_grad=False, device=device, **kwargs)
    return new_tensor


def _assign_device_and_grad(x, device, use_grad):
    try:
        if x.requires_grad != use_grad and dtype.is_float(x):
            x = x.requires_grad_(use_grad)
    except RuntimeError:
        pass
    if x.device.type != device:
        x = x.to(device=device)
    return x