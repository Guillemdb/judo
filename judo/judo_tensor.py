import numpy

from judo.judo_backend import Backend, torch
from judo.data_types import dtype, typing


def _new_torch_tensor(x, use_grad, device, *args, **kwargs):
    try:
        new_tensor = torch.tensor(x, *args, requires_grad=use_grad, device=device, **kwargs)
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


def to_backend_wrap(func, *args, **kwargs):  # Handle device placement automatically
    return to_backend(func(*args, **kwargs))


def copy_torch(x: torch.Tensor, requires_grad, device):
    grad = requires_grad if requires_grad is not None else Backend.use_grad()
    new_tensor = x.clone()
    if not grad:
        new_tensor = new_tensor.detach()
    new_tensor = to_backend(new_tensor, device=device, use_grad=requires_grad)
    return new_tensor


def copy(x, requires_grad: bool = None, device=None):
    if x is None:
        return
    if not dtype.is_tensor(x):
        x = JudoTensor(x)

    funcs = {
        "numpy": lambda x: x.copy(),
        "torch": lambda x: copy_torch(x, requires_grad, device),
    }
    return Backend.execute(x, funcs)


def astype(x, dtype):
    funcs = {
        "numpy": lambda x: x.astype(dtype),
        "torch": lambda x: x.to(dtype),
    }
    return Backend.execute(x, funcs)


def as_tensor(x, *args, **kwargs):
    funcs = {
        "numpy": lambda x: numpy.ascontiguousarray(x, *args, **kwargs),
        "torch": lambda x: torch.as_tensor(x, *args, **kwargs),
    }
    return Backend.execute(x, funcs)


def to_numpy(x, *args, **kwargs):
    try:
        if isinstance(x, numpy.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        else:
            return numpy.asarray(x, *args, **kwargs)
    except Exception:
        try:
            return numpy.array(x, *args, **kwargs)
        except Exception:
            return numpy.array(tuple(x), dtype=object)


def to_torch(
    x, use_grad: bool = None, device: str = None, copy: bool = False, *args, **kwargs
):
    use_grad = use_grad if use_grad is not None else Backend.use_grad()
    device = device if device is not None else str(Backend.get_device())
    if isinstance(x, torch.Tensor):
        _assign_device_and_grad(x, device=device, use_grad=use_grad)
        return x

    if isinstance(x, numpy.ndarray):
        x = (
            torch.from_numpy(x)
            if not copy
            else _new_torch_tensor(x, use_grad, device, *args, **kwargs)
        )

    elif not isinstance(x, torch.Tensor):
        if not copy:
            try:
                return as_tensor(x, *args, **kwargs)
            except Exception:
                x = _new_torch_tensor(x, use_grad, device, *args, **kwargs)
        else:
            x = _new_torch_tensor(x, use_grad, device, *args, **kwargs)
    x = _assign_device_and_grad(x, device, use_grad)
    return x


def to_backend(
    x: "typing.Tensor", use_grad: bool = None, device: str = None, *args, **kwargs
):
    if Backend.is_numpy():
        return to_numpy(x, *args, **kwargs)
    return to_torch(x, use_grad=use_grad, device=device, *args, **kwargs)


def match_backend(x, other):
    if isinstance(x, numpy.ndarray):
        return to_numpy(other)
    elif isinstance(x, torch.Tensor):
        return torch.tensor(other)


class JudoTensor:

    def __new__(cls, x, use_grad: bool = None, device: str = None, *args, **kwargs):
        return to_backend(x, use_grad=use_grad, device=device, *args, **kwargs)


