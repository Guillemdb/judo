from typing import Optional

import torch

RETURN_TENSOR_FUNCS = ["new_tensor", "new_full", "new_empty", "new_ones", "new_zeros", "real", "imag",
                       "abs", "abs_", "absolute", "absolute_", "acos", "acos_", "arccos", "arccos_",
                       "add", "add_"
                       ]

class JudoTorch(torch.Tensor):

    def __init__(self, data: torch.Tensor, device=None, dtype=None):
        self._t = data

    def __getattribute__(self, name):
        if name == "_t":
            return super().__getattribute__(name)
        else:
            return self._t.__getattribute__(name)

    def __deepcopy__(self, memo):
        copy = self._t.__deepcopy__(memo)
        return JudoTorch(copy)

    def __reduce_ex__(self, proto):
        return self._t.__reduce_ex__(proto)

    def __setstate__(self, state):
        self._t.__setstate__(state)

    def __repr__(self):
        return self._t.__repr__()

    def __add__(self, other):
        return self._t.__add__(other)

    def __reversed__(self):
        return JudoTorch(self._t.__reversed__())

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if hasattr(a, '_t') else a for a in args]
        ret = super().__torch_function__(func, types, args, kwargs)
        return JudoTorch(ret)
