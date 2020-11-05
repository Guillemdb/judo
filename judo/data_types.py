from typing import Any, Callable, Dict, Union

import numpy

from judo.judo_backend import Backend, torch

DATA_TYPE_NAMES = {
    "bool",
    "int",
    "uint8",
    "int16",
    "int32",
    "int64",
    "float",
    "float16",
    "float32",
    "float64",
    "hash_type",
}


def bool():
    funcs = {"numpy": lambda x: numpy.bool_, "torch": lambda x: torch.bool}
    return Backend.execute(None, funcs)


def int():
    funcs = {"numpy": lambda x: numpy.int64, "torch": lambda x: torch.int64}
    return Backend.execute(None, funcs)


def uint8():
    funcs = {"numpy": lambda x: numpy.uint8, "torch": lambda x: torch.uint8}
    return Backend.execute(None, funcs)


def int16():
    funcs = {"numpy": lambda x: numpy.int16, "torch": lambda x: torch.int16}
    return Backend.execute(None, funcs)


def int32():
    funcs = {"numpy": lambda x: numpy.int32, "torch": lambda x: torch.int32}
    return Backend.execute(None, funcs)


def int64():
    funcs = {"numpy": lambda x: numpy.int64, "torch": lambda x: torch.int64}
    return Backend.execute(None, funcs)


def float():
    funcs = {"numpy": lambda x: numpy.float32, "torch": lambda x: torch.float32}
    return Backend.execute(None, funcs)


def float16():
    funcs = {"numpy": lambda x: numpy.float16, "torch": lambda x: torch.float16}
    return Backend.execute(None, funcs)


def float32():
    funcs = {"numpy": lambda x: numpy.float32, "torch": lambda x: torch.float32}
    return Backend.execute(None, funcs)


def float64():
    funcs = {"numpy": lambda x: numpy.float64, "torch": lambda x: torch.float64}
    return Backend.execute(None, funcs)


def hash_type():
    funcs = {
        "numpy": lambda x: numpy.dtype("<U64") if Backend.use_true_hash() else numpy.int64,
        "torch": lambda x: torch.int64,
    }
    return Backend.execute(None, funcs)


class MetaScalar(type):
    @property
    def bool(cls):
        return bool()

    @property
    def uint8(cls):
        return uint8()

    @property
    def int16(cls):
        return int16()

    @property
    def int32(cls):
        return int32()

    @property
    def int64(cls):
        return int64()

    @property
    def int(cls):
        return int()

    @property
    def float(cls):
        return float()

    @property
    def float16(cls):
        return float16()

    @property
    def float32(cls):
        return float32()

    @property
    def float64(cls):
        return float64()

    @property
    def hash_type(cls):
        return hash_type()


class dtype(metaclass=MetaScalar):
    @classmethod
    def is_bool(cls, x):
        dtypes = (bool, dtype.bool)
        if cls.is_tensor(x):
            return x.dtype in dtypes
        return isinstance(x, dtypes)

    @classmethod
    def is_float(cls, x):
        dtypes = (float, cls.float64, cls.float32, cls.float16)
        if cls.is_tensor(x):
            return x.dtype in dtypes
        return isinstance(x, dtypes)

    @classmethod
    def is_int(cls, x):
        dtypes = (int, dtype.int64, dtype.int32, dtype.int16)
        if cls.is_tensor(x):
            return x.dtype in dtypes
        return isinstance(x, dtypes)

    @classmethod
    def is_tensor(cls, x):
        from judo.judo_tensor import tensor

        return isinstance(x, tensor.type)  # or cls.is_hash_tensor(x)

    @classmethod
    def to_node_id(cls, x):
        if Backend.is_numpy():
            return str(x) if Backend.use_true_hash() else int(x)
        elif Backend.is_torch():
            return int(x)


class MetaTyping(type):
    @property
    def int(self):
        try:
            return Union[int, dtype.int64, dtype.int32, dtype.int16]
        except Exception:
            return int

    @property
    def float(self):
        try:
            return Union[float, dtype.float64, dtype.float32, dtype.float16]
        except Exception:
            return float

    @property
    def bool(self):
        try:
            return Union[bool, dtype.bool]
        except Exception:
            return bool

    @property
    def Scalar(self):
        try:
            return Union[self.float, self.int]
        except Exception:
            return Union[int, float]


class typing(metaclass=MetaTyping):
    Tensor = Union[numpy.ndarray, torch.Tensor]
    StateDict = Dict[str, Dict[str, Any]]
    DistanceFunction = Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
