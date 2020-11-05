"""Functionality for running fragile in numpy and pytorch."""
import sys
import numpy
from judo.judo_backend import Backend, torch
from judo import data_types as _data_types
from judo.data_types import DATA_TYPE_NAMES, dtype, typing
from judo.judo_tensor import tensor
from judo.functions.api import API, AVAILABLE_FUNCTIONS
from judo.functions.hashing import hasher
from judo.functions.random import random_state


def __base_getattr(name):
    raise AttributeError(f"module '{module.__name__}' has no attribute '{name}'")


def __new_getattr(name):
    if name in DATA_TYPE_NAMES:
        return getattr(_data_types, name)()
    elif name in AVAILABLE_FUNCTIONS:
        return getattr(API, name)
    try:
        return __old_getattr(name)
    except AttributeError as e:
        if Backend.is_numpy():
            return getattr(numpy, name)
        elif Backend.is_torch():
            return getattr(torch, name)
        raise e


module = sys.modules[__name__]
__old_getattr = getattr(module, "__getattr__", __base_getattr)
module.__getattr__ = __new_getattr
is_bool = dtype.is_bool
is_int = dtype.is_int
is_float = dtype.is_float
is_tensor = dtype.is_tensor
