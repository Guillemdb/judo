from typing import Union

import numpy

from judo.judo_backend import torch

Tensor = Union[numpy.ndarray, torch.Tensor]
Vector = Union[numpy.ndarray, torch.Tensor]
Matrix = Union[numpy.ndarray, torch.Tensor]
Scalar = Union[int, float]
