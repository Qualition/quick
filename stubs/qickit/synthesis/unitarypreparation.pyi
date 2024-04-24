from qickit.circuit import *
import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Callable

__all__ = ['UnitaryPreparation']

class UnitaryPreparation(ABC, metaclass=abc.ABCMeta):
    circuit_framework: Incomplete
    encoder: Incomplete
    def __init__(self, circuit_framework: Circuit) -> None: ...
    def check_unitary(self, unitary: NDArray[np.complex128]) -> bool: ...
    @staticmethod
    def unitarymethod(method: Callable) -> Callable: ...
    @abstractmethod
    def prepare_unitary(self, unitary: NDArray[np.complex128]) -> Circuit: ...
