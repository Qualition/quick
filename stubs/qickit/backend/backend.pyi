import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from qickit.circuit import Circuit
from types import NotImplementedType

__all__ = ['Backend', 'NoisyBackend', 'FakeBackend']

class Backend(ABC, metaclass=abc.ABCMeta):
    device: str
    def __init__(self, device: str = 'CPU') -> None: ...
    @staticmethod
    def backendmethod(method): ...
    @abstractmethod
    def get_statevector(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    @abstractmethod
    def get_operator(self, circuit: Circuit) -> NDArray[np.complex128]: ...
    @abstractmethod
    def get_counts(self, circuit: Circuit, num_shots: int) -> dict[str, int]: ...
    @classmethod
    def __subclasscheck__(cls, C) -> bool: ...
    @classmethod
    def __subclasshook__(cls, C) -> bool | NotImplementedType: ...
    @classmethod
    def __instancecheck__(cls, C) -> bool: ...

class NoisyBackend(Backend, metaclass=abc.ABCMeta):
    single_qubit_error: float
    two_qubit_error: float
    noisy: bool
    def __init__(self, single_qubit_error: float, two_qubit_error: float, device: str = 'CPU') -> None: ...

class FakeBackend(Backend, metaclass=abc.ABCMeta):
    def __init__(self, device: str = 'CPU') -> None: ...
