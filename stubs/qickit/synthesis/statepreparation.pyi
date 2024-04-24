import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from qickit.circuit import Circuit
from qickit.data import Data
from typing import Type

__all__ = ['StatePreparation', 'Mottonen', 'Shende']

class StatePreparation(ABC, metaclass=abc.ABCMeta):
    circuit_framework: Incomplete
    encoder: Incomplete
    def __init__(self, circuit_framework: Type[Circuit]) -> None: ...
    @abstractmethod
    def prepare_state(self, state: NDArray[np.complex128] | Data, compression_percentage: float = 0.0, index_type: str = 'row') -> Circuit: ...

class Mottonen(StatePreparation):
    def prepare_state(self, state: NDArray[np.complex128] | Data, compression_percentage: float = 0.0, index_type: str = 'row') -> Circuit: ...

class Shende(StatePreparation):
    def prepare_state(self, state: NDArray[np.complex128] | Data, compression_percentage: float = 0.0, index_type: str = 'row') -> Circuit: ...
