import abc
from abc import ABC, abstractmethod
from qickit.circuit import Circuit

__all__ = ['CircuitOptimizer', 'CNOTOptimizer']

class CircuitOptimizer(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    @abstractmethod
    def optimize(self, circuit: Circuit) -> Circuit: ...

class CNOTOptimizer(CircuitOptimizer):
    def optimize(self, circuit: Circuit) -> Circuit: ...
