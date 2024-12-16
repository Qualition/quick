import abc
from abc import ABC, abstractmethod
from qickit.circuit import Circuit
from typing import Any, Type

__all__ = ["FromFramework"]

class FromFramework(ABC, metaclass=abc.ABCMeta):
    output_framework: Type[Circuit]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    @abstractmethod
    def convert(self, circuit: Any) -> Circuit: ...
