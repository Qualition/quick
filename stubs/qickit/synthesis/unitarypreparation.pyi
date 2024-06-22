import abc
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from qiskit_ibm_runtime import QiskitRuntimeService
from qickit.circuit import Circuit
from typing import Callable, Type

__all__ = ["UnitaryPreparation", "QiskitUnitaryTranspiler"]

class UnitaryPreparation(ABC, metaclass=abc.ABCMeta):
    output_framework: Type[Circuit]
    def __init__(self, output_framework: Circuit) -> None: ...
    def check_unitary(self, unitary: NDArray[np.complex128]) -> bool: ...
    @staticmethod
    def unitarymethod(method: Callable) -> Callable: ...
    @abstractmethod
    def prepare_unitary(self, unitary: NDArray[np.complex128]) -> Circuit: ...


class QiskitUnitaryTranspiler(UnitaryPreparation):
    ai_transpilation: bool
    service: QiskitRuntimeService | None
    backend_name: str | None
    def __init__(self, output_framework: type[Circuit], ai_transpilation: bool = False, service: QiskitRuntimeService | None = None, backend_name: str | None = None) -> None: ...
    def prepare_unitary(self, unitary: NDArray[np.complex128]) -> Circuit: ...
