import abc
import numpy as np
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numpy.typing import NDArray
from qickit.circuit import Circuit
from qickit.primitives import Operator
from qiskit_ibm_runtime import QiskitRuntimeService
from typing import Type


__all__ = ["UnitaryPreparation", "QiskitUnitaryTranspiler"]

class UnitaryPreparation(ABC, metaclass=abc.ABCMeta):
    output_framework: Type[Circuit]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    @abstractmethod
    def prepare_unitary(self, unitary: NDArray[np.complex128] | Operator) -> Circuit: ...
    @abstractmethod
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...


class QiskitUnitaryTranspiler(UnitaryPreparation):
    ai_transpilation: bool
    service: QiskitRuntimeService | None
    backend_name: str | None
    def __init__(self, output_framework: type[Circuit], ai_transpilation: bool = False, service: QiskitRuntimeService | None = None, backend_name: str | None = None) -> None: ...
    def prepare_unitary(self, unitary: NDArray[np.complex128] | Operator) -> Circuit: ...
    def apply_unitary(self, circuit: Circuit, unitary: NDArray[np.complex128] | Operator, qubit_indices: int | Sequence[int]) -> Circuit: ...
