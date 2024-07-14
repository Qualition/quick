import numpy as np
import qickit.primitives.ket as ket
from numpy.typing import NDArray
from qickit.types import Scalar
from typing import overload

__all__ = ['Operator']

class Operator:
    label: str
    data: NDArray[np.complex128]
    shape: tuple[int, int]
    num_qubits: int
    def __init__(self, data: NDArray[np.complex128], label: str | None = None) -> None: ...
    @staticmethod
    def ishermitian(data: NDArray[np.number]) -> None: ...
    @overload
    def __mul__(self, other: Scalar) -> Operator: ...
    @overload
    def __mul__(self, other: ket.Ket) -> ket.Ket: ...
    @overload
    def __mul__(self, other: Operator) -> Operator: ...
    def __rmul__(self, other: Scalar) -> Operator: ...
