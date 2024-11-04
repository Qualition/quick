import numpy as np
from numpy.typing import NDArray
from typing import Literal

__all__ = ["Gate"]

class Gate:
    name: str
    matrix: NDArray[np.complex128]
    num_qubits: int
    ordering: str
    def __init__(self, name: str, matrix: NDArray[np.complex128]) -> None: ...
    def adjoint(self) -> NDArray[np.complex128]: ...
    def control(self, num_control_qubits: int) -> Gate: ...
    def change_mapping(self, ordering: Literal["MSB", "LSB"]) -> None: ...
