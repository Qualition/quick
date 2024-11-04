import numpy as np
from numpy.typing import NDArray

__all__ = [
    "transform_to_magic_basis",
    "weyl_coordinates",
    "partition_eigenvalues",
    "remove_global_phase",
    "diagonalize_unitary_complex_symmetric",
    "decompose_two_qubit_product_gate",
    "TwoQubitWeylDecomposition"
]

def transform_to_magic_basis(U: NDArray[np.complex128], reverse: bool = False) -> NDArray[np.complex128]: ...
def weyl_coordinates(U: NDArray[np.complex128]) -> NDArray[np.float64]: ...
def partition_eigenvalues(eigenvalues: NDArray[np.complex128], atol: float = 1e-13) -> list[list[int]]: ...
def remove_global_phase(vector: NDArray[np.complex128], index: int | None = None) -> NDArray[np.complex128]: ...
def diagonalize_unitary_complex_symmetric(U, atol: float = 1e-13) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]: ...
def decompose_two_qubit_product_gate(special_unitary_matrix: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128], float]: ...

class TwoQubitWeylDecomposition:
    def __init__(self, unitary_matrix: NDArray[np.complex128]) -> None: ...
    @staticmethod
    def decompose_unitary(unitary_matrix: NDArray[np.complex128]) -> tuple[
            np.float64,
            np.float64,
            np.float64,
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            NDArray[np.complex128],
            float
        ]: ...
