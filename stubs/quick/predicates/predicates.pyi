# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/quick/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "is_square_matrix",
    "is_diagonal_matrix",
    "is_symmetric_matrix",
    "is_identity_matrix",
    "is_unitary_matrix",
    "is_hermitian_matrix",
    "is_positive_semidefinite_matrix",
    "is_isometry"
]

def is_square_matrix(matrix: NDArray[np.complex128]) -> bool: ...
def is_diagonal_matrix(matrix: NDArray[np.complex128], rtol: float = ..., atol: float = ...) -> bool: ...
def is_symmetric_matrix(matrix: NDArray[np.complex128], rtol: float = ..., atol: float = ...) -> bool: ...
def is_identity_matrix(matrix: NDArray[np.complex128], ignore_phase: bool = False, rtol: float = ..., atol: float = ...) -> bool: ...
def is_unitary_matrix(matrix: NDArray[np.complex128], rtol: float = ..., atol: float = ...) -> bool: ...
def is_hermitian_matrix(matrix: NDArray[np.complex128], rtol: float = ..., atol: float = ...) -> bool: ...
def is_positive_semidefinite_matrix(matrix: NDArray[np.complex128], rtol: float = ..., atol: float = ...) -> bool: ...
def is_isometry(matrix: NDArray[np.complex128], rtol: float = ..., atol: float = ...) -> bool: ...
