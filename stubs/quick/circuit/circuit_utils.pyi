# Copyright 2023-2025 Qualition Computing LLC.
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
    "decompose_multiplexor_rotations",
    "multiplexed_rz_angles",
    "extract_uvr_matrices",
    "extract_single_qubits_and_diagonal",
    "multiplexor_diagonal_matrix",
    "simplify",
    "repetition_search",
    "repetition_verify"
]

def decompose_multiplexor_rotations(
        angles: NDArray[np.float64],
        start_index: int,
        end_index: int,
        reverse_decomposition: bool
    ) -> NDArray[np.float64]: ...
def multiplexed_rz_angles(phi_1: float, phi_2: float) -> tuple[float, float]: ...
def extract_uvr_matrices(
        a: NDArray[np.complex128],
        b: NDArray[np.complex128]
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]: ...
def extract_single_qubits_and_diagonal(
        single_qubit_gates: list[NDArray[np.complex128]],
        num_qubits: int
    ) -> tuple[list[NDArray[np.complex128]], NDArray[np.complex128]]: ...
def multiplexor_diagonal_matrix(
        single_qubit_gates: list[NDArray[np.complex128]],
        num_qubits: int,
        simplified_controls: set[int]
    ) -> NDArray[np.complex128]: ...
def simplify(
        gate_list: list[NDArray[np.complex128]],
        num_controls: int
    ) -> tuple[set[int], list[NDArray[np.complex128]]]: ...
def repetition_search(
        mux: list[NDArray[np.complex128]],
        level: int
    ) -> tuple[set[int], list[NDArray[np.complex128]]]: ...
def repetition_verify(base, d, mux, mux_copy) -> tuple[bool, list[NDArray[np.complex128]]]: ...
