# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/QICKIT/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "update_angles",
    "decompose_uc_rotations",
    "extract_rz",
    "det_one_qubit",
    "demultiplex_single_uc",
    "decompose_ucg_help",
    "get_ucg_diagonal",
    "simplify",
    "repetition_search",
    "repetition_verify"
]

def update_angles(angle_1: float, angle_2: float) -> tuple[float, float]: ...
def decompose_uc_rotations(angles: NDArray[np.float64], start_index: int, end_index: int, reversed_dec: bool) -> None: ...
def extract_rz(phi_1: float, phi_2: float) -> tuple[float, float]: ...
def det_one_qubit(matrix: NDArray[np.complex128]) -> np.complex64: ...
def demultiplex_single_uc(a: NDArray[np.complex128], b: NDArray[np.complex128]) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]: ...
def decompose_ucg_help(sq_gates: list[NDArray[np.complex128]], num_qubits: int) -> tuple[list[NDArray[np.complex128]], NDArray[np.complex128]]: ...
def get_ucg_diagonal(sq_gates: list[NDArray[np.complex128]], num_qubits: int, simplified_controls: set[int]) -> NDArray[np.complex128]: ...
def simplify(gate_list: list[NDArray[np.complex128]], num_controls: int) -> tuple[set[int], list[NDArray[np.complex128]]]: ...
def repetition_search(mux: list[NDArray[np.complex128]], level: int) -> tuple[set[int], list[NDArray[np.complex128]]]: ...
def repetition_verify(base, d, mux, mux_copy) -> tuple[bool, list[NDArray[np.complex128]]]: ...
