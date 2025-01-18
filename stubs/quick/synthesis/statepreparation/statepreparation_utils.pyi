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

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray as NDArray

__all__ = [
    "gray_code",
    "compute_alpha_y",
    "compute_alpha_z",
    "compute_m",
    "compute_control_indices",
    "bloch_angles",
    "rotations_to_disentangle",
    "k_s",
    "a",
    "b",
    "reverse_qubit_state",
    "disentangling_single_qubit_gates",
    "apply_ucg",
    "apply_diagonal_gate",
    "apply_diagonal_gate_to_diag",
    "apply_multi_controlled_gate",
    "ucg_is_identity_up_to_global_phase",
    "merge_ucgate_and_diag",
    "construct_basis_states",
    "diag_is_identity_up_to_global_phase",
    "get_binary_rep_as_list",
    "get_qubits_by_label"
]

def gray_code(index: int) -> int: ...
def compute_alpha_y(magnitude: NDArray[np.float64], k: int, j: int) -> float: ...
def compute_alpha_z(phase: NDArray[np.float64], k: int, j: int) -> float: ...
def compute_m(k: int) -> NDArray[np.float64]: ...
def compute_control_indices(index: int) -> list[int]: ...
def bloch_angles(pair_of_complex: Sequence[complex]) -> tuple: ...
def rotations_to_disentangle(local_param: NDArray[np.complex128]) -> tuple: ...
def k_s(k: int, s: int) -> int: ...
def a(k: int, s: int) -> int: ...
def b(k: int, s: int) -> int: ...
def reverse_qubit_state(state: NDArray[np.complex128], basis_state: int) -> NDArray[np.complex128]: ...
def find_squs_for_disentangling(v: NDArray[np.complex128], k: int, s: int, n: int) -> list[NDArray[np.complex128]]: ...
def apply_ucg(m: NDArray[np.complex128], k: int, single_qubit_gates: list[NDArray[np.complex128]]) -> NDArray[np.complex128]: ...
def apply_diagonal_gate(
        m: NDArray[np.complex128],
        action_qubit_labels: list[int],
        diagonal: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
    ...
def apply_diagonal_gate_to_diag(
        m_diagonal: NDArray[np.complex128],
        action_qubit_labels: list[int],
        diagonal: NDArray[np.complex128],
        num_qubits: int
    ) -> NDArray[np.complex128]:
    ...
def apply_multi_controlled_gate(
        m: NDArray[np.complex128],
        control_labels: list[int],
        target_label: int,
        gate: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
    ...
def ucg_is_identity_up_to_global_phase(single_qubit_gates: list[NDArray[np.complex128]]) -> bool: ...
def merge_ucgate_and_diag(
        single_qubit_gates: list[NDArray[np.complex128]],
        diagonal: NDArray[np.complex128]
    ) -> list[NDArray[np.complex128]]:
    ...
def construct_basis_states(
        state_free: tuple[int, ...],
        control_set: set[int],
        target_label: int
    ) -> tuple[int, int]:
    ...
def diag_is_identity_up_to_global_phase(diagonal: NDArray[np.complex128]) -> bool: ...
def get_binary_rep_as_list(n: int, num_digits: int) -> list[int]: ...
def get_qubits_by_label(labels: list[int], qubits: list[int], num_qubits: int) -> list[int]: ...