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

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
import pytket
from quick.backend import Backend
from quick.circuit import Circuit
from typing import Callable, Literal

__all__ = ["TKETCircuit"]

GATES = Literal[
    "I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg",
    "RX", "RY", "RZ", "Phase", "U3",
    "MCX", "MCY", "MCZ", "MCH", "MCS", "MCSdg", "MCT", "MCTdg",
    "MCRX", "MCRY", "MCRZ", "MCPhase", "MCU3"
]

class TKETCircuit(Circuit):
    circuit: pytket.Circuit
    def __init__(self, num_qubits: int) -> None: ...
    @staticmethod
    def _define_gate_mapping() -> dict[str, Callable]: ...
    def _gate_mapping(
            self,
            gate: GATES,
            target_indices: int | Sequence[int],
            control_indices: int | Sequence[int] = [],
            angles: Sequence[float] = [0, 0, 0]
        ) -> None: ...
    def GlobalPhase(self, angle: float) -> None: ...
    measured: bool
    def measure(self, qubit_indices: int | Sequence[int]) -> None: ...
    def get_statevector(self, backend: Backend | None = None) -> NDArray[np.complex128]: ...
    def get_counts(self, num_shots: int, backend: Backend | None = None) -> dict[str, int]: ...
    def get_unitary(self) -> NDArray[np.complex128]: ...
    def reset_qubit(self, qubit_indices: int | Sequence[int]) -> None: ...
    def to_qasm(self, qasm_version: int=2) -> str: ...
    def draw(self) -> None: ...
