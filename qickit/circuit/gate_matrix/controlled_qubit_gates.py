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

""" Module for generating the matrix representation of a controlled quantum gate.
"""

from __future__ import annotations

__all__ = [
    "CX",
    "CY",
    "CZ",
    "CH",
    "CS",
    "CT"
]

from qickit.circuit.gate_matrix import Gate
from qickit.circuit.gate_matrix.single_qubit_gates import (
    PauliX, PauliY, PauliZ, Hadamard, S, T
)


CX: Gate = PauliX().control(1)
CY: Gate = PauliY().control(1)
CZ: Gate = PauliZ().control(1)
CH: Gate = Hadamard().control(1)
CS: Gate = S().control(1)
CT: Gate = T().control(1)