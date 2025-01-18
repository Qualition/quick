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

""" Utility functions for multi-controlled gate decompositions.
"""

from __future__ import annotations

__all__ = [
    "CCX",
    "C3X"
]

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quick.circuit import Circuit

# Constants
PI8 = np.pi / 8


def CCX(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the CCX gate into a circuit with only 1 and 2 qubit gates.

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the CCX to.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.

    Raises
    ------
    ValueError
        If the number of control qubits is not 1 or 2.
    """
    if len(control_indices) == 1:
        circuit.CX(control_indices[0], target_index)

    elif len(control_indices) == 2:
        circuit.H(target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Tdg(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.T(control_indices[1])
        circuit.Tdg(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.CX(control_indices[0], control_indices[1])
        circuit.T(target_index)
        circuit.T(control_indices[0])
        circuit.Tdg(control_indices[1])
        circuit.H(target_index)
        circuit.CX(control_indices[0], control_indices[1])

    else:
        raise ValueError(
            f"CCX only supports 1 or 2 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )

def C3X(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the C3X gate into a circuit with only 1 and 2 qubit gates.

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the C3X to.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.

    Raises
    ------
    ValueError
        If the number of control qubits is not 1, 2 or 3.
    """
    if len(control_indices) == 1:
        circuit.CX(control_indices[0], target_index)

    elif len(control_indices) == 2:
        CCX(circuit, [control_indices[0], control_indices[1]], target_index)

    elif len(control_indices) == 3:
        circuit.H(target_index)
        circuit.Phase(PI8, control_indices + [target_index])
        circuit.CX(control_indices[0], control_indices[1])
        circuit.Phase(-PI8, control_indices[1])
        circuit.CX(control_indices[0], control_indices[1])
        circuit.CX(control_indices[1], control_indices[2])
        circuit.Phase(-PI8, control_indices[2])
        circuit.CX(control_indices[0], control_indices[2])
        circuit.Phase(PI8, control_indices[2])
        circuit.CX(control_indices[1], control_indices[2])
        circuit.Phase(-PI8, control_indices[2])
        circuit.CX(control_indices[0], control_indices[2])
        circuit.CX(control_indices[2], target_index)
        circuit.Phase(-PI8, target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Phase(PI8, target_index)
        circuit.CX(control_indices[2], target_index)
        circuit.Phase(-PI8, target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.Phase(PI8, target_index)
        circuit.CX(control_indices[2], target_index)
        circuit.Phase(-PI8, target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Phase(PI8, target_index)
        circuit.CX(control_indices[2], target_index)
        circuit.Phase(-PI8, target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.H(target_index)

    else:
        raise ValueError(
            f"C3X only supports 1, 2 or 3 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )