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

from __future__ import annotations

__all__ = ["mcx_vchain_decomposition"]

import numpy as np
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit


def _CCX(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Decomposition of the CCX gate into a circuit with only 1 and 2 qubit gates.

    Parameters
    ----------
    `circuit` : qickit.circuit.Circuit
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
        raise ValueError(f"CCX only supports 1 or 2 control qubits. Received {len(control_indices)} control qubits.")

def _C3X(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Decomposition of the C3X gate into a circuit with only 1 and 2 qubit gates.

    Parameters
    ----------
    `circuit` : qickit.circuit.Circuit
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
        _CCX(circuit, [control_indices[0], control_indices[1]], target_index)

    elif len(control_indices) == 3:
        circuit.H(3)
        circuit.Phase(np.pi / 8, [0, 1, 2, 3])
        circuit.CX(0, 1)
        circuit.Phase(-np.pi / 8, 1)
        circuit.CX(0, 1)
        circuit.CX(1, 2)
        circuit.Phase(-np.pi / 8, 2)
        circuit.CX(0, 2)
        circuit.Phase(np.pi / 8, 2)
        circuit.CX(1, 2)
        circuit.Phase(-np.pi / 8, 2)
        circuit.CX(0, 2)
        circuit.CX(2, 3)
        circuit.Phase(-np.pi / 8, 3)
        circuit.CX(1, 3)
        circuit.Phase(np.pi / 8, 3)
        circuit.CX(2, 3)
        circuit.Phase(-np.pi / 8, 3)
        circuit.CX(0, 3)
        circuit.Phase(np.pi / 8, 3)
        circuit.CX(2, 3)
        circuit.Phase(-np.pi / 8, 3)
        circuit.CX(1, 3)
        circuit.Phase(np.pi / 8, 3)
        circuit.CX(2, 3)
        circuit.Phase(-np.pi / 8, 3)
        circuit.CX(0, 3)
        circuit.H(3)

    else:
        raise ValueError(f"C3X only supports 1, 2 or 3 control qubits. Received {len(control_indices)} control qubits.")

def get_num_ancillas(num_control_qubits: int) -> int:
    """ Get the number of ancilla qubits required for the V-chain decomposition.

    Parameters
    ----------
    `num_control_qubits` : int
        Number of control qubits.

    Returns
    -------
    `num_ancillas` : int
        Number of ancilla qubits required for the V-chain decomposition.
    """
    return max(0, num_control_qubits - 2)

def mcx_vchain_decomposition(
        num_control_qubits: int,
        output_framework: Type[Circuit]
    ) -> Circuit:
    """ Synthesize an MCX gate using the V-chain decomposition.

    Notes
    -----
    This implementation is based on the following paper:
    - Iten, Colbeck, Kukuljan, Home, Christandl.
    Quantum circuits for isometries (2016).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    This decomposition is to be used for decomposing multi-controlled gates with
    only U3 and CX gates. This decomposition is only usable for 4 or more control qubits.
    If the number of control qubits is less than 4, the decomposition will default to the
    standard MCX decomposition for CX, CCX and C3X gates.

    Parameters
    ----------
    `num_control_qubits` : int
        Number of control qubits.
    `output_framework` : type[Circuit]
        The class of the circuit framework to be used.

    Returns
    -------
    `circuit` : qickit.circuit.Circuit
        The circuit decomposition of the MCX gate using V-Chain.
    """
    # Initialize the circuit
    num_qubits = num_control_qubits + 1 + get_num_ancillas(num_control_qubits)
    qubits = list(range(num_qubits))
    circuit = output_framework(num_qubits)

    # The V-chain decomposition for the MCX gate only works for 4+ control qubits
    if num_control_qubits == 1:
        circuit.CX(0, 1)
        return circuit
    elif num_control_qubits == 2:
        _CCX(circuit, [0, 1], 2)
        return circuit
    elif num_control_qubits == 3:
        _C3X(circuit, [0, 1, 2], 3)
        return circuit

    # Define the control, target and ancilla qubits
    control_indices = qubits[:num_control_qubits]
    target_index = qubits[num_control_qubits]
    ancilla_indices = qubits[num_control_qubits + 1 :]
    num_ancillas = num_control_qubits - 2
    targets = [target_index] + ancilla_indices[:num_ancillas][::-1]

    # Perform V-Chain decomposition of the MCX gate
    for _ in range(2):
        for i in range(num_control_qubits):
            if i < num_control_qubits - 2:
                if targets[i] != target_index:
                    circuit.H(targets[i])
                    circuit.T(targets[i])
                    circuit.CX(control_indices[num_control_qubits - i - 1], targets[i])
                    circuit.Tdg(targets[i])
                    circuit.CX(ancilla_indices[num_ancillas - i - 1], targets[i])
                else:
                    controls = [
                        control_indices[num_control_qubits - i - 1],
                        ancilla_indices[num_ancillas - i - 1],
                    ]

                    _CCX(circuit, [controls[0], controls[1]], targets[i])
            else:
                circuit.H(targets[i])
                circuit.T(targets[i])
                circuit.CX(control_indices[num_control_qubits - i - 2], targets[i])
                circuit.Tdg(targets[i])
                circuit.CX(control_indices[num_control_qubits - i - 1], targets[i])
                circuit.T(targets[i])
                circuit.CX(control_indices[num_control_qubits - i - 2], targets[i])
                circuit.Tdg(targets[i])
                circuit.H(targets[i])
                break

        for i in range(num_ancillas - 1):
            circuit.CX(ancilla_indices[i], ancilla_indices[i + 1])
            circuit.T(ancilla_indices[i + 1])
            circuit.CX(control_indices[2 + i], ancilla_indices[i + 1])
            circuit.Tdg(ancilla_indices[i + 1])
            circuit.H(ancilla_indices[i + 1])

    return circuit