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

""" Multi-controlled rotation gate decompositions.

This implementation is based on Qiskit's implementation.
https://github.com/Qiskit/qiskit/blob/stable/0.46/qiskit/circuit/library/standard_gates/multi_control_rotation_gates.py
"""

from __future__ import annotations

__all__ = [
    "MCRX",
    "MCRY",
    "MCRZ"
]

import math
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quick.circuit import Circuit
from quick.circuit.gate_matrix import RX, RY, RZ
from quick.predicates import is_unitary_matrix
from quick.synthesis.gate_decompositions.multi_controlled_decomposition.mcx_vchain import MCXVChain
from quick.synthesis.gate_decompositions import OneQubitDecomposition

# Global MCXVChain object
MCX_VCHAIN_DECOMPOSER = MCXVChain()

# Constants
PI2 = np.pi / 2


def mcsu2_real_diagonal_decomposition(
        circuit: Circuit,
        control_indices: int | list[int],
        target_index: int,
        unitary: NDArray[np.complex128]
    ) -> None:
    """ Decomposition of a multi-controlled SU2 gate with real diagonal
    into a circuit with only CX and one qubit gates.

    Notes
    -----
    This decomposition is used to decompose MCRX, MCRY, and MCRZ gates
    using CX and one qubit gates. The decomposition breaks the control
    indices into two sets and utilizes the V-chain decomposition to
    define the multi-controlled gate. By dividing the control qubits
    we can use the other half as ancilla qubits for each V-chain
    decomposition.

    The implementation is based on the following paper section 3.1:
    [1] Vale , Azevedo , Araujo, Araujo , da Silva.
    Decomposition of Multi-controlled Special Unitary Single-Qubit Gates (2023).
    https://arxiv.org/pdf/2302.06377

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the multi-controlled U gate.
    `control_indices` : int | list[int]
        The control qubits for the MCX gate.
    `target_index` : int
        The target qubit for the MCX gate.
    `unitary` : NDArray[np.complex128]
        The 2x2 unitary matrix to become multi-controlled.

    Raises
    ------
    ValueError
        If the unitary is not a 2x2 matrix.
        If the unitary is not an unitary matrix.
        If the determinant of the unitary is not one.
        If the unitary does not have one real diagonal.
    """
    if unitary.shape != (2, 2):
        raise ValueError(f"The unitary must be a 2x2 matrix, but has shape {unitary.shape}.")

    if not is_unitary_matrix(unitary):
        raise ValueError(f"The unitary in must be an unitary matrix, but is {unitary}.")

    if not np.isclose(1.0, np.linalg.det(unitary)):
        raise ValueError("Invalid Value _mcsu2_real_diagonal requires det(unitary) equal to one.")

    is_main_diag_real = np.isclose(unitary[0, 0].imag, 0.0) and np.isclose(unitary[1, 1].imag, 0.0)
    is_secondary_diag_real = np.isclose(unitary[0, 1].imag, 0.0) and np.isclose(
        unitary[1, 0].imag, 0.0
    )

    if not is_main_diag_real and not is_secondary_diag_real:
        raise ValueError("The unitary must have one real diagonal.")

    if is_secondary_diag_real:
        x = unitary[0, 1]
        z = unitary[1, 1]
    else:
        x = -unitary[0, 1].real
        z = unitary[1, 1] - unitary[0, 1].imag * 1.0j

    if np.isclose(z, -1):
        a_op = [[1.0, 0.0], [0.0, 1.0j]]
    else:
        alpha_r = math.sqrt((math.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z.imag / (
            2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0))
        )
        alpha = alpha_r + 1.0j * alpha_i
        beta = x / (2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        a_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]]) # type: ignore

    one_qubit_decomposition = OneQubitDecomposition(output_framework=type(circuit))

    # Define the A gate which is an SU2 gate (Eq 7)
    a_gate = one_qubit_decomposition.prepare_unitary(np.array(a_op))
    a_gate_adjoint = a_gate.copy()
    a_gate_adjoint.horizontal_reverse()

    control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
    num_controls = len(control_indices)

    # Cut the control qubits into two sets
    # Each set will use the other half as ancilla qubits for the V-chain decomposition
    k_1 = math.ceil(num_controls / 2.0)
    k_2 = math.floor(num_controls / 2.0)

    if not is_secondary_diag_real:
        circuit.H(target_index)

    # Implement circuit from Fig. 7
    MCX_VCHAIN_DECOMPOSER.apply_decomposition(
        circuit,
        control_indices[:k_1],
        target_index,
        control_indices[k_1 : 2 * k_1 - 2]
    )
    circuit.add(a_gate, [target_index])

    MCX_VCHAIN_DECOMPOSER.apply_decomposition(
        circuit,
        control_indices[k_1:],
        target_index,
        control_indices[k_1 - k_2 + 2 : k_1]
    )
    circuit.add(a_gate_adjoint, [target_index])

    MCX_VCHAIN_DECOMPOSER.apply_decomposition(
        circuit,
        control_indices[:k_1],
        target_index,
        control_indices[k_1 : 2 * k_1 - 2]
    )
    circuit.add(a_gate, [target_index])

    MCX_VCHAIN_DECOMPOSER.apply_decomposition(
        circuit,
        control_indices[k_1:],
        target_index,
        control_indices[k_1 - k_2 + 2 : k_1]
    )
    circuit.add(a_gate_adjoint, [target_index])

    if not is_secondary_diag_real:
        circuit.H(target_index)

def MCRX(
        circuit: Circuit,
        theta: float,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Decomposition of the multi-controlled RX gate into a circuit with
    only CX and one qubit gates.

    Notes
    -----
    Uses MCSU2 real diagonal decomposition to decompose the multi-controlled
    RX gate into a circuit with only CX and one qubit gates. Here, we pass RX
    gate as the SU2.

    The implementation is based on the following paper section 3.1:
    [1] Vale , Azevedo , Araujo, Araujo , da Silva.
    Decomposition of Multi-controlled Special Unitary Single-Qubit Gates (2023).
    https://arxiv.org/pdf/2302.06377

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the multi-controlled RX gate.
    `theta` : float
        The angle of rotation.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    num_controls = len(control_indices)

    # Explicit decomposition for CRX
    if num_controls == 1:
        circuit.S(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.RY(-theta/2, target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.RY(theta/2, target_index)
        circuit.Sdg(target_index)

    else:
        mcsu2_real_diagonal_decomposition(
            circuit,
            control_indices,
            target_index,
            RX(theta).matrix
        )

def MCRY(
        circuit: Circuit,
        theta: float,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Decomposition of the multi-controlled RY gate into a circuit with
    only CX and one qubit gates.

    Notes
    -----
    Uses MCSU2 real diagonal decomposition to decompose the multi-controlled
    RY gate into a circuit with only CX and one qubit gates. Here, we pass RY
    gate as the SU2.

    The implementation is based on the following paper section 3.1:
    [1] Vale , Azevedo , Araujo, Araujo , da Silva.
    Decomposition of Multi-controlled Special Unitary Single-Qubit Gates (2023).
    https://arxiv.org/pdf/2302.06377

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the multi-controlled RY gate.
    `theta` : float
        The angle of rotation.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    num_controls = len(control_indices)

    # Explicit decomposition for CRY
    if num_controls == 1:
        circuit.RY(theta/2, target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.RY(-theta/2, target_index)
        circuit.CX(control_indices[0], target_index)

    else:
        mcsu2_real_diagonal_decomposition(
            circuit,
            control_indices,
            target_index,
            RY(theta).matrix,
        )

def MCRZ(
        circuit: Circuit,
        theta: float,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Decomposition of the multi-controlled RZ gate into a circuit with
    only CX and one qubit gates.

    Notes
    -----
    Uses MCSU2 real diagonal decomposition to decompose the multi-controlled
    RZ gate into a circuit with only CX and one qubit gates. Here, we pass RZ
    gate as the SU2.

    The implementation is based on the following paper section 3.1:
    [1] Vale , Azevedo , Araujo, Araujo , da Silva.
    Decomposition of Multi-controlled Special Unitary Single-Qubit Gates (2023).
    https://arxiv.org/pdf/2302.06377

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the multi-controlled RZ gate.
    `theta` : float
        The angle of rotation.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    num_controls = len(control_indices)

    # Explicit decomposition for CRZ
    if num_controls == 1:
        circuit.RZ(theta/2, target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.RZ(-theta/2, target_index)
        circuit.CX(control_indices[0], target_index)

    else:
        mcsu2_real_diagonal_decomposition(
            circuit,
            control_indices,
            target_index,
            RZ(theta).matrix,
        )