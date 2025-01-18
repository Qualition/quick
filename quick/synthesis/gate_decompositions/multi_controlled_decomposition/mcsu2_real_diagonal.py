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
mcx_vchain_decomposition = MCXVChain()

# Constants
PI2 = np.pi / 2


def generate_gray_code(num_bits: int) -> list[str]:
    """ Generate the gray code for ``num_bits`` bits.

    Parameters
    ----------
    `num_bits` : int
        The number of bits.

    Returns
    -------
    list[str]
        The gray code for the given number of bits.
    """
    if num_bits <= 0:
        raise ValueError("Cannot generate the gray code for less than 1 bit.")
    result = [0]
    for i in range(num_bits):
        result += [x + 2**i for x in reversed(result)]
    return [format(x, f"0{num_bits}b") for x in result]

def apply_cu(
        circuit: Circuit,
        angles: list[float],
        control_index: int,
        target_index: int
    ) -> None:
    """ Decomposition of the CU gate into a circuit with only 1 and 2 qubit gates.

    Notes
    -----
    This implementation is based on Nielson & Chuang 4.2 decomposition.

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the CU gate.
    `angles` : list[float]
        List of angles [theta, phi, lam].
    `control_index` : int
        Control qubit index.
    """
    theta, phi, lam = angles
    circuit.Phase((lam + phi) / 2, control_index)
    circuit.Phase((lam - phi) / 2, target_index)
    circuit.CX(control_index, target_index)
    circuit.Phase(-(phi + lam) / 2, target_index)
    circuit.RY(-theta / 2, target_index)
    circuit.CX(control_index, target_index)
    circuit.RY(theta / 2, target_index)
    circuit.Phase(phi, target_index)

def apply_mcu_graycode(
        circuit: Circuit,
        angles: list[float],
        control_indices: list[int],
        target_index: int
    ) -> None:
    """Apply multi-controlled u gate from controls to target using graycode
    pattern with single-step angles theta, phi, lam.

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the multi-controlled U gate.
    `angles` : list[float]
        List of angles [theta, phi, lam].
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    theta, phi, lam = angles
    n = len(control_indices)

    gray_code = generate_gray_code(n)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # Find left most set bit
        lm_pos = list(pattern).index("1")

        # Find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]

        if True in comp:
            pos = comp.index(True)
        else:
            pos = None

        if pos is not None:
            if pos != lm_pos:
                circuit.CX(control_indices[pos], control_indices[lm_pos])
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    circuit.CX(control_indices[idx], control_indices[lm_pos])

        # Check parity and undo rotation
        if pattern.count("1") % 2 == 0:
            # Inverse CU: u(theta, phi, lamb)^dagger = u(-theta, -lam, -phi)
            apply_cu(circuit, [-theta, -lam, -phi], control_indices[lm_pos], target_index)
        else:
            apply_cu(circuit, [theta, phi, lam], control_indices[lm_pos], target_index)

        last_pattern = pattern

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
    using CX and one qubit gates.

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
        s_op = [[1.0, 0.0], [0.0, 1.0j]]
    else:
        alpha_r = math.sqrt((math.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z.imag / (
            2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0))
        )
        alpha = alpha_r + 1.0j * alpha_i
        beta = x / (2.0 * math.sqrt((z.real + 1.0) * (math.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]]) # type: ignore

    one_qubit_decomposition = OneQubitDecomposition(output_framework=type(circuit))
    s_gate = one_qubit_decomposition.prepare_unitary(np.array(s_op))
    s_gate_adjoint = s_gate.copy()
    s_gate_adjoint.horizontal_reverse()

    control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
    num_controls = len(control_indices)
    k_1 = math.ceil(num_controls / 2.0)
    k_2 = math.floor(num_controls / 2.0)

    if not is_secondary_diag_real:
        circuit.H(target_index)

    mcx_vchain_decomposition.apply_decomposition(
        circuit,
        control_indices[:k_1],
        target_index,
        control_indices[k_1 : 2 * k_1 - 2]
    )
    circuit.add(s_gate, [target_index])

    mcx_vchain_decomposition.apply_decomposition(
        circuit,
        control_indices[k_1:],
        target_index,
        control_indices[k_1 - k_2 + 2 : k_1]
    )
    circuit.add(s_gate_adjoint, [target_index])

    mcx_vchain_decomposition.apply_decomposition(
        circuit,
        control_indices[:k_1],
        target_index,
        control_indices[k_1 : 2 * k_1 - 2]
    )
    circuit.add(s_gate, [target_index])

    mcx_vchain_decomposition.apply_decomposition(
        circuit,
        control_indices[k_1:],
        target_index,
        control_indices[k_1 - k_2 + 2 : k_1]
    )
    circuit.add(s_gate_adjoint, [target_index])

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

    elif num_controls < 4:
        theta_step = theta * (1 / (2 ** (num_controls - 1)))
        apply_mcu_graycode(
            circuit,
            [theta_step, -PI2, PI2],
            control_indices,
            target_index
        )

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

    elif num_controls < 4:
        theta_step = theta * (1 / (2 ** (num_controls - 1)))
        apply_mcu_graycode(
            circuit,
            [theta_step, 0, 0],
            control_indices,
            target_index,
        )

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