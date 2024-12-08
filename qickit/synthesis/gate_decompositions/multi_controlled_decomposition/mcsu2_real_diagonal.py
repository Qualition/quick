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

__all__ = [
    "MCRX",
    "MCRY",
    "MCRZ"
]

import math
import numpy as np
from numpy.typing import NDArray
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.circuit.gate_matrix import RX, RY, RZ
from qickit.predicates import is_unitary_matrix
from .mcx_vchain import mcx_vchain_decomposition
from qickit.synthesis.gate_decompositions import OneQubitDecomposition


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
    `circuit` : qickit.circuit.Circuit
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
    `circuit` : qickit.circuit.Circuit
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

# TODO: Must be implemented to update circuit inplace
def mcsu2_real_diagonal_decomposition(
        output_framework: Type[Circuit],
        unitary: NDArray[np.complex128],
        num_controls: int,
    ) -> Circuit:
    """ Decomposition of a multi-controlled SU2 gate with real diagonal
    into a circuit with only CX and one qubit gates.

    Notes
    -----
    This decomposition is used to decompose MCRX, MCRY, and MCRZ gates
    using CX and one qubit gates.

    Parameters
    ----------
    `output_framework` : Type[Circuit]
        The framework to use for the output circuit.
    `unitary` : NDArray[np.complex128]
        The 2x2 unitary matrix to become multi-controlled.
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    Circuit
        The circuit representation of the multi-controlled
        U2 gate with real diagonal.

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

    one_qubit_decomposition = OneQubitDecomposition(output_framework=output_framework)
    s_gate = one_qubit_decomposition.prepare_unitary(np.array(s_op))
    s_gate_adjoint = s_gate.copy()
    s_gate_adjoint.horizontal_reverse()

    k_1 = math.ceil(num_controls / 2.0)
    k_2 = math.floor(num_controls / 2.0)

    circuit = output_framework(num_controls + 1)
    controls = list(range(num_controls))
    target = num_controls

    if not is_secondary_diag_real:
        circuit.H(target)

    mcx_1 = mcx_vchain_decomposition(num_control_qubits=k_1, output_framework=output_framework)
    circuit.add(mcx_1, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.add(s_gate, [target])

    mcx_2 = mcx_vchain_decomposition(num_control_qubits=k_2, output_framework=output_framework)
    mcx_2.horizontal_reverse(adjoint=True)
    circuit.add(mcx_2, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.add(s_gate_adjoint, [target])

    mcx_3 = mcx_vchain_decomposition(num_control_qubits=k_1, output_framework=output_framework)
    circuit.add(mcx_3, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.add(s_gate, [target])

    mcx_4 = mcx_vchain_decomposition(num_control_qubits=k_2, output_framework=output_framework)
    circuit.add(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.add(s_gate_adjoint, [target])

    if not is_secondary_diag_real:
        circuit.H(target)

    return circuit

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
    `circuit` : qickit.circuit.Circuit
        The circuit to apply the multi-controlled RX gate.
    `theta` : float
        The angle of rotation.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    qubits = control_indices + [target_index]
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
            [theta_step, -np.pi / 2, np.pi / 2],
            control_indices,
            target_index
        )

    else:
        mcsu2_gate = mcsu2_real_diagonal_decomposition(
            type(circuit),
            RX(theta).matrix,
            num_controls=len(control_indices)
        )
        circuit.add(mcsu2_gate, qubits)

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
    `circuit` : qickit.circuit.Circuit
        The circuit to apply the multi-controlled RY gate.
    `theta` : float
        The angle of rotation.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    qubits = control_indices + [target_index]
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
        mcsu2_gate = mcsu2_real_diagonal_decomposition(
            type(circuit),
            RY(theta).matrix,
            num_controls=len(control_indices),
        )
        circuit.add(mcsu2_gate, qubits)

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
    `circuit` : qickit.circuit.Circuit
        The circuit to apply the multi-controlled RZ gate.
    `theta` : float
        The angle of rotation.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.
    """
    qubits = control_indices + [target_index]
    num_controls = len(control_indices)

    # Explicit decomposition for CRZ
    if num_controls == 1:
        circuit.RZ(theta/2, target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.RZ(-theta/2, target_index)
        circuit.CX(control_indices[0], target_index)

    else:
        mcsu2_gate = mcsu2_real_diagonal_decomposition(
            type(circuit),
            RZ(theta).matrix,
            num_controls=len(control_indices),
        )
        circuit.add(mcsu2_gate, qubits)