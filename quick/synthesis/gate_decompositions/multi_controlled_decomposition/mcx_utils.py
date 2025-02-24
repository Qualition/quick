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

""" Utility functions for multi-controlled gate decompositions. Contains explicit
decompositions for specific number of control qubits for MCX.
"""

from __future__ import annotations

__all__ = [
    "CCX",
    "RCCX",
    "C3X",
    "C3SX",
    "RC3X",
    "C4X"
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

    Notes
    -----
    The implementation is based on the following paper, Section IV, B, (3):
    [1] Maslov.
    On the advantages of using relative phase Toffolis with an application to multiple
    control Toffoli optimization (2016).
    https://arxiv.org/pdf/1508.03273

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

    Usage
    -----
    >>> CCX(circuit, [0, 1], 2)
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
        circuit.Tdg(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.T(target_index)
        circuit.H(target_index)

        # Phase correction
        circuit.T(control_indices[1])
        circuit.CX(control_indices[0], control_indices[1])
        circuit.T(control_indices[0])
        circuit.Tdg(control_indices[1])
        circuit.CX(control_indices[0], control_indices[1])

    else:
        raise ValueError(
            f"CCX only supports 1 or 2 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )

def RCCX(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the relative-phase CCX gate into a circuit
    with only 1 and 2 qubit gates.

    Notes
    -----
    The relative-phase CCX gate is a simplified Toffoli gate, also referred to
    as Margolus gate. This gate can be used in place of the Toffoli gate in cases
    where the Toffoli gate is uncomputed, or when the circuit is measured right
    after RCCX.

    The implementation is based on the following paper Fig. 3:
    [1] Maslov.
    On the advantages of using relative phase Toffolis with an application to multiple
    control Toffoli optimization (2016).
    https://arxiv.org/pdf/1508.03273

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
        If the number of control qubits is not 2.

    Usage
    -----
    >>> RCCX(circuit, [0, 1], 2)
    """
    if len(control_indices) == 2:
        circuit.H(target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Tdg(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Tdg(target_index)
        circuit.H(target_index)

    else:
        raise ValueError(
            f"RCCX only supports 2 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )

def C3X(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the C3X gate into a circuit with only 1 and 2 qubit gates.

    Notes
    -----
    The implementation is based on the following paper:
    [1] Barenco, Bennett, Cleve, DiVincenzo, Margolus, Shor, Sleator, Smolin, Weinfurter.
    Elementary gates for quantum computation (1995).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

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

    Usage
    -----
    >>> C3X(circuit, [0, 1, 2], 3)
    """
    num_controls = len(control_indices)

    if num_controls == 1:
        circuit.CX(control_indices[0], target_index)

    elif num_controls == 2:
        CCX(circuit, [control_indices[0], control_indices[1]], target_index)

    elif num_controls == 3:
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

def C3SX(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the 3 controlled sqrt(X) gate into a circuit with only 1 and 2 qubit gates.

    Notes
    -----
    The implementation is based on the following paper:
    [1] Barenco, Bennett, Cleve, DiVincenzo, Margolus, Shor, Sleator, Smolin, Weinfurter.
    Elementary gates for quantum computation (1995).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the C3SX to.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.

    Raises
    ------
    ValueError
        If the number of control qubits is not 3.

    Usage
    -----
    >>> C3SX(circuit, [0, 1, 2], 3)
    """
    if len(control_indices) == 3:
        circuit.H(target_index)
        circuit.CPhase(PI8, control_indices[0], target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[0], control_indices[1])
        circuit.H(target_index)
        circuit.CPhase(-PI8, control_indices[1], target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[0], control_indices[1])
        circuit.H(target_index)
        circuit.CPhase(PI8, control_indices[1], target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[1], control_indices[2])
        circuit.H(target_index)
        circuit.CPhase(-PI8, control_indices[2], target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[0], control_indices[2])
        circuit.H(target_index)
        circuit.CPhase(PI8, control_indices[2], target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[1], control_indices[2])
        circuit.H(target_index)
        circuit.CPhase(-PI8, control_indices[2], target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[0], control_indices[2])
        circuit.H(target_index)
        circuit.CPhase(PI8, control_indices[2], target_index)
        circuit.H(target_index)

    else:
        raise ValueError(
            f"C3SX only supports 3 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )

def RC3X(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the relative-phase C3X gate into a circuit
    with only 1 and 2 qubit gates.

    Notes
    -----
    The relative-phase C3X gate is a simplified Toffoli gate, also referred to
    as Margolus gate. This gate can be used in place of the Toffoli gate in cases
    where the Toffoli gate is uncomputed, or when the circuit is measured right
    after RC3X.

    The implementation is based on the following paper Fig. 4:
    [1] Maslov.
    On the advantages of using relative phase Toffolis with an application to multiple
    control Toffoli optimization (2016).
    https://arxiv.org/pdf/1508.03273

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the RC3X to.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.

    Raises
    ------
    ValueError
        If the number of control qubits is not 3.

    Usage
    -----
    >>> RC3X(circuit, [0, 1, 2], 3)
    """
    if len(control_indices) == 3:
        circuit.H(target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[2], target_index)
        circuit.Tdg(target_index)
        circuit.H(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Tdg(target_index)
        circuit.CX(control_indices[0], target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[1], target_index)
        circuit.Tdg(target_index)
        circuit.H(target_index)
        circuit.T(target_index)
        circuit.CX(control_indices[2], target_index)
        circuit.Tdg(target_index)
        circuit.H(target_index)

    else:
        raise ValueError(
            f"RC3X only supports 3 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )

def C4X(
        circuit: Circuit,
        control_indices: list[int],
        target_index: int
    ) -> None:
    """ Explicit decomposition of the C4X gate into a circuit with only 1 and 2 qubit gates.

    Notes
    -----
    The implementation is based on the following paper:
    [1] Barenco, Bennett, Cleve, DiVincenzo, Margolus, Shor, Sleator, Smolin, Weinfurter.
    Elementary gates for quantum computation (1995).
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    Parameters
    ----------
    `circuit` : quick.circuit.Circuit
        The circuit to apply the C4X to.
    `control_indices` : list[int]
        List of control qubit indices.
    `target_index` : int
        Target qubit index.

    Raises
    ------
    ValueError
        If the number of control qubits is not 1, 2, 3 or 4.

    Usage
    -----
    >>> C4X(circuit, [0, 1, 2, 3], 4)
    """
    num_controls = len(control_indices)

    if num_controls < 4:
        C3X(circuit, control_indices, target_index)

    elif num_controls == 4:
        circuit.H(target_index)
        circuit.CS(control_indices[3], target_index)
        circuit.H(target_index)
        RC3X(circuit, control_indices[:-1], control_indices[-1])
        circuit.H(target_index)
        circuit.CSdg(control_indices[3], target_index)
        circuit.H(target_index)

        # RC3X inverse
        rc3x_inverse = type(circuit)(4)
        RC3X(rc3x_inverse, control_indices[:-1], control_indices[-1])
        rc3x_inverse.horizontal_reverse()
        circuit.add(rc3x_inverse, control_indices)

        C3SX(circuit, control_indices[:-1], target_index)

    else:
        raise ValueError(
            f"C4X only supports 1, 2, 3 or 4 control qubits. "
            f"Received {len(control_indices)} control qubits."
        )