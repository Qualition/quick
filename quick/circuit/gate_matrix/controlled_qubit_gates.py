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

""" Module for generating the matrix representation of a controlled quantum gate.
"""

from __future__ import annotations

__all__ = [
    "CX",
    "CY",
    "CZ",
    "CH",
    "CS",
    "CSdg",
    "CT",
    "CTdg",
    "CRX",
    "CRY",
    "CRZ",
    "CPhase",
    "CU3",
    "MCX",
    "MCY",
    "MCZ",
    "MCH",
    "MCS",
    "MCSdg",
    "MCT",
    "MCTdg",
    "MCRX",
    "MCRY",
    "MCRZ",
    "MCPhase",
    "MCU3"
]

from quick.circuit.gate_matrix.single_qubit_gates import (
    PauliX, PauliY, PauliZ, Hadamard, S, Sdg, T, Tdg,
    RX, RY, RZ, Phase, U3
)
from quick.primitives import Operator


CX = PauliX.control(1)
CY = PauliY.control(1)
CZ = PauliZ.control(1)
CH = Hadamard.control(1)
CS = S.control(1)
CSdg = Sdg.control(1)
CT = T.control(1)
CTdg = Tdg.control(1)

def CRX(theta: float) -> Operator:
    """ Generate the controlled RX rotation gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the controlled RX rotation gate.
    """
    return RX(theta).control(1)

def CRY(theta: float) -> Operator:
    """ Generate the controlled RY rotation gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the controlled RY rotation gate.
    """
    return RY(theta).control(1)

def CRZ(theta: float) -> Operator:
    """ Generate the controlled RZ rotation gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the controlled RZ rotation gate.
    """
    return RZ(theta).control(1)

def CPhase(theta: float) -> Operator:
    """ Generate the controlled Phase gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The phase angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the controlled Phase gate.
    """
    return Phase(theta).control(1)

def CU3(
        theta: float,
        phi: float,
        lam: float
    ) -> Operator:
    """ Generate the controlled U3 gate given angle parameters
    theta, phi, and lam.

    Parameters
    ----------
    `theta` : float
        The theta angle in radians.
    `phi` : float
        The phi angle in radians.
    `lam` : float
        The lambda angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the controlled U3 gate.
    """
    return U3(theta, phi, lam).control(1)

def MCX(num_controls: int) -> Operator:
    """ Generate the multi-controlled X gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled X gate.
    """
    return PauliX.control(num_controls)

def MCY(num_controls: int) -> Operator:
    """ Generate the multi-controlled Y gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled Y gate.
    """
    return PauliY.control(num_controls)

def MCZ(num_controls: int) -> Operator:
    """ Generate the multi-controlled Z gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled Z gate.
    """
    return PauliZ.control(num_controls)

def MCH(num_controls: int) -> Operator:
    """ Generate the multi-controlled H gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled H gate.
    """
    return Hadamard.control(num_controls)

def MCS(num_controls: int) -> Operator:
    """ Generate the multi-controlled S gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled S gate.
    """
    return S.control(num_controls)

def MCSdg(num_controls: int) -> Operator:
    """ Generate the multi-controlled Sdg gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled Sdg gate.
    """
    return Sdg.control(num_controls)

def MCT(num_controls: int) -> Operator:
    """ Generate the multi-controlled T gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled T gate.
    """
    return T.control(num_controls)

def MCTdg(num_controls: int) -> Operator:
    """ Generate the multi-controlled Tdg gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled Tdg gate.
    """
    return Tdg.control(num_controls)

def MCRX(
        num_controls: int,
        theta: float
    ) -> Operator:
    """ Generate the multi-controlled RX gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled RX gate.
    """
    return RX(theta).control(num_controls)

def MCRY(
        num_controls: int,
        theta: float
    ) -> Operator:
    """ Generate the multi-controlled RY gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled RY gate.
    """
    return RY(theta).control(num_controls)

def MCRZ(
        num_controls: int,
        theta: float
    ) -> Operator:
    """ Generate the multi-controlled RZ gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled RZ gate.
    """
    return RZ(theta).control(num_controls)

def MCPhase(
        num_controls: int,
        theta: float
    ) -> Operator:
    """ Generate the multi-controlled Phase gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled Phase gate.
    """
    return Phase(theta).control(num_controls)

def MCU3(
        num_controls: int,
        theta: float,
        phi: float,
        lam: float
    ) -> Operator:
    """ Generate the multi-controlled U3 gate given the number of control qubits.

    Parameters
    ----------
    `num_controls` : int
        The number of control qubits.
    `theta` : float
        The theta angle in radians.
    `phi` : float
        The phi angle in radians.
    `lam` : float
        The lambda angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the multi-controlled U3 gate.
    """
    return U3(theta, phi, lam).control(num_controls)