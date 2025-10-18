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

""" Module for generating the matrix representation of a single-qubit quantum gate.
"""

from __future__ import annotations

__all__ = [
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "S",
    "Sdg",
    "T",
    "Tdg",
    "RX",
    "RY",
    "RZ",
    "Phase",
    "U3"
]

import numpy as np

from quick.primitives import Operator


PauliX = Operator(
    label="X",
    data=np.array([
        [0, 1],
        [1, 0]
    ])
)

PauliY = Operator(
    label="Y",
    data=np.array([
        [0, -1j],
        [1j, 0]
    ])
)

PauliZ = Operator(
    label="Z",
    data=np.array([
        [1, 0],
        [0, -1]
    ])
)

Hadamard = Operator(
    label="H",
    data=np.array([
        [1, 1],
        [1, -1]
    ]) / np.sqrt(2)
)

S = Operator(
    label="S",
    data=np.array([
        [1, 0],
        [0, 1j]
    ])
)

Sdg = S.adjoint()

T = Operator(
    label="T",
    data=np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ])
)

Tdg = T.adjoint()

def RX(theta: float) -> Operator:
    """ Generate the RX rotation gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the RX rotation gate.
    """
    return Operator(
        label=f"RX({theta})",
        data=np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])
    )

def RY(theta: float) -> Operator:
    """ Generate the RY rotation gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the RY rotation gate.
    """
    return Operator(
        label=f"RY({theta})",
        data=np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
    )

def RZ(theta: float) -> Operator:
    """ Generate the RZ rotation gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The rotation angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the RZ rotation gate.
    """
    return Operator(
        label=f"RZ({theta})",
        data=np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ])
    )

def Phase(theta: float) -> Operator:
    """ Generate the Phase gate given angle parameter
    theta.

    Parameters
    ----------
    `theta` : float
        The phase angle in radians.

    Returns
    -------
    quick.primitives.Operator
        The matrix representation of the Phase gate.
    """
    return Operator(
        label=f"Phase({theta})",
        data=np.array([
            [1, 0],
            [0, np.exp(1j * theta)]
        ])
    )

def U3(
        theta: float,
        phi: float,
        lam: float
    ) -> Operator:
    """ Generate the U3 gate given angle parameters
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
        The matrix representation of the U3 gate.
    """
    return Operator(
        label=f"U3({theta}, {phi}, {lam})",
        data=np.array([
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
        ])
    )