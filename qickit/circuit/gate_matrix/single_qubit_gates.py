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

""" Module for generating the matrix representation of a single-qubit quantum gate.
"""

from __future__ import annotations

__all__ = [
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "S",
    "T",
    "RX",
    "RY",
    "RZ",
    "U3",
    "Phase"
]

import numpy as np

from qickit.circuit.gate_matrix import Gate


class PauliX(Gate):
    """ `qickit.gate_matrix.PauliX` class represents the Pauli-X gate.
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.gate_matrix.PauliX` instance.
        """
        super().__init__(
            "X",
            np.array([
                [0, 1],
                [1, 0]
            ])
        )

class PauliY(Gate):
    """ `qickit.gate_matrix.PauliY` class represents the Pauli-Y gate.
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.gate_matrix.PauliY` instance.
        """
        super().__init__(
            "Y",
            np.array([
                [0, -1j],
                [1j, 0]
            ])
        )

class PauliZ(Gate):
    """ `qickit.gate_matrix.PauliZ` class represents the Pauli-Z gate.
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.gate_matrix.PauliZ` instance.
        """
        super().__init__(
            "Z",
            np.array([
                [1, 0],
                [0, -1]
            ])
        )

class Hadamard(Gate):
    """ `qickit.gate_matrix.Hadamard` class represents the Hadamard gate.
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.gate_matrix.Hadamard` instance.
        """
        super().__init__(
            "H",
            np.array([
                [1, 1],
                [1, -1]
            ]) / np.sqrt(2)
        )

class S(Gate):
    """ `qickit.gate_matrix.S` class represents the S gate.
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.gate_matrix.S` instance.
        """
        super().__init__(
            "S",
            np.array([
                [1, 0],
                [0, 1j]
            ])
        )

class T(Gate):
    """ `qickit.gate_matrix.T` class represents the T gate.
    """
    def __init__(self) -> None:
        """ Initialize a `qickit.gate_matrix.T` instance.
        """
        super().__init__(
            "T",
            np.array([
                [1, 0],
                [0, np.exp(1j * np.pi / 4)]
            ])
        )

class RX(Gate):
    """ `qickit.gate_matrix.RX` class represents the RX gate.
    """
    def __init__(self, theta: float) -> None:
        """ Initialize a `qickit.gate_matrix.RX` instance.
        """
        super().__init__(
            f"RX({theta})",
            np.array([
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)]
            ])
        )

class RY(Gate):
    """ `qickit.gate_matrix.RY` class represents the RY gate.
    """
    def __init__(self, theta: float) -> None:
        """ Initialize a `qickit.gate_matrix.RY` instance.
        """
        super().__init__(
            f"RY({theta})",
            np.array([
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)]
            ])
        )

class RZ(Gate):
    """ `qickit.gate_matrix.RZ` class represents the RZ gate.
    """
    def __init__(self, theta: float) -> None:
        """ Initialize a `qickit.gate_matrix.RZ` instance.
        """
        super().__init__(
            f"RZ({theta})",
            np.array([
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)]
            ])
        )

class U3(Gate):
    """ `qickit.gate_matrix.U3` class represents the U3 gate.
    """
    def __init__(self, theta: float, phi: float, lam: float) -> None:
        """ Initialize a `qickit.gate_matrix.U3` instance.
        """
        super().__init__(
            f"U3({theta}, {phi}, {lam})",
            np.array([
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + lam)) * np.cos(theta / 2)]
            ])
        )

class Phase(Gate):
    """ `qickit.gate_matrix.Phase` class represents the Phase gate.
    """
    def __init__(self, theta: float) -> None:
        """ Initialize a `qickit.gate_matrix.Phase` instance.
        """
        super().__init__(
            f"Phase({theta})",
            np.array([
                [1, 0],
                [0, np.exp(1j * theta)]
            ])
        )