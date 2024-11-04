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

__all__ = ["TestUniformlyControlledGates"]

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
import pytest
from typing import Type

from qickit.circuit import Circuit
from qickit.circuit.gate_matrix import PauliX, PauliY, Hadamard, RX, RY

from tests.circuit import CIRCUIT_FRAMEWORKS
from tests.circuit.gate_utils import (
    UCRX_unitary_matrix_3qubits_01control,
    UCRX_unitary_matrix_3qubits_10control,
    UCRX_unitary_matrix_4qubits_023control,
    UCRX_unitary_matrix_4qubits_213control,
    UCRY_unitary_matrix_3qubits_01control,
    UCRY_unitary_matrix_3qubits_10control,
    UCRY_unitary_matrix_4qubits_023control,
    UCRY_unitary_matrix_4qubits_213control,
    UCRZ_unitary_matrix_3qubits_01control,
    UCRZ_unitary_matrix_3qubits_10control,
    UCRZ_unitary_matrix_4qubits_023control,
    UCRZ_unitary_matrix_4qubits_213control,
    UC_unitary_matrix_no_diagonal_no_simplification_3qubits_01control_HXHX,
    UC_unitary_matrix_no_diagonal_no_simplification_3qubits_10control_HYHY,
    UC_unitary_matrix_no_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY,
    UC_unitary_matrix_no_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY,
    UC_unitary_matrix_diagonal_no_simplification_3qubits_01control_HXHX,
    UC_unitary_matrix_diagonal_no_simplification_3qubits_10control_HYHY,
    UC_unitary_matrix_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY,
    UC_unitary_matrix_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY,
    UC_unitary_matrix_no_diagonal_simplification_3qubits_01control_HXHX,
    UC_unitary_matrix_no_diagonal_simplification_3qubits_10control_HYHY,
    UC_unitary_matrix_no_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY,
    UC_unitary_matrix_no_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY,
    UC_unitary_matrix_diagonal_simplification_3qubits_01control_HXHX,
    UC_unitary_matrix_diagonal_simplification_3qubits_10control_HYHY,
    UC_unitary_matrix_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY,
    UC_unitary_matrix_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY
)


class TestUniformlyControlledGates:
    """ `tests.circuit.TestUniformlyControlledGates` is the class for testing
    uniformly controlled gates.

    These gates are:
    - `circuit.UCRX()`
    - `circuit.UCRY()`
    - `circuit.UCRZ()`
    - `circuit.Diagonal()`
    - `circuit.UC()`
    """
    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("angles, control_indices, target_index, expected", [
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5],
            [0, 1],
            2,
            UCRX_unitary_matrix_3qubits_01control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5],
            [1, 0],
            2,
            UCRX_unitary_matrix_3qubits_10control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9],
            [0, 2, 3],
            1,
            UCRX_unitary_matrix_4qubits_023control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9],
            [2, 1, 3],
            0,
            UCRX_unitary_matrix_4qubits_213control
        ]
    ])
    def test_UCRX(
            self,
            circuit_framework: Type[Circuit],
            angles: list[float],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UCRX` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `angles` : list[float]
            The angles of the rotation.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UCRX gate
        circuit.UCRX(angles, control_indices, target_index)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, decimal=8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("angles, control_indices, target_index, expected", [
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5],
            [0, 1],
            2,
            UCRY_unitary_matrix_3qubits_01control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5],
            [1, 0],
            2,
            UCRY_unitary_matrix_3qubits_10control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9],
            [0, 2, 3],
            1,
            UCRY_unitary_matrix_4qubits_023control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9],
            [2, 1, 3],
            0,
            UCRY_unitary_matrix_4qubits_213control
        ]
    ])
    def test_UCRY(
            self,
            circuit_framework: Type[Circuit],
            angles: list[float],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UCRY` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `angles` : list[float]
            The angles of the rotation.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UCRY gate
        circuit.UCRY(angles, control_indices, target_index)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, decimal=8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("angles, control_indices, target_index, expected", [
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5],
            [0, 1],
            2,
            UCRZ_unitary_matrix_3qubits_01control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5],
            [1, 0],
            2,
            UCRZ_unitary_matrix_3qubits_10control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9],
            [0, 2, 3],
            1,
            UCRZ_unitary_matrix_4qubits_023control
        ],
        [
            [np.pi/2, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, np.pi/8, np.pi/9],
            [2, 1, 3],
            0,
            UCRZ_unitary_matrix_4qubits_213control
        ]
    ])
    def test_UCRZ(
            self,
            circuit_framework: Type[Circuit],
            angles: list[float],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UCRZ` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `angles` : list[float]
            The angles of the rotation.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UCRZ gate
        circuit.UCRZ(angles, control_indices, target_index)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, decimal=8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("diagonal, qubit_indices", [
        [
            [1, 1, 1, -1, 1, -1, 1, -1],
            [0, 1, 2],
        ],
        [
            [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1],
            [0, 1, 2, 3],
        ]
    ])
    def test_Diagonal(
            self,
            circuit_framework: Type[Circuit],
            diagonal: list[int],
            qubit_indices: list[int]
        ) -> None:
        """ Test the `Diagonal` gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `diagonal` : list[int]
            The diagonal of the matrix.
        `qubit_indices` : list[int]
            The qubit indices.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(qubit_indices))

        # Apply the Diagonal gate
        circuit.Diagonal(np.array(diagonal), qubit_indices)

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), np.diag(diagonal).astype(complex), decimal=8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("single_qubit_gates, control_indices, target_index, expected", [
        [
            [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix],
            [0, 1],
            2,
            UC_unitary_matrix_no_diagonal_no_simplification_3qubits_01control_HXHX
        ],
        [
            [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix],
            [1, 0],
            2,
            UC_unitary_matrix_no_diagonal_no_simplification_3qubits_10control_HYHY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [0, 2, 3],
            1,
            UC_unitary_matrix_no_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [2, 1, 3],
            0,
            UC_unitary_matrix_no_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY
        ]
    ])
    def test_UC_no_diagonal_no_simplification(
            self,
            circuit_framework: Type[Circuit],
            single_qubit_gates: list[NDArray[np.complex128]],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UC` gate without diagonal and without simplification.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `single_qubit_gates` : list[NDArray[np.complex128]]
            The single-qubit gates.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UC gate
        circuit.UC(
            single_qubit_gates,
            control_indices,
            target_index,
            up_to_diagonal=False,
            multiplexor_simplification=False
        )

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("single_qubit_gates, control_indices, target_index, expected", [
        [
            [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix],
            [0, 1],
            2,
            UC_unitary_matrix_diagonal_no_simplification_3qubits_01control_HXHX
        ],
        [
            [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix],
            [1, 0],
            2,
            UC_unitary_matrix_diagonal_no_simplification_3qubits_10control_HYHY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [0, 2, 3],
            1,
            UC_unitary_matrix_diagonal_no_simplification_4qubits_023control_RXRYRXRYRXRY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [2, 1, 3],
            0,
            UC_unitary_matrix_diagonal_no_simplification_4qubits_213control_RXRYRXRYRXRY
        ]
    ])
    def test_UC_diagonal_no_simplification(
            self,
            circuit_framework: Type[Circuit],
            single_qubit_gates: list[NDArray[np.complex128]],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UC` gate with diagonal and without simplification.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `single_qubit_gates` : list[NDArray[np.complex128]]
            The single-qubit gates.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UC gate
        circuit.UC(
            single_qubit_gates,
            control_indices,
            target_index,
            up_to_diagonal=True,
            multiplexor_simplification=False
        )

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("single_qubit_gates, control_indices, target_index, expected", [
        [
            [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix],
            [0, 1],
            2,
            UC_unitary_matrix_no_diagonal_simplification_3qubits_01control_HXHX
        ],
        [
            [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix],
            [1, 0],
            2,
            UC_unitary_matrix_no_diagonal_simplification_3qubits_10control_HYHY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [0, 2, 3],
            1,
            UC_unitary_matrix_no_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [2, 1, 3],
            0,
            UC_unitary_matrix_no_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY
        ]
    ])
    def test_UC_no_diagonal_simplification(
            self,
            circuit_framework: Type[Circuit],
            single_qubit_gates: list[NDArray[np.complex128]],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UC` gate without diagonal and with simplification.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `single_qubit_gates` : list[NDArray[np.complex128]]
            The single-qubit gates.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UC gate
        circuit.UC(
            single_qubit_gates,
            control_indices,
            target_index,
            up_to_diagonal=False,
            multiplexor_simplification=True
        )

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("single_qubit_gates, control_indices, target_index, expected", [
        [
            [Hadamard().matrix, PauliX().matrix, Hadamard().matrix, PauliX().matrix],
            [0, 1],
            2,
            UC_unitary_matrix_diagonal_simplification_3qubits_01control_HXHX
        ],
        [
            [Hadamard().matrix, PauliY().matrix, Hadamard().matrix, PauliY().matrix],
            [1, 0],
            2,
            UC_unitary_matrix_diagonal_simplification_3qubits_10control_HYHY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [0, 2, 3],
            1,
            UC_unitary_matrix_diagonal_simplification_4qubits_023control_RXRYRXRYRXRY
        ],
        [
            [
                RX(np.pi/2).matrix,
                RY(np.pi/3).matrix,
                RX(np.pi/4).matrix,
                RY(np.pi/5).matrix,
                RX(np.pi/6).matrix,
                RY(np.pi/7).matrix,
                RX(np.pi/8).matrix,
                RY(np.pi/9).matrix
            ],
            [2, 1, 3],
            0,
            UC_unitary_matrix_diagonal_simplification_4qubits_213control_RXRYRXRYRXRY
        ]
    ])
    def test_UC_diagonal_simplification(
            self,
            circuit_framework: Type[Circuit],
            single_qubit_gates: list[NDArray[np.complex128]],
            control_indices: list[int],
            target_index: int,
            expected: NDArray[np.complex128]
        ) -> None:
        """ Test the `UC` gate with diagonal and simplification.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        `single_qubit_gates` : list[NDArray[np.complex128]]
            The single-qubit gates.
        `control_indices` : list[int]
            The control qubits.
        `target_index` : int
            The target qubit.
        `expected` : NDArray[np.complex128]
            The expected unitary matrix.
        """
        # Define the quantum circuit
        circuit = circuit_framework(len(control_indices) + 1)

        # Apply the UC gate
        circuit.UC(
            single_qubit_gates,
            control_indices,
            target_index,
            up_to_diagonal=True,
            multiplexor_simplification=True
        )

        # Ensure the unitary matrix is correct
        assert_almost_equal(circuit.get_unitary(), expected, 8)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UCPauliRot_invalid_num_angles(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the number of angles is invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.UCRX([np.pi/2], [1, 0], 2)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UCPauliRot_invalid_num_controls(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the number of control qubits is invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.UCRX([np.pi/2, np.pi/3, np.pi/4, np.pi/5], [0], 2)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UCPauliRot_invalid_rot_axis(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the rotation axis is invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.UCPauliRot([np.pi/2, np.pi/3, np.pi/4, np.pi/5], "invalid", [1, 0], 2) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_Diagonal_invalid_diagonal(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the diagonal is invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.Diagonal(np.array([1, 1, 1, -1, 1, -1, 1]), [0, 1, 2])

        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.Diagonal(np.array([1, 1, 1, -1, 1, -1, 1, 0]), [0, 1, 2])

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_Diagonal_invalid_number_of_qubits(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the number of qubits is invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.Diagonal(np.array([1, 1, 1, -1, 1, -1, 1, -1]), [0, 1])

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UC_invalid_single_qubit_gate(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the single-qubit gates are invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        # Invalid dimension
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.UC([np.array([[1, 0, 0], [0, 0, 1]])], [0, 1], 2)

        # Invalid number of single-qubit gates
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.UC([np.eye(2), np.eye(2), np.eye(2)], [0, 1], 2) # type: ignore

        # Non-unitary matrix
        with pytest.raises(ValueError):
            circuit = circuit_framework(2)
            circuit.UC([np.array([[1, 0], [0, 1]]), np.array([[1, 0], [1, 1]])], [0], 1)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UC_invalid_number_of_qubits(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test the case when the number of qubits is invalid.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The quantum circuit framework.
        """
        with pytest.raises(ValueError):
            circuit = circuit_framework(3)
            circuit.UC([np.eye(2), np.eye(2)], [0, 1], 2) # type: ignore