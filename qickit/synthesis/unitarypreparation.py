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

__all__ = ["UnitaryPreparation", "QiskitUnitaryTranspiler"]

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from numpy.typing import NDArray
from typing import Type, TYPE_CHECKING

# Qiskit imports
from qiskit import QuantumCircuit, transpile # type: ignore

# Import `qickit.circuit.Circuit` instances
if TYPE_CHECKING:
    from qickit.circuit import Circuit

# Import `qickit.types.collection.Collection`
from qickit.types import Collection


class UnitaryPreparation(ABC):
    """ `qickit.UnitaryPreparation` is the class for preparing quantum operators.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    """
    def __init__(self,
                 output_framework: Type[Circuit]) -> None:
        """ Initalize a Unitary Preparation instance.
        """
        # Define the QC framework
        self.output_framework = output_framework

    @staticmethod
    def check_unitary(unitary: NDArray[np.complex128]) -> None:
        """ Check if the input matrix is a valid unitary matrix.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128]
            The quantum unitary operator.

        Raises
        ------
        ValueError
            Input matrix is not a unitary matrix.
        """
        if not np.allclose(np.eye(unitary.shape[0]), unitary.conj().T @ unitary):
            raise ValueError("Input matrix is not a unitary matrix.")

    @staticmethod
    def check_unitary_size(unitary: NDArray[np.complex128]) -> None:
        """ Check if the unitary matrix is the correct size.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128]
            The quantum unitary operator.

        Raises
        ------
        ValueError
            The `unitary_matrix` must have a size of 2^N x 2^N.
        """
        # Calculate the number of qubits needed to implement the operator
        num_qubits = int(np.log2(unitary.shape[0]))

        # Define the size of the operation for checking
        size = 2 ** num_qubits

        # Check if the unitary matrix is the correct size
        if not (len(unitary) == size and len(unitary[0]) == size):
            raise ValueError(f"The `unitary_matrix` must have a size of {size} x {size}.")

    @staticmethod
    def check_num_qubits(num_qubits: int,
                         unitary: NDArray[np.complex128]) -> None:
        """ Check if the number of qubits is correct for the unitary matrix.

        Parameters
        ----------
        `num_qubits` : int
            The number of qubits passed to implement the operator.
        `unitary` : NDArray[np.complex128]
            The quantum unitary operator.

        Raises
        ------
        ValueError
            The number of qubits is not correct for the unitary matrix.
        """
        # Calculate the number of qubits needed to implement the operator
        num_qubits_needed = int(np.log2(unitary.shape[0]))

        # Check if the number of qubits is correct for the unitary matrix
        if not num_qubits == num_qubits_needed:
            raise ValueError(f"The number of qubits must be {num_qubits_needed}.")

    @staticmethod
    def unitarymethod(method):
        """ Decorator for unitary methods.

        Parameters
        ----------
        `method` : Callable
            The method to decorate.

        Returns
        -------
        `wrapper` : Callable
            The decorated method.
        """
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            # Check if the input matrix is a valid unitary matrix
            instance.check_unitary(args[0])

            # Check if the unitary matrix is the correct size
            instance.check_unitary_size(args[0])

            # Check if the number of qubits is correct for the unitary matrix
            instance.check_num_qubits(len(args[1]), args[0])

            return method(instance, *args, **kwargs)

        return wrapper

    @abstractmethod
    def prepare_unitary(self,
                        unitary: NDArray[np.complex128],
                        qubit_indices: int | Collection[int]) -> Circuit:
        """ Prepare the quantum unitary operator.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128]
            The quantum unitary operator.
        `qubit_indices` : int | Collection[int]
            The index of the qubit(s) to apply the gate to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit for preparing the unitary operator.
        """


class QiskitUnitaryTranspiler(UnitaryPreparation):
    """ `qickit.QiskitUnitaryTranspiler` is the class for preparing quantum operators using Qiskit transpiler.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.

    Notes
    -----
    The `QiskitTranspiler` class uses the Qiskit transpiler to prepare the quantum unitary operator.
    This is also how `qickit.circuit.Circuit` by default implements the `.unitary()` operation. Users
    can customize this class to implement custom transpilers for different hardware. For example, one
    can change the set of gates the circuit is transpiled to, or can change the optimization level.
    """
    def __init__(self,
                 output_framework: Type[Circuit]) -> None:
        """ Initalize a Qiskit Transpiler instance.
        """
        super().__init__(output_framework)

    @UnitaryPreparation.unitarymethod
    def prepare_unitary(self,
                        unitary: NDArray[np.complex128],
                        qubit_indices: int | Collection[int]) -> Circuit:
        # Convert the qubit indices to a list if it is a range
        if isinstance(qubit_indices, range):
            qubit_indices = list(qubit_indices)

        # Get the number of qubits needed to implement the operator
        num_qubits = len(qubit_indices) if isinstance(qubit_indices, Collection) else 1

        # Create a qiskit circuit
        qiskit_circuit = QuantumCircuit(num_qubits, num_qubits)

        # Initialize the qickit circuit
        circuit = self.output_framework(num_qubits, num_qubits)

        # Apply the unitary matrix to the circuit
        qiskit_circuit.unitary(unitary, range(num_qubits))

        # Transpile the unitary operator to a series of CX and U3 gates
        transpiled_circuit = transpile(qiskit_circuit,
                                       basis_gates=["u3", "cx"],
                                       optimization_level=3,
                                       seed_transpiler=0)

        # Iterate over the gates in the transpiled circuit
        for gate in transpiled_circuit.data:
            # Add the U3 gate
            if gate[0].name in ["u", "u3"]:
                circuit.U3(gate[0].params, gate[1][0]._index)

            # Add the CX gate
            else:
                circuit.CX(gate[1][0]._index, gate[1][1]._index)

        # Update the global phase
        circuit.GlobalPhase(transpiled_circuit.global_phase)

        return circuit