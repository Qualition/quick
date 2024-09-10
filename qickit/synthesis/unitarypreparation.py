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
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Type, TYPE_CHECKING, SupportsIndex

from qiskit import QuantumCircuit, transpile # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
from qiskit_transpiler_service.transpiler_service import TranspilerService # type: ignore

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Operator


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
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:
        """ Initalize a Unitary Preparation instance.
        """
        # Define the QC framework
        self.output_framework = output_framework

    @abstractmethod
    def prepare_unitary(
            self,
            unitary: NDArray[np.complex128] | Operator
        ) -> Circuit:
        """ Prepare the quantum unitary operator.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The quantum unitary operator.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit for preparing the unitary operator.
        """

    @abstractmethod
    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:
        """ Apply the quantum unitary operator to a quantum circuit.

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit.
        `unitary` : NDArray[np.complex128] | qickit.primitives.Operator
            The quantum unitary operator.
        `qubit_indices` : int | Sequence[int]
            The qubit indices to apply the unitary operator to.

        Returns
        -------
        `circuit` : qickit.circuit.Circuit
            The quantum circuit with the unitary operator applied.
        """


class QiskitUnitaryTranspiler(UnitaryPreparation):
    """ `qickit.QiskitUnitaryTranspiler` is the class for preparing quantum operators using Qiskit transpiler.

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `ai_transpilation` : bool, optional, default=False
        Whether to use Qiskit's AI transpiler.
    `service`: qiskit_ibm_runtime.QiskitRuntimeService, optional
        The Qiskit Runtime service. Only needed if `ai`=True.
    `backend_name`: str, optional
        The name of the backend to use for transpilation. Only needed if `ai`=True.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `ai_transpilation` : bool
        Whether to use Qiskit's AI transpiler.
    `service`: qiskit_ibm_runtime.QiskitRuntimeService
        The Qiskit Runtime service.
    `backend_name`: str
        The name of the backend to use for transpilation.

    Raises
    ------
    ValueError
        The Qiskit Runtime service must be provided for AI transpilation.
        The name of the backend must be provided for AI transpilation.

    Notes
    -----
    The `QiskitTranspiler` class uses the Qiskit transpiler to prepare the quantum unitary operator.
    This is also how `qickit.circuit.Circuit` by default implements the `.unitary()` operation. Users
    can customize this class to implement custom transpilers for different hardware. For example, one
    can change the set of gates the circuit is transpiled to, or can change the optimization level.

    A good resource is IBM Quantum Challenge 2024: https://github.com/qiskit-community/ibm-quantum-challenge-2024/tree/main
    """
    def __init__(
            self,
            output_framework: Type[Circuit],
            ai_transpilation: bool=False,
            service: QiskitRuntimeService | None = None,
            backend_name: str | None = None
        ) -> None:

        super().__init__(output_framework)
        self.ai_transpilation = ai_transpilation

        if ai_transpilation and service is None:
            raise ValueError("The Qiskit Runtime service must be provided for AI transpilation.")
        self.service = service

        if ai_transpilation and backend_name is None:
            raise ValueError("The name of the backend must be provided for AI transpilation.")
        self.backend_name = backend_name

    def prepare_unitary(
            self,
            unitary: NDArray[np.complex128] | Operator
        ) -> Circuit:

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        # Get the number of qubits needed to implement the operator
        num_qubits = unitary.num_qubits

        # Initialize the qickit circuit
        circuit = self.output_framework(num_qubits)

        # Apply the unitary matrix to the circuit
        # and return the circuit
        return self.apply_unitary(circuit, unitary, range(num_qubits))

    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        # Get the number of qubits needed to implement the operator
        num_qubits = unitary.num_qubits

        # Create a qiskit circuit
        qiskit_circuit = QuantumCircuit(num_qubits, num_qubits)

        # Apply the unitary matrix to the circuit
        qiskit_circuit.unitary(unitary.data, range(num_qubits))

        # Transpile the unitary operator to a series of CX and U3 gates
        if self.ai_transpilation:
            # Use the Qiskit AI transpiler
            ai_transpiler = TranspilerService(
                backend_name=self.backend_name,
                optimization_level=3,
                ai="true",
                ai_layout_mode="OPTIMIZE"
            )
            transpiled_circuit = ai_transpiler.run(qiskit_circuit)

        else:
            # Use the Qiskit transpiler
            transpiled_circuit = transpile(
                qiskit_circuit,
                basis_gates=["u3", "cx"],
                optimization_level=3,
                seed_transpiler=0
            )

        # Apply the U3 and CX gates to the qickit circuit
        for gate in transpiled_circuit.data: # type: ignore
            if gate.operation.name in ["u", "u3"]:
                circuit.U3(gate.operation.params, qubit_indices[gate.qubits[0]._index])
            else:
                circuit.CX(qubit_indices[gate.qubits[0]._index], qubit_indices[gate.qubits[1]._index])

        # Update the global phase
        circuit.GlobalPhase(transpiled_circuit.global_phase) # type: ignore

        return circuit