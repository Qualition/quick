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

""" Wrapper class for using the Qiskit transpiler to prepare
quantum unitary operators in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["QiskitUnitaryTranspiler"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import SupportsIndex, Type, TYPE_CHECKING

from qiskit import QuantumCircuit, transpile # type: ignore
from qiskit.transpiler.passes import unitary_synthesis_plugin_names # type: ignore
from qiskit_ibm_runtime import QiskitRuntimeService # type: ignore
from qiskit_transpiler_service.transpiler_service import TranspilerService # type: ignore

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.primitives import Operator
from qickit.synthesis.unitarypreparation import UnitaryPreparation


class QiskitUnitaryTranspiler(UnitaryPreparation):
    """ `qickit.QiskitUnitaryTranspiler` is the class for preparing quantum operators using Qiskit transpiler.

    Notes
    -----
    The `qiskit.transpiler` library is a quantum circuit optimization library developed by IBM Quantum
    to optimize, route, and transpile quantum circuits given a set of constraints. The implementation
    utilizes LightSABRE approach, and by default, Shende's Shannon Decomposition for unitary synthesis.

    For more information on Qiskit transpiler:
    - Documentation:
    https://qiskit.org/documentation/apidoc/transpiler.html.
    - Source code:
    https://github.com/Qiskit/qiskit/tree/main/qiskit/transpiler
    - Publication:
    https://arxiv.org/pdf/2409.08368

    Parameters
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `ai_transpilation` : bool, optional, default=False
        Whether to use Qiskit's AI transpiler.
    `unitary_synthesis_plugin` : str, optional, default="default"
        The unitary synthesis plugin to use for preparing quantum unitary
        operators. The available plugins are:
        - "default": Shende's Shannon Decomposition.
        - "sk": Solovay-Kitaev.
        - "aqc": Approximate Quantum Compilation.
    `service`: qiskit_ibm_runtime.QiskitRuntimeService, optional
        The Qiskit Runtime service. Only needed if `ai`=True.
    `backend_name`: str, qiskit.primitives.backend.Backend, optional
        The backend to use for transpilation. For AI transpilation, the name
        of the backend must be provided.

    Attributes
    ----------
    `output_framework` : type[qickit.circuit.Circuit]
        The quantum circuit framework.
    `ai_transpilation` : bool
        Whether to use Qiskit's AI transpiler.
    `unitary_synthesis_plugin` : str
        The unitary synthesis plugin to use for preparing quantum unitary
        operators.
    `service`: qiskit_ibm_runtime.QiskitRuntimeService
        The Qiskit Runtime service.
    `backend_name`: str | None
        The name of the backend to use for transpilation.

    Raises
    ------
    TypeError
        - If the output framework is not a subclass of `qickit.circuit.Circuit`.
    ValueError
        - The Qiskit Runtime service must be provided for AI transpilation.
        - The name of the backend must be provided for AI transpilation.
        - Invalid unitary synthesis plugin.
    """
    def __init__(
            self,
            output_framework: Type[Circuit],
            ai_transpilation: bool=False,
            unitary_synthesis_plugin: str="default",
            service: QiskitRuntimeService | None = None,
            backend_name: str | None = None
        ) -> None:

        super().__init__(output_framework)
        self.ai_transpilation = ai_transpilation
        self.backend_name = None

        if ai_transpilation:
            if service is None:
                raise ValueError("The Qiskit Runtime service must be provided for AI transpilation.")
            match backend_name:
                case None:
                    raise ValueError("The name of the backend must be provided for AI transpilation.")
                case str():
                    supported_backends = TranspilerService(optimization_level=3)\
                        .transpiler_service.get_supported_backends()

                    if backend_name not in supported_backends:
                        raise ValueError(f"Invalid backend: {backend_name}.")
                    self.backend_name = backend_name

        if unitary_synthesis_plugin not in unitary_synthesis_plugin_names():
            raise ValueError(f"Invalid unitary synthesis plugin: {unitary_synthesis_plugin}.")
        self.unitary_synthesis_plugin = unitary_synthesis_plugin

    def apply_unitary(
            self,
            circuit: Circuit,
            unitary: NDArray[np.complex128] | Operator,
            qubit_indices: int | Sequence[int]
        ) -> Circuit:

        if not isinstance(unitary, (np.ndarray, Operator)):
            try:
                unitary = np.array(unitary).astype(complex)
            except (ValueError, TypeError):
                raise TypeError(f"The operator must be a numpy array or an Operator object. Received {type(unitary)} instead.")

        if isinstance(unitary, np.ndarray):
            unitary = Operator(unitary)

        if isinstance(qubit_indices, SupportsIndex):
            qubit_indices = [qubit_indices]

        if not all(isinstance(qubit_index, SupportsIndex) for qubit_index in qubit_indices):
            raise TypeError("All qubit indices must be integers.")

        if not len(qubit_indices) == unitary.num_qubits:
            raise ValueError("The number of qubit indices must match the number of qubits in the unitary.")

        # Get the number of qubits needed to implement the operator
        num_qubits = unitary.num_qubits

        # Create a qiskit circuit
        qiskit_circuit = QuantumCircuit(num_qubits, num_qubits)

        # Apply the unitary matrix to the circuit
        qiskit_circuit.unitary(unitary.data, range(num_qubits))

        # Transpile the unitary operator to a series of CX and U3 gates
        if self.ai_transpilation:
            transpile_params = {
                "basis_gates": ["u3", "cx"],
                "unitary_synthesis_method": self.unitary_synthesis_plugin,
            }
            # Use the Qiskit AI transpiler
            ai_transpiler = TranspilerService(
                backend_name=self.backend_name,
                optimization_level=3,
                qiskit_transpile_options=transpile_params,
                ai="true",
                ai_layout_mode="OPTIMIZE"
            )
            transpiled_circuit = ai_transpiler.run(qiskit_circuit)
        else:
            # Use the Qiskit SDK transpiler
            transpiled_circuit = transpile(
                qiskit_circuit,
                unitary_synthesis_method=self.unitary_synthesis_plugin,
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