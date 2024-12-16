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

""" Wrapper class for using IBM Qiskit in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["QiskitCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister # type: ignore
from qiskit.circuit.library import ( # type: ignore
    RXGate, RYGate, RZGate, HGate, XGate, YGate, # type: ignore
    ZGate, SGate, SdgGate, TGate, TdgGate, U3Gate, # type: ignore
    PhaseGate, IGate # type: ignore
)
from qiskit.primitives import BackendSamplerV2 as BackendSampler # type: ignore
from qiskit_aer import AerSimulator # type: ignore
import qiskit.qasm2 as qasm2 # type: ignore
import qiskit.qasm3 as qasm3 # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit
from qickit.circuit.circuit import GATES


class QiskitCircuit(Circuit):
    """ `qickit.circuit.QiskitCircuit` is the wrapper for using IBM Qiskit in Qickit SDK.

    Notes
    -----
    IBM Qiskit is an open-source SDK for working with quantum computers at the level of
    extended quantum circuits, operators, and primitives.

    For more information on IBM Qiskit:
    - Documentation:
    https://qiskit.org/documentation/
    - Source code:
    https://github.com/Qiskit/qiskit
    - Publication:
    https://arxiv.org/pdf/2405.08810

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `circuit` : qiskit.QuantumCircuit
        The circuit.
    `gate_mapping` : dict[str, Callable]
        The mapping of the gates in the input quantum computing
        framework to the gates in qickit.
    `measured_qubits` : set[int]
        The set of measured qubits indices.
    `circuit_log` : list[dict]
        The circuit log.
    `global_phase` : float
        The global phase of the circuit.
    `process_gate_params_flag` : bool
        The flag to process the gate parameters.

    Raises
    ------
    TypeError
        - Number of qubits bits must be integers.
    ValueError
        - Number of qubits bits must be greater than 0.

    Usage
    -----
    >>> circuit = QiskitCircuit(num_qubits=2)
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:

        super().__init__(num_qubits=num_qubits)

        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        self.circuit: QuantumCircuit = QuantumCircuit(qr, cr)
        self.gate_mapping = self._define_gate_mapping()

    @staticmethod
    def _define_gate_mapping() -> dict[str, Callable]:
        # Define lambda factory for non-parameterized gates
        def const(x):
            return lambda _angles: x

        # Note that Qickit only uses U3, CX, and Global Phase gates and constructs the other gates
        # by performing decomposition
        # However, if the user wants to override the decomposition and use the native gates, they
        # can do so by using the below gate mapping
        gate_mapping = {
            "I": const(IGate()),
            "X": const(XGate()),
            "Y": const(YGate()),
            "Z": const(ZGate()),
            "H": const(HGate()),
            "S": const(SGate()),
            "Sdg": const(SdgGate()),
            "T": const(TGate()),
            "Tdg": const(TdgGate()),
            "RX": lambda angles: RXGate(angles[0]),
            "RY": lambda angles: RYGate(angles[0]),
            "RZ": lambda angles: RZGate(angles[0]),
            "Phase": lambda angles: PhaseGate(angles[0]),
            "U3": lambda angles: U3Gate(theta=angles[0], phi=angles[1], lam=angles[2])
        }

        return gate_mapping

    def _gate_mapping(
            self,
            gate: GATES,
            target_indices: int | Sequence[int],
            control_indices: int | Sequence[int] = [],
            angles: Sequence[float] = [0, 0, 0]
        ) -> None:

        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        gate_operation = self.gate_mapping[gate](angles)

        if control_indices:
            for target_index in target_indices:
                self.circuit.append(gate_operation.control(len(control_indices)), [*control_indices[:], target_index])
            return

        for target_index in target_indices:
            self.circuit.append(gate_operation, [target_index])

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate
        self.circuit.global_phase += angle
        self.global_phase += angle

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        self.circuit.measure(qubit_indices, qubit_indices)

        # Set the measurement as applied
        for qubit_index in qubit_indices:
            self.measured_qubits.add(qubit_index)

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:

        if backend is None:
            state_vector = Statevector(self.circuit).data
        else:
            state_vector = backend.get_statevector(self)

        return np.array(state_vector)

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:

        num_qubits_to_measure = len(self.measured_qubits)

        if len(self.measured_qubits) == 0:
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = self.copy() # type: ignore

        if backend is None:
            # Transpile the circuit to the backend
            # This is to counter https://github.com/Qiskit/qiskit/issues/13162
            circuit.transpile()

            # If no backend is provided, use the AerSimualtor
            base_backend: BackendSampler = BackendSampler(backend=AerSimulator())
            result = base_backend.run([circuit.circuit], shots=num_shots).result()

            # Extract the counts from the result
            counts = result[0].join_data().get_counts() # type: ignore

            partial_counts = {}

            # Parse the binary strings to filter out the unmeasured qubits
            for key in counts.keys():
                new_key = ''.join(key[::-1][i] for i in range(len(key)) if i in circuit.measured_qubits)
                partial_counts[new_key[::-1]] = counts[key]

            counts = partial_counts

            # Fill the counts array with zeros for the missing states
            counts = {
                f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) \
                for i in range(2**num_qubits_to_measure)
            }

            # Sort the counts by their keys (basis states)
            counts = dict(sorted(counts.items()))

        else:
            counts = backend.get_counts(circuit=circuit, num_shots=num_shots)

        return counts

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = self.copy() # type: ignore

        # Get the unitary matrix of the circuit
        unitary = Operator(circuit.circuit).data

        return np.array(unitary)

    def reset_qubit(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.reset_qubit.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        for qubit_index in qubit_indices:
            self.circuit.reset(qubit_index)

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        if qasm_version == 2:
            return qasm2.dumps(self.circuit)
        elif qasm_version == 3:
            return qasm3.dumps(self.circuit)
        else:
            raise ValueError("The QASM version must be either 2 or 3.")

    def draw(self) -> None:
        self.circuit.draw(output="mpl")