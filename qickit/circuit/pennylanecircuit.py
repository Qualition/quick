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

""" Wrapper class for using Xanadu's PennyLane in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["PennylaneCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING

import pennylane as qml # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit
from qickit.circuit.circuit import GATES


class PennylaneCircuit(Circuit):
    """ `qickit.circuit.PennylaneCircuit` is the wrapper for using Xanadu's PennyLane in Qickit SDK.

    Notes
    -----
    Xanadu's PennyLane is a cross-platform Python library for quantum computing,
    quantum machine learning, and quantum chemistry.

    For more information on PennyLane:
    - Documentation:
    https://docs.pennylane.ai/en/stable/code/qml.html
    - Source Code:
    https://github.com/PennyLaneAI/pennylane
    - Publication:
    https://arxiv.org/pdf/1811.04968

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `circuit` : list[qml.Operation]
        The circuit.
    `gate_mapping` : dict[str, Callable]
        The mapping of the gates in the input quantum computing
        framework to the gates in qickit.
    `device` : qml.Device
        The PennyLane device to use.
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
    >>> circuit = PennylaneCircuit(num_qubits=2)
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:

        super().__init__(num_qubits=num_qubits)

        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit: list[qml.Operation] = []

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
            "I": const(qml.Identity(0).matrix()),
            "X": const(qml.PauliX(0).matrix()),
            "Y": const(qml.PauliY(0).matrix()),
            "Z": const(qml.PauliZ(0).matrix()),
            "H": const(qml.Hadamard(wires=0).matrix()),
            "S": const(qml.S(wires=0).matrix()),
            "Sdg": const(qml.adjoint(qml.S(0)).matrix()), # type: ignore
            "T": const(qml.T(wires=0).matrix()),
            "Tdg": const(qml.adjoint(qml.T(0)).matrix()), # type: ignore
            "RX": lambda angles: qml.RX(phi=angles[0], wires=0).matrix(), # type: ignore
            "RY": lambda angles: qml.RY(phi=angles[0], wires=0).matrix(), # type: ignore
            "RZ": lambda angles: qml.RZ(phi=angles[0], wires=0).matrix(), # type: ignore
            "Phase": lambda angles: qml.PhaseShift(phi=angles[0], wires=0).matrix(), # type: ignore
            "U3": lambda angles: qml.U3(theta=angles[0], phi=angles[1], delta=angles[2], wires=0).matrix() # type: ignore
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
        # Apply the gate operation to the specified qubits
        gate_operation = self.gate_mapping[gate](angles)

        if control_indices:
            for target_index in target_indices:
                self.circuit.append(
                qml.ControlledQubitUnitary(
                    gate_operation,
                    control_wires=control_indices,
                    wires=target_index
                )
            )
            return

        for target_index in target_indices:
            self.circuit.append(
                qml.QubitUnitary(gate_operation, wires=target_index)
            )

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate
        global_phase = qml.GlobalPhase
        self.circuit.append(global_phase(-angle))
        self.global_phase += angle

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        # In PennyLane, we apply measurements in '.get_statevector', and '.get_counts'
        # methods
        # This is due to the need for PennyLane quantum functions to return measurement results
        # Therefore, we do not need to do anything here
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Set the measurement as applied
        for qubit_index in qubit_indices:
            self.measured_qubits.add(qubit_index)
            self.circuit.append((qml.measure(qubit_index), False)) # type: ignore

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy() # type: ignore

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        def compile_circuit() -> qml.StateMP:
            """ Compile the circuit.

            Parameters
            ----------
            circuit : Collection[qml.Op]
                The list of operations representing the circuit.

            Returns
            -------
            qml.StateMP
                The state vector of the circuit.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                if isinstance(op, tuple):
                    qml.measure(op[0].wires[0], reset=op[1]) # type: ignore
                    continue

                qml.apply(op)

            return qml.state()

        if backend is None:
            state_vector = qml.QNode(compile_circuit, circuit.device)()
        else:
            state_vector = backend.get_statevector(circuit)

        return np.array(state_vector)

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:

        np.random.seed(0)

        if len(self.measured_qubits) == 0:
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy() # type: ignore

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        def compile_circuit() -> qml.CountsMp:
            """ Compile the circuit.

            Parameters
            ----------
            circuit : Collection[qml.Op]
                The list of operations representing the circuit.

            Returns
            -------
            Collection[qml.ProbabilityMP]
                The list of probability measurements.
            """
            # Apply the operations in the circuit
            for op in circuit.circuit:
                if isinstance(op, tuple):
                    qml.measure(op[0].wires[0], reset=op[1]) # type: ignore
                    continue

                qml.apply(op)

            return qml.counts(wires=circuit.measured_qubits, all_outcomes=True)

        if backend is None:
            device = qml.device(circuit.device.name, wires=circuit.num_qubits, shots=num_shots)
            result = qml.QNode(compile_circuit, device)()
            counts = {list(result.keys())[i]: int(list(result.values())[i]) for i in range(len(result))}
        else:
            result = backend.get_counts(self, num_shots=num_shots)

        return counts

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: PennylaneCircuit = self.copy() # type: ignore

        # PennyLane uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        def compile_circuit() -> None:
            """ Compile the circuit.

            Parameters
            ----------
            `circuit` : Collection[qml.Op]
                The list of operations representing the circuit.
            """
            if circuit.circuit == [] or (
                isinstance(circuit.circuit[0], qml.GlobalPhase) and len(circuit.circuit) == 1
            ):
                for i in range(circuit.num_qubits):
                    circuit.circuit.append(qml.Identity(wires=i))

            # Apply the operations in the circuit
            for op in circuit.circuit:
                if isinstance(op, tuple):
                    qml.measure(op[0].wires[0], reset=op[1]) # type: ignore
                    continue

                qml.apply(op)

        # Run the circuit and define the unitary matrix
        unitary = np.array(qml.matrix(compile_circuit, wire_order=range(self.num_qubits))(), dtype=complex) # type: ignore

        return unitary

    def reset_qubit(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.reset_qubit.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        for qubit_index in qubit_indices:
            self.circuit.append((qml.measure(qubit_index), True)) # type: ignore

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        from qickit.circuit import QiskitCircuit

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        pass