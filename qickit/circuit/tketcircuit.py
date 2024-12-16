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

""" Wrapper class for using Quantinuum's TKET in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["TKETCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING

from pytket import Circuit as TKCircuit
from pytket import OpType
from pytket.circuit import Op, QControlBox
from pytket.extensions.qiskit import AerBackend, AerStateBackend

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit
from qickit.circuit.circuit import GATES


class TKETCircuit(Circuit):
    """ `qickit.circuit.TKETCircuit` is the wrapper for using Quantinuum's TKET in Qickit SDK.

    Notes
    -----
    Quantinuum's TKET is a quantum software development kit for building, compiling,
    and simulating quantum circuits.

    For more information on the TKET:
    - Documentation:
    https://tket.quantinuum.com/api-docs/
    - Source Code:
    https://github.com/CQCL/tket
    - Publication:
    https://arxiv.org/pdf/2003.10611

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `circuit` : pytket.Circuit
        The TKET circuit.
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
    >>> circuit = TKETCircuit(num_qubits=2)
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:

        super().__init__(num_qubits=num_qubits)

        self.circuit: TKCircuit = TKCircuit(self.num_qubits, self.num_qubits)

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
            "I": const((OpType.noop,)),
            "X": const((OpType.X,)),
            "Y": const((OpType.Y,)),
            "Z": const((OpType.Z,)),
            "H": const((OpType.H,)),
            "S": const((OpType.S,)),
            "Sdg": const((OpType.Sdg,)),
            "T": const((OpType.T,)),
            "Tdg": const((OpType.Tdg,)),
            "RX": lambda angles: (OpType.Rx, angles[0]/np.pi),
            "RY": lambda angles: (OpType.Ry, angles[0]/np.pi),
            "RZ": lambda angles: (OpType.Rz, angles[0]/np.pi),
            "Phase": lambda angles: (OpType.U1, angles[0]/np.pi),
            "U3": lambda angles: (OpType.U3, [angles[i]/np.pi for i in range(3)])
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
            gate_operation = QControlBox(
                Op.create(*gate_operation),
                len(control_indices)
            )

            for target_index in target_indices:
                self.circuit.add_qcontrolbox(gate_operation, [*control_indices[:], target_index]) # type: ignore
            return

        for target_index in target_indices:
            self.circuit.add_gate(*gate_operation, [target_index]) # type: ignore

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate, and apply it to the circuit
        self.circuit.add_phase(angle/np.pi)
        self.global_phase += angle

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Measure the qubits
        for index in qubit_indices:
            self.circuit.Measure(index, index)
            self.measured_qubits.add(index)

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:

        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = self.copy() # type: ignore

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            base_backend = AerStateBackend()
            circuit = base_backend.get_compiled_circuits([circuit.circuit]) # type: ignore
            state_vector = base_backend.run_circuit(circuit[0]).get_state() # type: ignore
        else:
            state_vector = backend.get_statevector(circuit)

        return np.array(state_vector)

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:

        num_qubits_to_measure = len(self.measured_qubits)

        if num_qubits_to_measure == 0:
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = self.copy() # type: ignore

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            # If no backend is provided, use the AerBackend
            base_backend = AerBackend()
            compiled_circuit = base_backend.get_compiled_circuits([circuit.circuit]) # type: ignore
            result = base_backend.run_circuit(compiled_circuit[0], n_shots=num_shots, seed=0) # type: ignore

            # Extract the counts from the result
            counts = {"".join(map(str, basis_state)): num_counts
                      for basis_state, num_counts in result.get_counts().items()}

            partial_counts = {}

            # Parse the binary strings to filter out the unmeasured qubits
            for key in counts.keys():
                new_key = ''.join(key[i] for i in range(len(key)) if i in circuit.measured_qubits)
                partial_counts[new_key] = counts[key]

            counts = partial_counts

            # Fill the counts array with zeros for the missing states
            counts = {
                f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) \
                for i in range(2**num_qubits_to_measure)
            }

        else:
            counts = backend.get_counts(circuit=circuit, num_shots=num_shots)

        return counts

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = self.copy() # type: ignore

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Run the circuit and define the unitary matrix
        unitary = circuit.circuit.get_unitary()

        return np.array(unitary)

    def reset_qubit(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.reset_qubit.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        for qubit_index in qubit_indices:
            self.circuit.Reset(qubit_index)

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        from qickit.circuit import QiskitCircuit

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        pass