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

__all__ = ["TKETCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

from pytket import Circuit as TKCircuit
from pytket import OpType
from pytket.circuit import Op, QControlBox
from pytket.extensions.qiskit import AerBackend, AerStateBackend

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit, QiskitCircuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation


class TKETCircuit(Circuit):
    """ `qickit.circuit.TKETCircuit` is the wrapper for using Quantinuum's TKET in Qickit SDK.
    ref: https://arxiv.org/pdf/2003.10611

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
    `measured_qubits` : set[int]
        The set of measured qubits indices.
    `circuit_log` : list[dict]
        The circuit log.
    `process_gate_params_flag` : bool
        The flag to process the gate parameters.

    Raises
    ------
    TypeError
        Number of qubits bits must be integers.
    ValueError
        Number of qubits bits must be greater than 0.

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

    def _single_qubit_gate(
            self,
            gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"],
            qubit_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "I": lambda: (OpType.noop,),
            "X": lambda: (OpType.X,),
            "Y": lambda: (OpType.Y,),
            "Z": lambda: (OpType.Z,),
            "H": lambda: (OpType.H,),
            "S": lambda: (OpType.S,),
            "Sdg": lambda: (OpType.Sdg,),
            "T": lambda: (OpType.T,),
            "Tdg": lambda: (OpType.Tdg,),
            "RX": lambda: (OpType.Rx, angle/np.pi),
            "RY": lambda: (OpType.Ry, angle/np.pi),
            "RZ": lambda: (OpType.Rz, angle/np.pi),
            "Phase": lambda: (OpType.U1, angle/np.pi)
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        single_qubit_gate = gate_mapping[gate]()

        # Apply the gate to the specified qubit(s)
        for index in qubit_indices:
            self.circuit.add_gate(*single_qubit_gate, [index]) # type: ignore

    def U3(
            self,
            angles: Sequence[float],
            qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.U3.__name__, params=locals())

        # Create a single qubit unitary gate
        u3 = OpType.U3
        self.circuit.add_gate(u3, [angles[i]/np.pi for i in range(3)], [qubit_index])

    def SWAP(
            self,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.SWAP.__name__, params=locals())

        # Create a SWAP gate
        swap = OpType.SWAP
        self.circuit.add_gate(swap, [first_qubit_index, second_qubit_index])

    def _controlled_qubit_gate(
            self,
            gate: Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": lambda: QControlBox(Op.create(OpType.X), len(control_indices)),
            "Y": lambda: QControlBox(Op.create(OpType.Y), len(control_indices)),
            "Z": lambda: QControlBox(Op.create(OpType.Z), len(control_indices)),
            "H": lambda: QControlBox(Op.create(OpType.H), len(control_indices)),
            "S": lambda: QControlBox(Op.create(OpType.S), len(control_indices)),
            "Sdg": lambda: QControlBox(Op.create(OpType.Sdg), len(control_indices)),
            "T": lambda: QControlBox(Op.create(OpType.T), len(control_indices)),
            "Tdg": lambda: QControlBox(Op.create(OpType.Tdg), len(control_indices)),
            "RX": lambda: QControlBox(Op.create(OpType.Rx, angle/np.pi), len(control_indices)),
            "RY": lambda: QControlBox(Op.create(OpType.Ry, angle/np.pi), len(control_indices)),
            "RZ": lambda: QControlBox(Op.create(OpType.Rz, angle/np.pi), len(control_indices)),
            "Phase": lambda: QControlBox(Op.create(OpType.U1, angle/np.pi), len(control_indices))
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        controlled_gate = gate_mapping[gate]()

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(controlled_gate, [*control_indices[:], target_index])

    def MCU3(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.MCU3.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        u3 = Op.create(OpType.U3, [angles[i]/np.pi for i in range(3)])
        mcu3 = QControlBox(u3, len(control_indices))

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcu3, [*control_indices[:], target_index])

    def MCSWAP(
            self,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:

        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of control_indices
        swap = Op.create(OpType.SWAP)
        mcswap = QControlBox(swap, len(control_indices))

        self.circuit.add_gate(mcswap, [*control_indices[:], first_target_index, second_target_index])

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate, and apply it to the circuit
        self.circuit.add_phase(angle/np.pi)

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Check if any of the qubits have already been measured
        if any(qubit_index in self.measured_qubits for qubit_index in qubit_indices):
            raise ValueError("The qubit(s) have already been measured.")

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
            counts = {f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) \
                      for i in range(2**num_qubits_to_measure)}

        else:
            counts = backend.get_counts(circuit=circuit, num_shots=num_shots)

        return counts

    def get_depth(self) -> int:
        circuit = self.convert(QiskitCircuit)
        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = self.copy() # type: ignore

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Run the circuit and define the unitary matrix
        unitary = circuit.circuit.get_unitary()

        return np.array(unitary)

    def transpile(
            self,
            direct_transpile: bool=True,
            synthesis_method: UnitaryPreparation | None = None
        ) -> None:

        # Convert to `qickit.circuit.QiskitCircuit` to transpile the circuit
        qiskit_circuit = self.convert(QiskitCircuit)
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.TKETCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(TKETCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        pass