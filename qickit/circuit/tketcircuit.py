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
import copy
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING
from typing import Dict, Tuple, Union

from pytket import Circuit as TKCircuit
from pytket import OpType
from pytket.circuit import Op, QControlBox
from pytket.extensions.qiskit import AerBackend

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
    `measured_qubits` : list[bool]
        The measurement status of the qubits.
    `circuit_log` : list[dict]
        The circuit log.

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
    def __init__(self,
                 num_qubits: int) -> None:
        super().__init__(num_qubits=num_qubits)

        self.circuit: TKCircuit = TKCircuit(self.num_qubits, self.num_qubits)

    def _single_qubit_gate(self,
                           gate: Literal["I", "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"],
                           qubit_indices: int | Sequence[int],
                           angle: float=0) -> None:
        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "I": (OpType.noop,),
            "X": (OpType.X,),
            "Y": (OpType.Y,),
            "Z": (OpType.Z,),
            "H": (OpType.H,),
            "S": (OpType.S,),
            "T": (OpType.T,),
            "RX": (OpType.Rx, angle/np.pi),
            "RY": (OpType.Ry, angle/np.pi),
            "RZ": (OpType.Rz, angle/np.pi)
        }

        # Apply the gate to the specified qubit(s)
        for index in qubit_indices:
            self.circuit.add_gate(*gate_mapping[gate], [index]) # type: ignore

    def U3(self,
           angles: Sequence[float],
           qubit_index: int) -> None:
        self.process_gate_params(gate=self.U3.__name__, params=locals().copy())

        # Create a single qubit unitary gate
        u3 = OpType.U3

        self.circuit.add_gate(u3, [angles[i]/np.pi for i in range(3)], [qubit_index])

    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        self.process_gate_params(gate=self.SWAP.__name__, params=locals().copy())

        # Create a SWAP gate
        swap = OpType.SWAP

        self.circuit.add_gate(swap, [first_qubit, second_qubit])

    def _controlled_qubit_gate(self,
                               gate: Literal["I", "X", "Y", "Z", "H", "S", "T", "RX", "RY", "RZ"],
                               control_indices: int | Sequence[int],
                               target_indices: int | Sequence[int],
                               angle: float=0) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": QControlBox(Op.create(OpType.X), len(control_indices)),
            "Y": QControlBox(Op.create(OpType.Y), len(control_indices)),
            "Z": QControlBox(Op.create(OpType.Z), len(control_indices)),
            "H": QControlBox(Op.create(OpType.H), len(control_indices)),
            "S": QControlBox(Op.create(OpType.S), len(control_indices)),
            "T": QControlBox(Op.create(OpType.T), len(control_indices)),
            "RX": QControlBox(Op.create(OpType.Rx, angle/np.pi), len(control_indices)),
            "RY": QControlBox(Op.create(OpType.Ry, angle/np.pi), len(control_indices)),
            "RZ": QControlBox(Op.create(OpType.Rz, angle/np.pi), len(control_indices))
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(gate_mapping[gate], [*control_indices[:], target_index])

    def MCU3(self,
             angles: Sequence[float],
             control_indices: int | Sequence[int],
             target_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.MCU3.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        u3 = Op.create(OpType.U3, [angles[i]/np.pi for i in range(3)])
        mcu3 = QControlBox(u3, len(control_indices))

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.add_qcontrolbox(mcu3, [*control_indices[:], target_index])

    def MCSWAP(self,
               control_indices: int | Sequence[int],
               first_target_index: int,
               second_target_index: int) -> None:
        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of control_indices
        swap = Op.create(OpType.SWAP)
        mcswap = QControlBox(swap, len(control_indices))

        self.circuit.add_gate(mcswap, [*control_indices[:], first_target_index, second_target_index])

    def GlobalPhase(self,
                    angle: float) -> None:
        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals().copy())

        # Create a Global Phase gate, and apply it to the circuit
        self.circuit.add_phase(angle/np.pi)

    def measure(self,
                qubit_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.measure.__name__, params=locals().copy())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Check if any of the qubits have already been measured
        if any(self.measured_qubits[qubit_index] for qubit_index in qubit_indices):
            raise ValueError("The qubit(s) have already been measured")

        # Measure the qubits
        if isinstance(qubit_indices, int):
            self.circuit.Measure(qubit_indices, qubit_indices)
        elif isinstance(qubit_indices, Sequence):
            for index in qubit_indices:
                self.circuit.Measure(index, index)

        # Set the measurement as applied
        list(map(self.measured_qubits.__setitem__, qubit_indices, [True]*len(qubit_indices)))

    def get_statevector(self,
                        backend: Backend | None = None,
                        magnitude_only: bool=False) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = copy.deepcopy(self)

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            state_vector = circuit.circuit.get_statevector()
        else:
            state_vector = backend.get_statevector(circuit)

        if magnitude_only:
            # Round off the small values to 0 (values below 1e-12 are set to 0)
            state_vector = np.round(state_vector, 12)

            # Create masks for real and imaginary parts
            real_mask = (state_vector.imag == 0)
            imag_mask = (state_vector.real == 0)

            # Calculate the sign for each part
            real_sign = np.sign(state_vector.real) * real_mask
            imag_sign = np.sign(state_vector.imag) * imag_mask

            # Calculate the sign for complex numbers
            complex_sign = np.sign(state_vector.real * (np.abs(state_vector.real) <= np.abs(state_vector.imag)) + \
                                state_vector.imag * (np.abs(state_vector.imag) < np.abs(state_vector.real))) * \
                                ~(real_mask | imag_mask)

            # Define the signs for the real and imaginary components
            signs = real_sign + imag_sign + complex_sign

            # Multiply the state vector by the signs
            state_vector = signs * np.abs(state_vector)

        return np.array(state_vector)

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict[str, int]:
        if not(any(self.measured_qubits)):
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = copy.deepcopy(self)

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Extract what qubits are measured
        qubits_to_measure = [i for i in range(circuit.num_qubits) if circuit.measured_qubits[i]]
        num_qubits_to_measure = len(qubits_to_measure)

        if backend is None:
            # If no backend is provided, use the AerBackend
            base_backend = AerBackend()
            circuit = base_backend.get_compiled_circuits([circuit.circuit]) # type: ignore
            result = base_backend.run_circuit(circuit[0], n_shots=num_shots, seed=0) # type: ignore
            # Get the counts
            counts = {"".join(map(str, basis_state)): num_counts
                      for basis_state, num_counts in result.get_counts().items()}
            # Extract the counts for the measured qubits
            counts = {str(key[int(self.num_qubits-num_qubits_to_measure):]): value for key, value in counts.items()}
            # Fill in the missing counts
            counts = {f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) for i in range(2**num_qubits_to_measure)}
        else:
            counts = backend.get_counts(circuit=circuit, num_shots=num_shots)

        return counts

    def get_depth(self) -> int:
        circuit = self.convert(QiskitCircuit)
        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: TKETCircuit = copy.deepcopy(self)

        # PyTKET uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Run the circuit and define the unitary matrix
        unitary = circuit.circuit.get_unitary()

        return np.array(unitary)

    def transpile(self,
                  direct_transpile: bool=True,
                  synthesis_method: UnitaryPreparation | None = None) -> None:
        # Convert to `qickit.circuit.QiskitCircuit` to transpile the circuit
        qiskit_circuit = self.convert(QiskitCircuit)
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.TKETCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(TKETCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(self,
                qasm_version: int=2) -> str:
        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        pass