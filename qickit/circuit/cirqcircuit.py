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
import cirq
import numpy as np
from numpy.typing import NDArray
from qickit.circuit import Circuit

__all__ = ["CirqCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, SWAP, I

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit, QiskitCircuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation


class CirqCircuit(Circuit):
    """ `qickit.circuit.CirqCircuit` is the wrapper for using Google's Cirq in Qickit SDK.
    ref: https://zenodo.org/records/11398048

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `qr` : cirq.LineQubit
        The quantum bit register.
    `measurement_keys`: list[str]
        The measurement keys.
    `circuit` : cirq.Circuit
        The circuit.
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
    >>> circuit = CirqCircuit(num_qubits=2)
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:

        super().__init__(num_qubits=num_qubits)

        self.qr = cirq.LineQubit.range(self.num_qubits)
        self.measurement_keys: list[str] = []

        # Define the circuit (Need to add an identity, otherwise `.get_unitary()`
        # returns the state instead of the operator of the circuit)
        self.circuit: cirq.Circuit = cirq.Circuit()
        self.circuit.append(I(self.qr[0]))

    def _single_qubit_gate(
            self,
            gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
            qubit_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define a mapping for the single qubit gates
        gate_mapping = {
            "I": lambda: I,
            "X": lambda: X,
            "Y": lambda: Y,
            "Z": lambda: Z,
            "H": lambda: H,
            "S": lambda: S,
            "Sdg": lambda: S**-1,
            "T": lambda: T,
            "Tdg": lambda: T**-1,
            "RX": lambda: Rx(rads=angle),
            "RY": lambda: Ry(rads=angle),
            "RZ": lambda: Rz(rads=angle)
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        single_qubit_gate = gate_mapping[gate]()

        # Apply the single qubit gate to each qubit index
        for qubit_index in qubit_indices:
            self.circuit.append(single_qubit_gate(self.qr[qubit_index]))

    def U3(
            self,
            angles: Sequence[float],
            qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.U3.__name__, params=locals())

        # Define the unitary matrix for the U3 gate
        u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
              [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
                                                           np.cos(angles[0]/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        self.circuit.append(U3().on(self.qr[qubit_index]))

    def SWAP(
            self,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.SWAP.__name__, params=locals())
        swap = cirq.SWAP
        self.circuit.append(swap(self.qr[first_qubit_index], self.qr[second_qubit_index]))

    def _controlled_qubit_gate(
            self,
            gate: Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define a mapping for the controlled qubit gates
        gate_mapping = {
            "X": lambda: cirq.ControlledGate(sub_gate=X, num_controls=len(control_indices)),
            "Y": lambda: cirq.ControlledGate(sub_gate=Y, num_controls=len(control_indices)),
            "Z": lambda: cirq.ControlledGate(sub_gate=Z, num_controls=len(control_indices)),
            "H": lambda: cirq.ControlledGate(sub_gate=H, num_controls=len(control_indices)),
            "S": lambda: cirq.ControlledGate(sub_gate=S, num_controls=len(control_indices)),
            "Sdg": lambda: cirq.ControlledGate(sub_gate=S**-1, num_controls=len(control_indices)),
            "T": lambda: cirq.ControlledGate(sub_gate=T, num_controls=len(control_indices)),
            "Tdg": lambda: cirq.ControlledGate(sub_gate=T**-1, num_controls=len(control_indices)),
            "RX": lambda: cirq.ControlledGate(sub_gate=Rx(rads=angle), num_controls=len(control_indices)),
            "RY": lambda: cirq.ControlledGate(sub_gate=Ry(rads=angle), num_controls=len(control_indices)),
            "RZ": lambda: cirq.ControlledGate(sub_gate=Rz(rads=angle), num_controls=len(control_indices))
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        controlled_qubit_gate = gate_mapping[gate]()

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                controlled_qubit_gate(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    def MCU3(
            self,
            angles: Sequence[float],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.MCU3.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a single qubit unitary gate
        u3 = [[np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
              [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
                                                           np.cos(angles[0]/2)]]

        # Define the U3 gate class
        class U3(cirq.Gate):
            def __init__(self):
                super(U3, self)

            def _num_qubits_(self):
                return 1

            def _unitary_(self):
                return np.array(u3)

            def _circuit_diagram_info_(self, args):
                return "U3"

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of `control_indices` with the specified angle
        mcu3 = cirq.ControlledGate(sub_gate=U3(), num_controls=len(control_indices))

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                mcu3(*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    def MCSWAP(
            self,
            control_indices: int | Sequence[int],
            first_target_index: int,
            second_target_index: int
        ) -> None:

        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of `control_indices`
        mcswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=len(control_indices))

        # Apply the MCSWAP gate to the circuit controlled by all control indices to the target indices
        self.circuit.append(
            mcswap(*map(self.qr.__getitem__, control_indices),
                   self.qr[first_target_index], self.qr[second_target_index])
        )

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate (Cirq takes in e^i*angle as the argument)
        global_phase = cirq.GlobalPhaseGate(np.exp(1j*angle))

        self.circuit.append(global_phase())

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

        # We must sort the indices as Cirq doesn't understand that the order of measurements
        # is irrelevant
        # This is done to ensure that the measurements are consistent across different
        # framework
        for qubit_index in sorted(qubit_indices):
            self.circuit.append(cirq.measure(self.qr[qubit_index], key=f"q{qubit_index}"))
            self.measurement_keys.append(f"q{qubit_index}")

        self.measurement_keys = sorted(self.measurement_keys)

        # Set the measurement as applied
        for qubit_index in qubit_indices:
            self.measured_qubits.add(qubit_index)

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:
        circuit: CirqCircuit = self.copy()  # type: ignore
        circuit.vertical_reverse()
        state_vector = circuit.circuit.final_state_vector(qubit_order=self.qr) if backend is None else backend.get_statevector(circuit)
        return np.array(state_vector)

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict:

        num_qubits_to_measure = len(self.measured_qubits)

        if num_qubits_to_measure == 0:
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = self.copy() # type: ignore

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            # If no backend is provided, use the `cirq.Simulator`
            base_backend = cirq.Simulator()
            # Run the circuit to get the result
            result = base_backend.run(circuit.circuit, repetitions=num_shots)
            # Using the `multi_measurement_histogram` method to get the counts we can
            # get the counts given the measurement keys, allowing for partial measurement
            # without post-processing
            counts = dict(result.multi_measurement_histogram(keys=circuit.measurement_keys))
            counts = {''.join(map(str, key)): value for key, value in counts.items()}
            for i in range(2**num_qubits_to_measure):
                basis = format(int(i),"0{}b".format(num_qubits_to_measure))
                if basis not in counts:
                    counts[basis] = 0
                else:
                    counts[basis] = int(counts[basis])
            counts = dict(sorted(counts.items()))

        else:
            counts = backend.get_counts(circuit, num_shots)

        return counts

    def get_depth(self) -> int:
        circuit = self.convert(QiskitCircuit)
        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = self.copy() # type: ignore

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the unitary matrix
        unitary = cirq.unitary(circuit.circuit)

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

        # Convert back to `qickit.circuit.CirqCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(CirqCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        print(self.circuit)