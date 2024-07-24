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

__all__ = ["CirqCircuit"]

from collections.abc import Sequence
import copy
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
    >>> circuit = CirqCircuit(num_qubits=2)
    """
    def __init__(self,
                 num_qubits: int) -> None:
        super().__init__(num_qubits=num_qubits)

        self.qr = cirq.LineQubit.range(self.num_qubits)
        self.measurement_keys = []

        # Define the circuit (Need to add an identity, otherwise `.get_unitary()`
        # returns the state instead of the operator of the circuit)
        self.circuit: cirq.Circuit = cirq.Circuit()
        self.circuit.append(I(self.qr[0]))

    def _single_qubit_gate(self,
                           gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
                           qubit_indices: int | Sequence[int],
                           angle: float=0) -> None:
        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "I": I,
            "X": X,
            "Y": Y,
            "Z": Z,
            "H": H,
            "S": S,
            "Sdg": S**-1,
            "T": T,
            "Tdg": T**-1,
            "RX": Rx(rads=angle),
            "RY": Ry(rads=angle),
            "RZ": Rz(rads=angle)
        }

        # Apply the gate to the specified qubit(s)
        for index in qubit_indices:
            self.circuit.append(gate_mapping[gate](self.qr[index]))

    def U3(self,
           angles: Sequence[float],
           qubit_index: int) -> None:
        self.process_gate_params(gate=self.U3.__name__, params=locals().copy())

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

    def SWAP(self,
             first_qubit_index: int,
             second_qubit_index: int) -> None:
        self.process_gate_params(gate=self.SWAP.__name__, params=locals().copy())
        swap = cirq.SWAP
        self.circuit.append(swap(self.qr[first_qubit_index], self.qr[second_qubit_index]))

    def _controlled_qubit_gate(self,
                               gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
                               control_indices: int | Sequence[int],
                               target_indices: int | Sequence[int],
                               angle: float=0) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": cirq.ControlledGate(sub_gate=X, num_controls=len(control_indices)),
            "Y": cirq.ControlledGate(sub_gate=Y, num_controls=len(control_indices)),
            "Z": cirq.ControlledGate(sub_gate=Z, num_controls=len(control_indices)),
            "H": cirq.ControlledGate(sub_gate=H, num_controls=len(control_indices)),
            "S": cirq.ControlledGate(sub_gate=S, num_controls=len(control_indices)),
            "Sdg": cirq.ControlledGate(sub_gate=S**-1, num_controls=len(control_indices)),
            "T": cirq.ControlledGate(sub_gate=T, num_controls=len(control_indices)),
            "Tdg": cirq.ControlledGate(sub_gate=T**-1, num_controls=len(control_indices)),
            "RX": cirq.ControlledGate(sub_gate=Rx(rads=angle), num_controls=len(control_indices)),
            "RY": cirq.ControlledGate(sub_gate=Ry(rads=angle), num_controls=len(control_indices)),
            "RZ": cirq.ControlledGate(sub_gate=Rz(rads=angle), num_controls=len(control_indices))
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(
                gate_mapping[gate](*map(self.qr.__getitem__, control_indices), self.qr[target_index])
            )

    def MCU3(self,
             angles: Sequence[float],
             control_indices: int | Sequence[int],
             target_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.MCU3.__name__, params=locals().copy())

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

    def MCSWAP(self,
               control_indices: int | Sequence[int],
               first_target_index: int,
               second_target_index: int) -> None:
        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of `control_indices`
        mcswap = cirq.ControlledGate(sub_gate=SWAP, num_controls=len(control_indices))

        # Apply the MCSWAP gate to the circuit controlled by all control indices to the target indices
        self.circuit.append(
            mcswap(*map(self.qr.__getitem__, control_indices),
                   self.qr[first_target_index], self.qr[second_target_index])
        )

    def GlobalPhase(self,
                    angle: float) -> None:
        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals().copy())

        # Create a Global Phase gate (Cirq takes in e^i*angle as the argument)
        global_phase = cirq.GlobalPhaseGate(np.exp(1j*angle))

        self.circuit.append(global_phase())

    def measure(self,
                qubit_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.measure.__name__, params=locals().copy())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Check if any of the qubits have already been measured
        if any(self.measured_qubits[qubit_index] for qubit_index in qubit_indices):
            raise ValueError("The qubit(s) have already been measured.")

        # Measure the qubits
        # NOTE: We must sort the indices as Cirq doesn't understand that the order of measurements
        # is irrelevant. This is done to ensure that the measurements are consistent across different
        # framework.
        for qubit_index in sorted(qubit_indices):
            self.circuit.append(cirq.measure(self.qr[qubit_index], key=f"q{qubit_index}"))
            self.measurement_keys.append(f"q{qubit_index}")

        # Sort the measurement keys (as explained in the NOTE above)
        self.measurement_keys = sorted(self.measurement_keys)

        # Set the measurement as applied
        list(map(self.measured_qubits.__setitem__, qubit_indices, [True]*len(qubit_indices)))

    def get_statevector(self,
                        backend: Backend | None = None,
                        magnitude_only: bool=False) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            state_vector = circuit.circuit.final_state_vector(qubit_order=self.qr)
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
                   backend: Backend | None = None) -> dict:
        if not(any(self.measured_qubits)):
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Extract what qubits are measured
        qubits_to_measure = [i for i in range(circuit.num_qubits) if circuit.measured_qubits[i]]
        num_qubits_to_measure = len(qubits_to_measure)

        if backend is None:
            # If no backend is provided, use the `cirq.Simulator`
            base_backend = cirq.Simulator()
            # Run the circuit to get the result
            result = base_backend.run(circuit.circuit, repetitions=num_shots)
            # Format the result to get the counts
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
        circuit: CirqCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the unitary matrix
        unitary = cirq.unitary(circuit.circuit)

        return np.array(unitary)

    def transpile(self,
                  direct_transpile: bool=True,
                  synthesis_method: UnitaryPreparation | None = None) -> None:
        # Convert to `qickit.circuit.QiskitCircuit` to transpile the circuit
        qiskit_circuit = self.convert(QiskitCircuit)
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.CirqCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(CirqCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(self,
                qasm_version: int=2) -> str:
        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        print(self.circuit)