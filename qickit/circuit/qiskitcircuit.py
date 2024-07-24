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

__all__ = ["QiskitCircuit"]

from collections.abc import Sequence
import copy
import matplotlib.figure
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile # type: ignore
from qiskit.circuit.library import (RXGate, RYGate, RZGate, HGate, XGate, YGate, # type: ignore
                                    ZGate, SGate, SdgGate, TGate, TdgGate, U3Gate, # type: ignore
                                    SwapGate, GlobalPhaseGate, IGate) # type: ignore
from qiskit.primitives import BackendSampler # type: ignore
from qiskit_aer import AerSimulator # type: ignore
import qiskit.qasm2 as qasm2 # type: ignore
import qiskit.qasm3 as qasm3 # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation, QiskitUnitaryTranspiler


class QiskitCircuit(Circuit):
    """ `qickit.circuit.QiskitCircuit` is the wrapper for using IBM Qiskit in Qickit SDK.
    ref: https://arxiv.org/pdf/2405.08810

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
    >>> circuit = QiskitCircuit(num_qubits=2)
    """
    def __init__(self,
                 num_qubits: int) -> None:
        super().__init__(num_qubits=num_qubits)

        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        self.circuit: QuantumCircuit = QuantumCircuit(qr, cr)

    def _single_qubit_gate(self,
                           gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
                           qubit_indices: int | Sequence[int],
                           angle: float=0) -> None:
        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "I": IGate(),
            "X": XGate(),
            "Y": YGate(),
            "Z": ZGate(),
            "H": HGate(),
            "S": SGate(),
            "Sdg": SdgGate(),
            "T": TGate(),
            "Tdg": TdgGate(),
            "RX": RXGate(angle),
            "RY": RYGate(angle),
            "RZ": RZGate(angle)
        }

        # Apply the gate to the specified qubit(s)
        for index in qubit_indices:
            self.circuit.append(gate_mapping[gate], [index])

    def U3(self,
           angles: Sequence[float],
           qubit_index: int) -> None:
        self.process_gate_params(gate=self.U3.__name__, params=locals().copy())

        # Create a single qubit unitary gate
        u3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2])

        self.circuit.append(u3, [qubit_index])

    def SWAP(self,
             first_qubit_index: int,
             second_qubit_index: int) -> None:
        self.process_gate_params(gate=self.SWAP.__name__, params=locals().copy())

        # Create a SWAP gate
        swap = SwapGate()

        self.circuit.append(swap, [first_qubit_index, second_qubit_index])

    def _controlled_qubit_gate(self,
                               gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
                               control_indices: int | Sequence[int],
                               target_indices: int | Sequence[int],
                               angle: float=0) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": XGate().control(len(control_indices)),
            "Y": YGate().control(len(control_indices)),
            "Z": ZGate().control(len(control_indices)),
            "H": HGate().control(len(control_indices)),
            "S": SGate().control(len(control_indices)),
            "Sdg": SdgGate().control(len(control_indices)),
            "T": TGate().control(len(control_indices)),
            "Tdg": TdgGate().control(len(control_indices)),
            "RX": RXGate(angle).control(len(control_indices)),
            "RY": RYGate(angle).control(len(control_indices)),
            "RZ": RZGate(angle).control(len(control_indices))
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(gate_mapping[gate], [*control_indices[:], target_index])

    def MCU3(self,
             angles: Sequence[float],
             control_indices: int | Sequence[int],
             target_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.MCU3.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        mcu3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2]).control(len(control_indices))

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(mcu3, [*control_indices[:], target_index])

    def MCSWAP(self,
               control_indices: int | Sequence[int],
               first_target_index: int,
               second_target_index: int) -> None:
        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of control_indices
        mcswap = SwapGate().control(len(control_indices))

        self.circuit.append(mcswap, [*control_indices[:], first_target_index, second_target_index])

    def GlobalPhase(self,
                    angle: float) -> None:
        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals().copy())

        # Create a Global Phase gate
        global_phase = GlobalPhaseGate(angle)

        self.circuit.append(global_phase, (), ())

    def measure(self,
                qubit_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.measure.__name__, params=locals().copy())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # Check if any of the qubits have already been measured
        if any(self.measured_qubits[qubit_index] for qubit_index in qubit_indices):
            raise ValueError("The qubit(s) have already been measured")

        self.circuit.measure(qubit_indices, qubit_indices)

        # Set the measurement as applied
        list(map(self.measured_qubits.__setitem__, qubit_indices, [True]*len(qubit_indices)))

    def get_statevector(self,
                        backend: Backend | None = None,
                        magnitude_only: bool=False) -> NDArray[np.complex128]:
        if backend is None:
            state_vector = Statevector(self.circuit).data
        else:
            state_vector = backend.get_statevector(self)

        if magnitude_only:
            # Round off the small values to 0 (values below 1e-12 are set to 0)
            state_vector = np.round(state_vector, 12)

            # Create masks for real and imaginary parts
            real_mask = (state_vector.imag == 0)
            imaginary_mask = (state_vector.real == 0)

            # Calculate the sign for each part
            real_sign = np.sign(state_vector.real) * real_mask
            imaginary_sign = np.sign(state_vector.imag) * imaginary_mask

            # Calculate the sign for complex numbers
            complex_sign = np.sign(state_vector.real * (np.abs(state_vector.real) <= np.abs(state_vector.imag)) + \
                                state_vector.imag * (np.abs(state_vector.imag) < np.abs(state_vector.real))) * \
                                ~(real_mask | imaginary_mask)

            # Define the signs for the real and imaginary components
            signs = real_sign + imaginary_sign + complex_sign

            # Multiply the state vector by the signs
            state_vector = signs * np.abs(state_vector)

        return np.array(state_vector)

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict[str, int]:
        if not(any(self.measured_qubits)):
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = copy.deepcopy(self)

        # Extract what qubits are measured
        qubits_to_measure = [i for i in range(circuit.num_qubits) if circuit.measured_qubits[i]]
        num_qubits_to_measure = len(qubits_to_measure)

        if backend is None:
            # If no backend is provided, use the AerSimualtor
            base_backend: BackendSampler = BackendSampler(AerSimulator())
            result = base_backend.run(circuit.circuit, shots=num_shots, seed_simulator=0).result()
            # Extract the quasi-probability distribution from the first result
            quasi_dist = result.quasi_dists[0]
            # Convert the quasi-probability distribution to counts
            counts = {bin(k)[2:].zfill(num_qubits_to_measure): int(v * num_shots) for k, v in quasi_dist.items()}
            # Fill the counts array with zeros for the missing states
            counts = {f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) for i in range(2**num_qubits_to_measure)}
            # Sort the counts by their keys (basis states)
            counts = dict(sorted(counts.items()))
        else:
            counts = backend.get_counts(circuit=circuit, num_shots=num_shots)

        return counts

    def get_depth(self) -> int:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = copy.deepcopy(self)

        # Transpile the circuit to U3 and CX gates
        circuit.transpile()

        return circuit.circuit.depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = copy.deepcopy(self)

        # Get the unitary matrix of the circuit
        unitary = Operator(circuit.circuit).data

        return np.array(unitary)

    def transpile(self,
                  direct_transpile: bool=True,
                  synthesis_method: UnitaryPreparation | None = None) -> None:
        if direct_transpile:
            # Transpile the circuit (this returns a `qiskit.QuantumCircuit` instance)
            transpiled_circuit = transpile(self.circuit,
                                           optimization_level=3,
                                           basis_gates=['u3', 'cx'],
                                           seed_transpiler=0)

            # Define a `qickit.circuit.QiskitCircuit` instance from the transpiled circuit
            transpiled_circuit = self.from_qiskit(transpiled_circuit, QiskitCircuit)

        else:
            if synthesis_method is None:
                # If no synthesis method is provided, use the default QiskitUnitaryTranspiler
                synthesis_method = QiskitUnitaryTranspiler(output_framework=type(self))

            # Get the unitary matrix of the circuit
            unitary_matrix = self.get_unitary()

            # Prepare the unitary matrix
            transpiled_circuit = synthesis_method.prepare_unitary(unitary_matrix)

        # Update the circuit
        self.circuit_log = transpiled_circuit.circuit_log
        self.circuit = transpiled_circuit.circuit

    def to_qasm(self,
                qasm_version:int = 2) -> str:
        if qasm_version == 2:
            qasm = qasm2.dumps(self.circuit)
        elif qasm_version == 3:
            qasm = qasm3.dumps(self.circuit)
        else:
            raise ValueError("The QASM version must be either 2 or 3.")

        return qasm

    def draw(self) -> None:
        self.circuit.draw(output='mpl')