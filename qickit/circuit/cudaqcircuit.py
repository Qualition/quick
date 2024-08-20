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

__all__ = ["CUDAQCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

import cudaq # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit, QiskitCircuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation


class CUDAQCircuit(Circuit):
    """ `qickit.circuit.CUDAQCircuit` is the wrapper for using NVIDIA's cuda-quantum in Qickit SDK.
    ref: https://ieeexplore.ieee.org/document/10247886

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `qr` : cudaq.qvector
        The quantum bit register.
    `circuit` : cudaq.kernel
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
    >>> circuit = CUDAQCircuit(num_qubits=2)
    """
    def __init__(self,
                 num_qubits: int) -> None:
        super().__init__(num_qubits=num_qubits)

        self.circuit = cudaq.make_kernel()
        self.qr = self.circuit.qalloc(self.num_qubits)

    def _non_parameterized_single_qubit_gate(self,
                                             gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg"],
                                             qubit_indices: int | Sequence[int]) -> None:
        # Cudaq does not support the identity gate
        if gate == "I":
            return

        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "X": self.circuit.x,
            "Y": self.circuit.y,
            "Z": self.circuit.z,
            "H": self.circuit.h,
            "S": self.circuit.s,
            "T": self.circuit.t
        }

        # Apply the gate to the specified qubit(s)
        for index in qubit_indices:
            gate_mapping[gate](self.qr[index])

    def _parameterized_single_qubit_gate(self,
                                         gate: Literal["RX", "RY", "RZ"],
                                         qubit_indices: int | Sequence[int],
                                         angle: float) -> None:
        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define the gate mapping for the parameterized single qubit gates
        gate_mapping = {
            "RX": self.circuit.rx,
            "RY": self.circuit.ry,
            "RZ": self.circuit.rz
        }

        # Apply the gate to the specified qubit(s)
        for index in qubit_indices:
            gate_mapping[gate](angle, self.qr[index])

    def _single_qubit_gate(self,
                           gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
                           qubit_indices: int | Sequence[int],
                           angle: float=0) -> None:
        # With cuda-quantum we cannot abstract the single qubit gates as a single method
        # without committing wrong abstraction, hence we will simply wrap the gates in
        # the `_single_qubit_gate` method
        if gate in ["I", "X", "Y", "Z", "H", "S", "T"]:
            self._non_parameterized_single_qubit_gate(gate, qubit_indices) # type: ignore
        elif gate in ["RX", "RY", "RZ"]:
            self._parameterized_single_qubit_gate(gate, qubit_indices, angle) # type: ignore

    def U3(self,
           angles: Sequence[float],
           qubit_index: int) -> None:
        self.process_gate_params(gate=self.U3.__name__, params=locals().copy())
        self.circuit.u3(angles[0], angles[1], angles[2], self.qr[qubit_index])

    def SWAP(self,
             first_qubit_index: int,
             second_qubit_index: int) -> None:
        self.process_gate_params(gate=self.SWAP.__name__, params=locals().copy())
        self.circuit.swap(self.qr[first_qubit_index], self.qr[second_qubit_index])

    def _non_parameterized_controlled_gate(self,
                                           gate: Literal["X", "Y", "Z", "H", "S", "T"],
                                           control_indices: Sequence[int],
                                           target_indices: Sequence[int]) -> None:
        # Define the gate mapping for the non-parameterized controlled gates
        gate_mapping = {
            "X": self.circuit.cx,
            "Y": self.circuit.cy,
            "Z": self.circuit.cz,
            "H": self.circuit.ch,
            "S": self.circuit.cs,
            "T": self.circuit.ct
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            gate_mapping[gate](*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    def _parameterized_controlled_gate(self,
                                       gate: Literal["RX", "RY", "RZ"],
                                       angles: float,
                                       control_indices: Sequence[int],
                                       target_indices: Sequence[int]) -> None:
        # Define the gate mapping for the parameterized controlled gates
        gate_mapping = {
            "RX": self.circuit.crx,
            "RY": self.circuit.cry,
            "RZ": self.circuit.crz,
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            gate_mapping[gate](angles, *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    def _controlled_qubit_gate(self,
                               gate: Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ"],
                               control_indices: int | Sequence[int],
                               target_indices: int | Sequence[int],
                               angle: float=0) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # With cuda-quantum we cannot abstract the controlled gates as a single method
        # without committing wrong abstraction, hence we will simply wrap the gates in
        # the `_controlled_qubit_gate` method
        if gate in ["X", "Y", "Z", "H", "S", "T"]:
            self._non_parameterized_controlled_gate(gate, control_indices, target_indices) # type: ignore
        elif gate in ["RX", "RY", "RZ"]:
            self._parameterized_controlled_gate(gate, angle, control_indices, target_indices) # type: ignore

    def MCU3(self,
             angles: Sequence[float],
             control_indices: int | Sequence[int],
             target_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.MCU3.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the Multi-Controlled U3 gate controlled by all control indices to each target index
        # NOTE: CUDAQ version 0.8 will provide native support of U3 gates
        for target_index in target_indices:
            self.circuit.cu3(angles[0], angles[1], angles[2],
                             *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    def MCSWAP(self,
               control_indices: int | Sequence[int],
               first_target_index: int,
               second_target_index: int) -> None:
        self.process_gate_params(gate=self.MCSWAP.__name__, params=locals().copy())

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Apply the MCSWAP gate to the circuit controlled by all control indices to the target indices
        self.circuit.cswap(*map(self.qr.__getitem__, control_indices),
                           self.qr[first_target_index], self.qr[second_target_index])

    def GlobalPhase(self,
                    angle: float) -> None:
        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals().copy())

        global_phase = np.array([[np.exp(1j * angle), 0],
                                 [0, np.exp(1j * angle)]], dtype=np.complex128)

        cudaq.register_operation("global_phase", global_phase)
        self.circuit.global_phase(self.qr[0])

    def measure(self,
                qubit_indices: int | Sequence[int]) -> None:
        self.process_gate_params(gate=self.measure.__name__, params=locals().copy())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        if any(self.measured_qubits[qubit_index] for qubit_index in qubit_indices):
            raise ValueError("The qubit(s) have already been measured")

        # Measure the qubits
        for qubit_index in qubit_indices:
            self.circuit.mz(self.qr[qubit_index])

        # Set the measurement as applied
        list(map(self.measured_qubits.__setitem__, qubit_indices, [True]*len(qubit_indices)))

    def get_statevector(self,
                        backend: Backend | None = None,
                        magnitude_only: bool=False) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CUDAQCircuit = self.convert(CUDAQCircuit) # type: ignore

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            state_vector = np.array(cudaq.get_state(circuit.circuit))
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

        return state_vector

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict:
        if not(any(self.measured_qubits)):
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the measurement and vertical reverse operations are applied inplace
        circuit: CUDAQCircuit = self.convert(CUDAQCircuit) # type: ignore

        # CUDAQ uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            result = str(cudaq.sample(circuit.circuit, num_shots)).split()[1:-1]
            counts = {pair.split(":")[0]: int(pair.split(":")[1]) for pair in result}

        else:
            counts = backend.get_counts(circuit, num_shots)

        return counts

    def get_depth(self) -> int:
        circuit = self.convert(QiskitCircuit)
        return circuit.get_depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CUDAQCircuit = self.convert(CUDAQCircuit) # type: ignore

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # TODO: Define the unitary matrix (need 0.9 version of cuda-quantum)
        unitary: list = []

        return np.array(unitary)

    def transpile(self,
                  direct_transpile: bool=True,
                  synthesis_method: UnitaryPreparation | None = None) -> None:
        # Convert to `qickit.circuit.QiskitCircuit` to transpile the circuit
        qiskit_circuit = self.convert(QiskitCircuit)
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.CUDAQCircuit` to update the circuit
        updated_circuit = qiskit_circuit.convert(CUDAQCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(self,
                qasm_version: int=2) -> str:
        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        print(cudaq.draw(self.circuit))