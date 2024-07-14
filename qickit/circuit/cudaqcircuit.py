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

import copy
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

import cudaq # type: ignore

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit, QiskitCircuit
from qickit.synthesis.unitarypreparation import UnitaryPreparation
from qickit.types import Collection


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
                                             gate: Literal["I", "X", "Y", "Z", "H", "S", "T"],
                                             qubit_indices: int | Collection[int]) -> None:
        # Define the gate mapping for the non-parameterized single qubit gates
        gate_mapping = {
            "I": None,
            "X": self.circuit.x,
            "Y": self.circuit.y,
            "Z": self.circuit.z,
            "H": self.circuit.h,
            "S": self.circuit.s,
            "T": self.circuit.t
        }

        # Apply the gate to the specified qubit(s)
        if isinstance(qubit_indices, Collection):
            for index in qubit_indices:
                gate_mapping[gate](self.qr[index])
        else:
            gate_mapping[gate](self.qr[qubit_indices])

    def _parameterized_single_qubit_gate(self,
                                         gate: Literal["RX", "RY", "RZ", "U3"],
                                         angles: float | Collection[float],
                                         qubit_index: int) -> None:
        # Define the gate mapping for the parameterized single qubit gates
        gate_mapping = {
            "RX": self.circuit.rx,
            "RY": self.circuit.ry,
            "RZ": self.circuit.rz,
            "U3": None
        }

        # Apply the gate to the specified qubit
        gate_mapping[gate](angles, self.qr[qubit_index])

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # NOTE: CUDAQ version 0.8 will provide native support of U3 gates
        self.circuit.rz(angles[2], self.qr[qubit_index])
        self.circuit.rx(np.pi/2, self.qr[qubit_index])
        self.circuit.rz(angles[0], self.qr[qubit_index])
        self.circuit.rx(-np.pi/2, self.qr[qubit_index])
        self.circuit.rz(angles[1], self.qr[qubit_index])

    def _non_parameterized_controlled_gate(self,
                                           gate: Literal["X", "Y", "Z", "H", "S", "T"],
                                           control_indices: int | Collection[int],
                                           target_indices: int | Collection[int]) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

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
                                       gate: Literal["RX", "RY", "RZ", "U3"],
                                       angles: float | Collection[float],
                                       control_indices: int | Collection[int],
                                       target_indices: int | Collection[int]) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define the gate mapping for the parameterized controlled gates
        gate_mapping = {
            "RX": self.circuit.crx,
            "RY": self.circuit.cry,
            "RZ": self.circuit.crz,
            "U3": None
        }

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            gate_mapping[gate](angles, *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        self.circuit.swap(self.qr[first_qubit], self.qr[second_qubit])

    @Circuit.gatemethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        # NOTE: CUDAQ version 0.8 will provide native support of U3 gates
        self.circuit.crz(angles[2], self.qr[control_index], self.qr[target_index])
        self.circuit.crx(np.pi/2, self.qr[control_index], self.qr[target_index])
        self.circuit.crz(angles[0], self.qr[control_index], self.qr[target_index])
        self.circuit.crx(-np.pi/2, self.qr[control_index], self.qr[target_index])
        self.circuit.crz(angles[1], self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CSWAP(self,
              control_index: int,
              first_target_index: int,
              second_target_index: int) -> None:
        self.circuit.cswap(self.qr[control_index],
                           self.qr[first_target_index], self.qr[second_target_index])

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the Multi-Controlled U3 gate controlled by all control indices to each target index
        # NOTE: CUDAQ version 0.8 will provide native support of U3 gates
        for target_index in target_indices:
            self.circuit.crz(angles[2],
                             *map(self.qr.__getitem__, control_indices), self.qr[target_index])
            self.circuit.crx(np.pi/2,
                             *map(self.qr.__getitem__, control_indices), self.qr[target_index])
            self.circuit.crz(angles[0],
                             *map(self.qr.__getitem__, control_indices), self.qr[target_index])
            self.circuit.crx(-np.pi/2,
                             *map(self.qr.__getitem__, control_indices), self.qr[target_index])
            self.circuit.crz(angles[1],
                             *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCSWAP(self,
               control_indices: int | Collection[int],
               first_target_index: int,
               second_target_index: int) -> None:
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Apply the MCSWAP gate to the circuit controlled by all control indices to the target indices
        self.circuit.cswap(*map(self.qr.__getitem__, control_indices),
                           self.qr[first_target_index], self.qr[second_target_index])

    @Circuit.gatemethod
    def GlobalPhase(self,
                    angle: float) -> None:
        # TODO: Apply the Global Phase gate to the circuit
        self.circuit

    @Circuit.gatemethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
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
                        backend: Backend | None = None) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CUDAQCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            state_vector = np.array(cudaq.get_state(circuit.circuit))
        else:
            state_vector = backend.get_statevector(circuit)

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
        circuit: CUDAQCircuit = copy.deepcopy(self)

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
        circuit: CUDAQCircuit = copy.deepcopy(self)

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the unitary matrix
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