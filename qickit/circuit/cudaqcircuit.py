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
from typing import TYPE_CHECKING

# CUDA-Q imports
import cudaq # type: ignore

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit, QiskitCircuit

# Import `qickit.backend.Backend`
if TYPE_CHECKING:
    from qickit.backend import Backend

# import `qickit.synthesis.unitarypreparation.QiskitUnitaryTranspiler`
from qickit.synthesis.unitarypreparation import UnitaryPreparation

# Import `qickit.types.collection.Collection`
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
    """
    def __init__(self,
                 num_qubits: int) -> None:
        super().__init__(num_qubits=num_qubits)

        # Initialize the cudaq kernel
        self.circuit = cudaq.make_kernel()

        # Define the quantum bit register
        self.qr = self.circuit.qalloc(self.num_qubits)

    @Circuit.gatemethod
    def Identity(self,
                 qubit_indices: int | Collection[int]) -> None:
        # cuda-quantum does not have an Identity gate
        pass

    @Circuit.gatemethod
    def X(self,
          qubit_indices: int | Collection[int]) -> None:
        # Check if the qubit_indices is a collection
        if isinstance(qubit_indices, Collection):
            # If it is, apply the X gate to each qubit in the collection
            for index in qubit_indices:
                self.circuit.x(self.qr[index])
        else:
            # If it's not a collection, apply the X gate to the single qubit
            self.circuit.x(self.qr[qubit_indices])

    @Circuit.gatemethod
    def Y(self,
          qubit_indices: int | Collection[int]) -> None:
        # Check if the qubit_indices is a collection
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Y gate to each qubit in the collection
            for index in qubit_indices:
                self.circuit.y(self.qr[index])
        else:
            # If it's not a collection, apply the Y gate to the single qubit
            self.circuit.y(self.qr[qubit_indices])

    @Circuit.gatemethod
    def Z(self,
          qubit_indices: int | Collection[int]) -> None:
        # Check if the qubit_indices is a collection
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Z gate to each qubit in the collection
            for index in qubit_indices:
                self.circuit.z(self.qr[index])
        else:
            # If it's not a collection, apply the Z gate to the single qubit
            self.circuit.z(self.qr[qubit_indices])

    @Circuit.gatemethod
    def H(self,
          qubit_indices: int | Collection[int]) -> None:
        # Check if the qubit_indices is a collection
        if isinstance(qubit_indices, Collection):
            # If it is, apply the H gate to each qubit in the collection
            for index in qubit_indices:
                self.circuit.h(self.qr[index])
        else:
            # If it's not a collection, apply the H gate to the single qubit
            self.circuit.h(self.qr[qubit_indices])

    @Circuit.gatemethod
    def S(self,
          qubit_indices: int | Collection[int]) -> None:
        # Check if the qubit_indices is a collection
        if isinstance(qubit_indices, Collection):
            # If it is, apply the S gate to each qubit in the collection
            for index in qubit_indices:
                self.circuit.s(self.qr[index])
        else:
            # If it's not a collection, apply the S gate to the single qubit
            self.circuit.s(self.qr[qubit_indices])

    @Circuit.gatemethod
    def T(self,
          qubit_indices: int | Collection[int]) -> None:
        # Check if the qubit_indices is a collection
        if isinstance(qubit_indices, Collection):
            # If it is, apply the T gate to each qubit in the collection
            for index in qubit_indices:
                self.circuit.t(self.qr[index])
        else:
            # If it's not a collection, apply the T gate to the single qubit
            self.circuit.t(self.qr[qubit_indices])

    @Circuit.gatemethod
    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.rx(angle, self.qr[qubit_index])

    @Circuit.gatemethod
    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.ry(angle, self.qr[qubit_index])

    @Circuit.gatemethod
    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.rz(angle, self.qr[qubit_index])

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # Apply the U3 gate to the circuit at the specified qubit
        # NOTE: CUDAQ version 0.8 will provide native support of U3 gates
        self.circuit.rz(angles[2], self.qr[qubit_index])
        self.circuit.rx(np.pi/2, self.qr[qubit_index])
        self.circuit.rz(angles[0], self.qr[qubit_index])
        self.circuit.rx(-np.pi/2, self.qr[qubit_index])
        self.circuit.rz(angles[1], self.qr[qubit_index])

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        # Apply the SWAP gate to the circuit at the specified qubits
        self.circuit.swap(self.qr[first_qubit], self.qr[second_qubit])

    @Circuit.gatemethod
    def CX(self,
           control_index: int,
           target_index: int) -> None:
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.cx(self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CY(self,
           control_index: int,
           target_index: int) -> None:
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.cy(self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.cz(self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CH(self,
           control_index: int,
           target_index: int) -> None:
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.ch(self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CS(self,
           control_index: int,
           target_index: int) -> None:
        # Apply the CS gate to the circuit at the specified control and target qubits
        self.circuit.cs(self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CT(self,
           control_index: int,
           target_index: int) -> None:
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.ct(self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Apply the CRX gate to the circuit at the specified control and target qubits
        self.circuit.crx(angle, self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.cry(angle, self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.crz(angle, self.qr[control_index], self.qr[target_index])

    @Circuit.gatemethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        # Apply the CU3 gate to the circuit at the specified control and target qubits
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
        # Apply the CSWAP gate to the circuit at the specified control and target qubits
        self.circuit.cswap(self.qr[control_index],
                           self.qr[first_target_index], self.qr[second_target_index])

    @Circuit.gatemethod
    def MCX(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.cx(*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCY(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.cy(*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCZ(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.cz(*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCH(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCH gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.ch(*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCS(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCS gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.cs(*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCT(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCT gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.ct(*map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCRX(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.crx(angle, *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCRY(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.cry(angle, *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCRZ(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.crz(angle, *map(self.qr.__getitem__, control_indices), self.qr[target_index])

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Apply the MCU3 gate to the circuit at the control and target qubits
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
        # Ensure control_indices is always treated as a list
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Apply the MCSWAP gate to the circuit at the control and target qubits
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
            # Define the state vector
            state_vector = np.array(cudaq.get_state(circuit.circuit))

        else:
            # Run the circuit on the specified backend and define the state vector
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
            # Run the circuit to get the result
            result = str(cudaq.sample(circuit.circuit, num_shots)).split()[1:-1]
            # Format the result as a dictionary to get the counts
            counts = {pair.split(":")[0]: int(pair.split(":")[1]) for pair in result}

        else:
            # Run the circuit on the specified backend
            counts = backend.get_counts(circuit, num_shots)

        return counts

    def get_depth(self) -> int:
        # Convert the circuit to Qiskit
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
        # Convert to `qickit.circuit.QiskitCircuit`
        qiskit_circuit = self.convert(QiskitCircuit)

        # Transpile the circuit
        qiskit_circuit.transpile(direct_transpile=direct_transpile,
                                 synthesis_method=synthesis_method)

        # Convert back to `qickit.circuit.CUDAQCircuit`
        updated_circuit = qiskit_circuit.convert(CUDAQCircuit)
        self.circuit_log = updated_circuit.circuit_log
        self.circuit = updated_circuit.circuit

    def to_qasm(self,
                qasm_version: int=2) -> str:
        # Convert the circuit to QASM
        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        print(cudaq.draw(self.circuit))