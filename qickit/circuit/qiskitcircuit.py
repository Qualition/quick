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

import copy
import matplotlib.figure
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

# Qiskit imports
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister # type: ignore
from qiskit.circuit.library import (RXGate, RYGate, RZGate, HGate, XGate, YGate, # type: ignore
                                    ZGate, SGate, TGate, U3Gate, SwapGate, CXGate, # type: ignore
                                    CYGate, CZGate, CHGate, CSGate, CSwapGate, # tyoe: ignore
                                    GlobalPhaseGate, IGate) # type: ignore
from qiskit.primitives import BackendSampler # type: ignore
from qiskit_aer.aerprovider import AerSimulator # type: ignore
from qiskit.quantum_info import Statevector, Operator # type: ignore

# Import `qickit.circuit.Circuit`
from qickit.circuit import Circuit

# Import `qickit.backend.Backend`
if TYPE_CHECKING:
    from qickit.backend import Backend

# Import `qickit.types.collection.Collection`
from qickit.types import Collection


class QiskitCircuit(Circuit):
    """ `qickit.circuit.QiskitCircuit` is the wrapper for using IBM Qiskit in Qickit SDK.

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `num_clbits` : int
        Number of classical bits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `num_clbits` : int
        Number of classical bits in the circuit.
    `circuit` : qiskit.QuantumCircuit
        The circuit.
    `measured` : bool
        If the circuit has been measured.
    `circuit_log` : list[dict]
        The circuit log.
    """
    def __init__(self,
                 num_qubits: int,
                 num_clbits: int) -> None:
        super().__init__(num_qubits=num_qubits,
                         num_clbits=num_clbits)

        # Define the quantum bit register
        qr = QuantumRegister(self.num_qubits)
        # Define the classical bit register
        cr = ClassicalRegister(self.num_clbits)

        # Define the circuit
        self.circuit = QuantumCircuit(qr, cr)

    @Circuit.gatemethod
    def Identity(self,
                 qubit_indices: int | Collection[int]) -> None:
        # Create an Identity gate
        identity = IGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Identity gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(identity, [index])
        else:
            # If it's not a list, apply the Identity gate to the single qubit
            self.circuit.append(identity, [qubit_indices])

    @Circuit.gatemethod
    def X(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-X gate
        x = XGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the X gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(x, [index])
        else:
            # If it's not a list, apply the X gate to the single qubit
            self.circuit.append(x, [qubit_indices])

    @Circuit.gatemethod
    def Y(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Y gate
        y = YGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Y gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(y, [index])
        else:
            # If it's not a list, apply the Y gate to the single qubit
            self.circuit.append(y, [qubit_indices])

    @Circuit.gatemethod
    def Z(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Pauli-Z gate
        z = ZGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the Z gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(z, [index])
        else:
            # If it's not a list, apply the Z gate to the single qubit
            self.circuit.append(z, [qubit_indices])

    @Circuit.gatemethod
    def H(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Hadamard gate
        h = HGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the H gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(h, [index])
        else:
            # If it's not an list, apply the H gate to the single qubit
            self.circuit.append(h, [qubit_indices])

    @Circuit.gatemethod
    def S(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-S gate
        s = SGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the S gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(s, [index])
        else:
            # If it's not a list, apply the S gate to the single qubit
            self.circuit.append(s, [qubit_indices])

    @Circuit.gatemethod
    def T(self,
          qubit_indices: int | Collection[int]) -> None:
        # Create a Clifford-T gate
        t = TGate()

        # Check if the qubit_indices is a list
        if isinstance(qubit_indices, Collection):
            # If it is, apply the T gate to each qubit in the list
            for index in qubit_indices:
                self.circuit.append(t, [index])
        else:
            # If it's not a list, apply the T gate to the single qubit
            self.circuit.append(t, [qubit_indices])

    @Circuit.gatemethod
    def RX(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RX gate with the specified angle
        rx = RXGate(angle)
        # Apply the RX gate to the circuit at the specified qubit
        self.circuit.append(rx, [qubit_index])

    @Circuit.gatemethod
    def RY(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RY gate with the specified angle
        ry = RYGate(angle)
        # Apply the RY gate to the circuit at the specified qubit
        self.circuit.append(ry, [qubit_index])

    @Circuit.gatemethod
    def RZ(self,
           angle: float,
           qubit_index: int) -> None:
        # Create an RZ gate with the specified angle
        rz = RZGate(angle)
        # Apply the RZ gate to the circuit at the specified qubit
        self.circuit.append(rz, [qubit_index])

    @Circuit.gatemethod
    def U3(self,
           angles: Collection[float],
           qubit_index: int) -> None:
        # Create a single qubit unitary gate
        u3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2])
        # Apply the U3 gate to the circuit at the specified qubit
        self.circuit.append(u3, [qubit_index])

    @Circuit.gatemethod
    def SWAP(self,
             first_qubit: int,
             second_qubit: int) -> None:
        # Create a SWAP gate
        swap = SwapGate()
        # Apply the SWAP gate to the circuit at the specified qubits
        self.circuit.append(swap, [first_qubit, second_qubit])

    @Circuit.gatemethod
    def CX(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-X gate
        cx = CXGate()
        # Apply the CX gate to the circuit at the specified control and target qubits
        self.circuit.append(cx, [control_index, target_index])

    @Circuit.gatemethod
    def CY(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Y gate
        cy = CYGate()
        # Apply the CY gate to the circuit at the specified control and target qubits
        self.circuit.append(cy, [control_index, target_index])

    @Circuit.gatemethod
    def CZ(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-Z gate
        cz = CZGate()
        # Apply the CZ gate to the circuit at the specified control and target qubits
        self.circuit.append(cz, [control_index, target_index])

    @Circuit.gatemethod
    def CH(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-H gate
        ch = CHGate()
        # Apply the CH gate to the circuit at the specified control and target qubits
        self.circuit.append(ch, [control_index, target_index])

    @Circuit.gatemethod
    def CS(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-S gate
        cz = CSGate()
        # Apply the CS gate to the circuit at the specified control and target qubits
        self.circuit.append(cz, [control_index, target_index])

    @Circuit.gatemethod
    def CT(self,
           control_index: int,
           target_index: int) -> None:
        # Create a Controlled-T gate
        ct = TGate().control(1)
        # Apply the CT gate to the circuit at the specified control and target qubits
        self.circuit.append(ct, [control_index, target_index])

    @Circuit.gatemethod
    def CRX(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RX gate with the specified angle
        crx = RXGate(angle).control(1)
        # Apply the CRX gate to the circuit at the specified control and target qubits
        self.circuit.append(crx, [control_index, target_index])

    @Circuit.gatemethod
    def CRY(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RY gate with the specified angle
        cry = RYGate(angle).control(1)
        # Apply the CRY gate to the circuit at the specified control and target qubits
        self.circuit.append(cry, [control_index, target_index])

    @Circuit.gatemethod
    def CRZ(self,
            angle: float,
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-RZ gate with the specified angle
        crz = RZGate(angle).control(1)
        # Apply the CRZ gate to the circuit at the specified control and target qubits
        self.circuit.append(crz, [control_index, target_index])

    @Circuit.gatemethod
    def CU3(self,
            angles: Collection[float],
            control_index: int,
            target_index: int) -> None:
        # Create a Controlled-U3 gate with the specified angles
        cu3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2]).control(1)
        # Apply the CU3 gate to the circuit at the specified control and target qubits
        self.circuit.append(cu3, [control_index, target_index])

    @Circuit.gatemethod
    def CSWAP(self,
              control_index: int,
              first_target_index: int,
              second_target_index: int) -> None:
        # Create a Controlled-SWAP gate
        cswap = CSwapGate()
        # Apply the CSWAP gate to the circuit at the specified control and target qubits
        self.circuit.append(cswap, [control_index, first_target_index, second_target_index])

    @Circuit.gatemethod
    def MCX(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled X gate with the number of control qubits equal to
        # the length of control_indices
        mcx = XGate().control(len(control_indices))

        # Apply the MCX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcx, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCY(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled Y gate with the number of control qubits equal to
        # the length of control_indices
        mcy = YGate().control(len(control_indices))

        # Apply the MCY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcy, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCZ(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled Z gate with the number of control qubits equal to
        # the length of control_indices
        mcz = ZGate().control(len(control_indices))

        # Apply the MCZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcz, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCH(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled H gate with the number of control qubits equal to
        # the length of control_indices
        mch = HGate().control(len(control_indices))

        # Apply the MCH gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mch, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCS(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled S gate with the number of control qubits equal to
        # the length of control_indices
        mcs = SGate().control(len(control_indices))

        # Apply the MCS gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcs, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCT(self,
            control_indices: int | Collection[int],
            target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create an Multi-Controlled T gate with the number of control qubits equal to
        # the length of control_indices
        mct = TGate().control(len(control_indices))

        # Apply the MCT gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mct, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCRX(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled RX gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        mcrx = RXGate(angle).control(len(control_indices))

        # Apply the MCRX gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcrx, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCRY(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled RY gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        mcry = RYGate(angle).control(len(control_indices))

        # Apply the MCRY gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcry, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCRZ(self,
             angle: float,
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled RZ gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        mcrz = RZGate(angle).control(len(control_indices))

        # Apply the MCRZ gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcrz, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCU3(self,
             angles: Collection[float],
             control_indices: int | Collection[int],
             target_indices: int | Collection[int]) -> None:
        # Ensure control_indices and target_indices are always treated as lists
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Create a Multi-Controlled U3 gate with the number of control qubits equal to
        # the length of control_indices with the specified angle
        mcu3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2]).control(len(control_indices))

        # Apply the MCU3 gate to the circuit at the control and target qubits
        for target_index in target_indices:
            self.circuit.append(mcu3, control_indices[:] + [target_index])

    @Circuit.gatemethod
    def MCSWAP(self,
               control_indices: int | Collection[int],
               first_target_index: int,
               second_target_index: int) -> None:
        # Ensure control_indices is always treated as a list
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Create a Multi-Controlled SWAP gate with the number of control qubits equal to
        # the length of control_indices
        mcswap = SwapGate().control(len(control_indices))

        # Apply the MCSWAP gate to the circuit at the control and target qubits
        self.circuit.append(mcswap, control_indices[:] + [first_target_index, second_target_index])

    @Circuit.gatemethod
    def GlobalPhase(self,
                    angle: float) -> None:
        # Create a Global Phase gate
        global_phase = GlobalPhaseGate(angle)

        # Apply the Global Phase gate to the circuit
        self.circuit.append(global_phase, (), ())

    @Circuit.gatemethod
    def measure(self,
                qubit_indices: int | Collection[int]) -> None:
        # Measure the qubits
        self.circuit.measure(qubit_indices, qubit_indices)

        # Set the measurement as applied
        self.measured = True

    def get_statevector(self,
                        backend: Backend | None = None) -> Collection[float]:
        if backend is None:
            # Run the circuit and define the state vector
            state_vector = Statevector(self.circuit).data

        else:
            # Run the circuit on the specified backend and define the state vector
            state_vector = backend.get_statevector(self)

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

        return state_vector

    def get_counts(self,
                   num_shots: int,
                   backend: Backend | None = None) -> dict[str, int]:
        if self.measured is False:
            self.measure(range(self.num_qubits))

        if backend is None:
            # If no backend is provided, use the AerSimualtor
            base_backend: BackendSampler = BackendSampler(AerSimulator())
            # Run the circuit
            result = base_backend.run(self.circuit, shots=num_shots, seed_simulator=0).result()
            # Extract the quasi-probability distribution from the first result
            quasi_dist = result.quasi_dists[0]
            # Convert the quasi-probability distribution to counts
            counts = {bin(k)[2:].zfill(self.num_qubits): int(v * num_shots) for k, v in quasi_dist.items()}
            # Fill the counts array with zeros for the missing states
            counts = {f'{i:0{self.num_qubits}b}': counts.get(f'{i:0{self.num_qubits}b}', 0) for i in range(2**self.num_qubits)}
            # Sort the counts by their keys (basis states)
            counts = dict(sorted(counts.items()))

        else:
            # Run the circuit on the specified backend
            counts = backend.get_counts(self, num_shots)

        return counts

    def get_depth(self) -> int:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = copy.deepcopy(self)

        # Transpile the circuit to U3 and CX gates
        circuit.transpile()

        return circuit.circuit.depth()

    def get_unitary(self) -> NDArray[np.number]:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = copy.deepcopy(self)

        # Get the unitary matrix of the circuit
        unitary = Operator(circuit.circuit).data

        return np.array(unitary)

    def to_qasm(self) -> str:
        # Convert the circuit to QASM
        qasm = self.circuit.qasm()

        return qasm

    def draw(self) -> matplotlib.figure.Figure:
        return self.circuit.draw(output='mpl')