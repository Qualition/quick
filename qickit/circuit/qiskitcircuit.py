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
import numpy as np
from numpy.typing import NDArray
from typing import Literal, TYPE_CHECKING

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile # type: ignore
from qiskit.circuit.library import (RXGate, RYGate, RZGate, HGate, XGate, YGate, # type: ignore
                                    ZGate, SGate, SdgGate, TGate, TdgGate, U3Gate, # type: ignore
                                    PhaseGate, SwapGate, GlobalPhaseGate, IGate) # type: ignore
from qiskit.primitives import BackendSamplerV2 as BackendSampler # type: ignore
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
        Number of qubits bits must be integers.
    ValueError
        Number of qubits bits must be greater than 0.

    Usage
    -----
    >>> circuit = QiskitCircuit(num_qubits=2)
    """
    def __init__(
            self,
            num_qubits: int
        ) -> None:

        super().__init__(num_qubits=num_qubits)

        qr = QuantumRegister(self.num_qubits)
        cr = ClassicalRegister(self.num_qubits)
        self.circuit: QuantumCircuit = QuantumCircuit(qr, cr)

    def _single_qubit_gate(
            self,
            gate: Literal["I", "X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"],
            qubit_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        qubit_indices = [qubit_indices] if isinstance(qubit_indices, int) else qubit_indices

        # Define a mapping for the single qubit gates
        gate_mapping = {
            "I": lambda: IGate(),
            "X": lambda: XGate(),
            "Y": lambda: YGate(),
            "Z": lambda: ZGate(),
            "H": lambda: HGate(),
            "S": lambda: SGate(),
            "Sdg": lambda: SdgGate(),
            "T": lambda: TGate(),
            "Tdg": lambda: TdgGate(),
            "RX": lambda: RXGate(angle),
            "RY": lambda: RYGate(angle),
            "RZ": lambda: RZGate(angle),
            "Phase": lambda: PhaseGate(angle)
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        single_qubit_gate = gate_mapping[gate]()

        # Apply the single qubit gate to each qubit index
        for qubit_index in qubit_indices:
            self.circuit.append(single_qubit_gate, [qubit_index])

    def U3(
            self,
            angles: Sequence[float],
            qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.U3.__name__, params=locals())

        # Create a single qubit unitary gate
        u3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2])
        self.circuit.append(u3, [qubit_index])

    def SWAP(
            self,
            first_qubit_index: int,
            second_qubit_index: int
        ) -> None:

        self.process_gate_params(gate=self.SWAP.__name__, params=locals())

        # Create a SWAP gate
        swap = SwapGate()
        self.circuit.append(swap, [first_qubit_index, second_qubit_index])

    def _controlled_qubit_gate(
            self,
            gate: Literal["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "RX", "RY", "RZ", "Phase"],
            control_indices: int | Sequence[int],
            target_indices: int | Sequence[int],
            angle: float=0
        ) -> None:

        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices
        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices

        # Define a mapping for the controlled qubit gates
        gate_mapping = {
            "X": lambda: XGate().control(len(control_indices)),
            "Y": lambda: YGate().control(len(control_indices)),
            "Z": lambda: ZGate().control(len(control_indices)),
            "H": lambda: HGate().control(len(control_indices)),
            "S": lambda: SGate().control(len(control_indices)),
            "Sdg": lambda: SdgGate().control(len(control_indices)),
            "T": lambda: TGate().control(len(control_indices)),
            "Tdg": lambda: TdgGate().control(len(control_indices)),
            "RX": lambda: RXGate(angle).control(len(control_indices)),
            "RY": lambda: RYGate(angle).control(len(control_indices)),
            "RZ": lambda: RZGate(angle).control(len(control_indices)),
            "Phase": lambda: PhaseGate(angle).control(len(control_indices))
        }

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the abstraction
        controlled_gate = gate_mapping[gate]()

        # Apply the controlled gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(controlled_gate, [*control_indices[:], target_index])

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
        mcu3 = U3Gate(theta=angles[0], phi=angles[1], lam=angles[2]).control(len(control_indices))

        # Apply the MCU3 gate controlled by all control indices to each target index
        for target_index in target_indices:
            self.circuit.append(mcu3, [*control_indices[:], target_index])

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
        mcswap = SwapGate().control(len(control_indices))

        self.circuit.append(mcswap, [*control_indices[:], first_target_index, second_target_index])

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate
        global_phase = GlobalPhaseGate(angle)
        self.circuit.append(global_phase, (), ())
        self.global_phase += angle

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

        self.circuit.measure(qubit_indices, qubit_indices)

        # Set the measurement as applied
        for qubit_index in qubit_indices:
            self.measured_qubits.add(qubit_index)

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:

        if backend is None:
            state_vector = Statevector(self.circuit).data
        else:
            state_vector = backend.get_statevector(self)

        return np.array(state_vector)

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:

        num_qubits_to_measure = len(self.measured_qubits)

        if len(self.measured_qubits) == 0:
            raise ValueError("At least one qubit must be measured.")

        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = self.copy() # type: ignore

        if backend is None:
            # If no backend is provided, use the AerSimualtor
            base_backend: BackendSampler = BackendSampler(backend=AerSimulator())
            result = base_backend.run([circuit.circuit], shots=num_shots).result()

            # Extract the counts from the result
            counts = result[0].join_data().get_counts() # type: ignore

            partial_counts = {}

            # Parse the binary strings to filter out the unmeasured qubits
            for key in counts.keys():
                new_key = ''.join(key[::-1][i] for i in range(len(key)) if i in circuit.measured_qubits)
                partial_counts[new_key[::-1]] = counts[key]

            counts = partial_counts

            # Fill the counts array with zeros for the missing states
            counts = {f'{i:0{num_qubits_to_measure}b}': counts.get(f'{i:0{num_qubits_to_measure}b}', 0) \
                      for i in range(2**num_qubits_to_measure)}

            # Sort the counts by their keys (basis states)
            counts = dict(sorted(counts.items()))

        else:
            counts = backend.get_counts(circuit=circuit, num_shots=num_shots)

        return counts

    def get_depth(self) -> int:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = self.copy() # type: ignore

        # Transpile the circuit to U3 and CX gates
        circuit.transpile()

        return circuit.circuit.depth()

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the transpilation operation is inplace
        circuit: QiskitCircuit = self.copy() # type: ignore

        # Get the unitary matrix of the circuit
        unitary = Operator(circuit.circuit).data

        return np.array(unitary)

    def transpile(
            self,
            direct_transpile: bool=True,
            synthesis_method: UnitaryPreparation | None = None
        ) -> None:

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

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        if qasm_version == 2:
            return qasm2.dumps(self.circuit)
        elif qasm_version == 3:
            return qasm3.dumps(self.circuit)
        else:
            raise ValueError("The QASM version must be either 2 or 3.")

    def draw(self) -> None:
        self.circuit.draw(output='mpl')