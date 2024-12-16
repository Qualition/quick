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

""" Wrapper class for using Google's Cirq in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["CirqCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING

import cirq
from cirq.ops import Rx, Ry, Rz, X, Y, Z, H, S, T, I

if TYPE_CHECKING:
    from qickit.backend import Backend
from qickit.circuit import Circuit
from qickit.circuit.circuit import GATES


class U3(cirq.Gate):
    def __init__(self, angles: Sequence[float]) -> None:
        super(U3, self)
        self.angles = angles

    def _num_qubits_(self) -> int:
        return 1

    def _unitary_(self) -> NDArray[np.complex128]:
        angles = self.angles

        u3 = [
            [np.cos(angles[0]/2), -np.exp(1j*angles[2]) * np.sin(angles[0]/2)],
            [np.exp(1j*angles[1]) * np.sin(angles[0]/2), np.exp(1j*(angles[1] + angles[2])) * \
            np.cos(angles[0]/2)]
        ]

        return np.array(u3)

    def _circuit_diagram_info_(self) -> str:
        return "U3"


class CirqCircuit(Circuit):
    """ `qickit.circuit.CirqCircuit` is the wrapper for using Google's Cirq in Qickit SDK.

    Notes
    -----
    Google's Cirq is a Python framework for creating, editing, and invoking Noisy Intermediate
    Scale Quantum (NISQ) circuits.

    For more information on Cirq:
    - Documentation:
    https://quantumai.google/reference/python/cirq/all_symbols
    - Source code:
    https://github.com/quantumlib/Cirq

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
    `gate_mapping` : dict[str, Callable]
        The mapping of the gates in the input quantum computing
        framework to the gates in qickit.
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
        - Number of qubits bits must be integers.
    ValueError
        - Number of qubits bits must be greater than 0.

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
        # We also need to apply the identity gate to all qubits to ensure that the
        # unitary accounts for all qubits, including idle ones
        self.circuit: cirq.Circuit = cirq.Circuit()
        for i in range(self.num_qubits):
            self.circuit.append(I(self.qr[i]))

    @staticmethod
    def _define_gate_mapping() -> dict[str, Callable]:
        # Define lambda factory for non-parameterized gates
        def const(x):
            return lambda _angles: x

        # Note that Qickit only uses U3, CX, and Global Phase gates and constructs the other gates
        # by performing decomposition
        # However, if the user wants to override the decomposition and use the native gates, they
        # can do so by using the below gate mapping
        gate_mapping = {
            "I": const(I),
            "X": const(X),
            "Y": const(Y),
            "Z": const(Z),
            "H": const(H),
            "S": const(S),
            "Sdg": const(S**-1),
            "T": const(T),
            "Tdg": const(T**-1),
            "RX": lambda angles: Rx(rads=angles[0]),
            "RY": lambda angles: Ry(rads=angles[0]),
            "RZ": lambda angles: Rz(rads=angles[0]),
            "Phase": lambda angles: cirq.ZPowGate(exponent=angles[0]/np.pi),
            "U3": lambda angles: U3(angles)
        }

        return gate_mapping

    def _gate_mapping(
            self,
            gate: GATES,
            target_indices: int | Sequence[int],
            control_indices: int | Sequence[int] = [],
            angles: Sequence[float] = [0, 0, 0]
        ) -> None:

        target_indices = [target_indices] if isinstance(target_indices, int) else target_indices
        control_indices = [control_indices] if isinstance(control_indices, int) else control_indices

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the polymorphism
        gate_operation = self.gate_mapping[gate](angles)

        if control_indices:
            gate_operation = cirq.ControlledGate(
                sub_gate=gate_operation,
                num_controls=len(control_indices)
            )

            for target_index in target_indices:
                self.circuit.append(
                    gate_operation(
                        *map(self.qr.__getitem__, control_indices),
                        self.qr[target_index]
                    )
                )
            return

        for target_index in target_indices:
            self.circuit.append(gate_operation(self.qr[target_index]))

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Create a Global Phase gate (Cirq takes in e^i*angle as the argument)
        global_phase = cirq.GlobalPhaseGate(np.exp(1j*angle))

        self.circuit.append(global_phase())
        self.global_phase += angle

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        # We must sort the indices as Cirq interprets that the order of measurements
        # is relevant
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

        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = self.copy() # type: ignore

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        if backend is None:
            state_vector = circuit.circuit.final_state_vector(qubit_order=self.qr)
        else:
            state_vector = backend.get_statevector(circuit)

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

    def get_unitary(self) -> NDArray[np.complex128]:
        # Copy the circuit as the operations are applied inplace
        circuit: CirqCircuit = self.copy() # type: ignore

        # Cirq uses MSB convention for qubits, so we need to reverse the qubit indices
        circuit.vertical_reverse()

        # Define the unitary matrix
        unitary = cirq.unitary(circuit.circuit)

        return np.array(unitary)

    def reset_qubit(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.reset_qubit.__name__, params=locals())

        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]

        for qubit_index in qubit_indices:
            self.circuit.append(cirq.ResetChannel()(self.qr[qubit_index]))

    def to_qasm(
            self,
            qasm_version: int=2
        ) -> str:

        from qickit.circuit import QiskitCircuit

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        print(self.circuit)