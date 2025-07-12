# Copyright 2023-2025 Qualition Computing LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Qualition/quick/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Wrapper class for using Quimb in quick SDK.
"""

from __future__ import annotations

__all__ = ["QuimbCircuit"]

from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray
from typing import Callable, TYPE_CHECKING

import quimb.tensor as qtn # type: ignore
from quimb.gates import I, X, Y, Z, H, S, T, RX, RY, RZ, U3 # type: ignore
from quimb.gen.operators import ncontrolled_gate as control # type: ignore

if TYPE_CHECKING:
    from quick.backend import Backend
from quick.circuit import Circuit
from quick.circuit.circuit import GATES


class QuimbCircuit(Circuit):
    """ `quick.circuit.QuimbCircuit` is the wrapper for using Quimb in quick SDK.

    Notes
    -----
    Quimb is A python library for quantum information and many-body calculations
    including tensor networks.

    For more information on Quimb:
    - Documentation:
    https://quimb.readthedocs.io/en/latest/
    - Source code:
    https://github.com/jcmgray/quimb

    Parameters
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.

    Attributes
    ----------
    `num_qubits` : int
        Number of qubits in the circuit.
    `circuit` : qtn.Circuit
        The circuit.
    `gate_mapping` : dict[str, Callable]
        The mapping of the gates in the input quantum computing
        framework to the gates in quick.
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

        self.circuit: qtn.Circuit = qtn.Circuit(num_qubits)

        # We also need to apply the identity gate to all qubits to ensure that the
        # unitary accounts for all qubits, including idle ones
        for i in range(num_qubits):
            self.circuit.apply_gate(I, i)

    @staticmethod
    def _define_gate_mapping() -> dict[str, Callable]:
        # Define lambda factory for non-parameterized gates
        def const(x):
            return lambda _angles: x

        # Note that quick only uses U3, CX, and Global Phase gates and constructs the other gates
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
            "Sdg": const(U3(0, 0, -np.pi/2)),
            "T": const(T),
            "Tdg": const(U3(0, 0, -np.pi/4)),
            "RX": lambda angles: RX(angles[0]),
            "RY": lambda angles: RY(angles[0]),
            "RZ": lambda angles: RZ(angles[0]),
            "Phase": lambda angles: U3(0, 0, angles[0]),
            "U3": lambda angles: U3(*angles)
        }

        return gate_mapping

    def _gate_mapping(
            self,
            gate: GATES,
            target_indices: int | Sequence[int],
            control_indices: int | Sequence[int] = [],
            angles: Sequence[float] = (0, 0, 0)
        ) -> None:

        targets = [target_indices] if isinstance(target_indices, int) else list(target_indices)
        controls = [control_indices] if isinstance(control_indices, int) else list(control_indices)

        # Given Quimb uses MSB convention, we will explicitly
        # convert the qubit indices to LSB convention
        # This will help performance by avoiding `circuit.vertical_reverse()` calls
        for i, index in enumerate(targets):
            targets[i] = self.num_qubits - 1 - index

        for i, index in enumerate(controls):
            controls[i] = self.num_qubits - 1 - index

        # Lazily extract the value of the gate from the mapping to avoid
        # creating all the gates at once, and to maintain the polymorphism
        gate_operation: qtn.Gate = self.gate_mapping[gate](angles)

        if controls:
            gate_operation = control(ncontrol=len(controls), gate=gate_operation) # type: ignore

            for target_index in targets:
                self.circuit.apply_gate(
                    gate_operation,
                    *controls,
                    target_index
                )
            return

        for target_index in targets:
            self.circuit.apply_gate(gate_operation, target_index)

    def GlobalPhase(
            self,
            angle: float
        ) -> None:

        self.process_gate_params(gate=self.GlobalPhase.__name__, params=locals())

        # Quimb currently does not have a native global phase gate, thus we simply
        # multiply the state and/or the unitary by e^(i*angle) to apply the global
        # phase
        # The counts are not affected by the global phase
        self.global_phase += angle

    def measure(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        self.process_gate_params(gate=self.measure.__name__, params=locals())

        qubits = [qubit_indices] if isinstance(qubit_indices, int) else list(qubit_indices)

        # Given Quimb uses MSB convention, we will explicitly
        # convert the qubit indices to LSB convention
        # This will help performance by avoiding `circuit.vertical_reverse()` calls
        for i, index in enumerate(qubits):
            qubits[i] = self.num_qubits - 1 - index

        for qubit_index in qubits:
            self.measured_qubits.add(qubit_index)

    def get_statevector(
            self,
            backend: Backend | None = None,
        ) -> NDArray[np.complex128]:

        if backend is None:
            psi: qtn.tensor_arbgeom.TensorNetworkGenVector = self.circuit.psi
            state_vector = np.array(psi.to_dense()).flatten()

            # Apply the global phase to the state vector
            state_vector *= np.exp(1j * self.global_phase)
        else:
            state_vector = backend.get_statevector(self)

        return state_vector

    def get_counts(
            self,
            num_shots: int,
            backend: Backend | None = None
        ) -> dict[str, int]:

        if len(self.measured_qubits) == 0:
            raise ValueError("At least one qubit must be measured.")

        if backend is None:
            samples = self.circuit.sample(C=num_shots, qubits=self.measured_qubits)

            counts: dict[str, int] = {}
            for sample in samples:
                key = "".join(str(bit) for bit in sample)
                counts[key] = counts.get(key, 0) + 1

        else:
            counts = backend.get_counts(circuit=self, num_shots=num_shots)

        return counts

    def get_unitary(self) -> NDArray[np.complex128]:
        uni: qtn.tensor_arbgeom.TensorNetworkGenOperator = self.circuit.get_uni()
        unitary = np.array(uni.to_dense())

        # Apply the global phase to the unitary
        unitary *= np.exp(1j * self.global_phase)

        return unitary

    def get_tensor_network(self) -> qtn.TensorNetwork:
        """ Get the tensor network representation of the circuit.

        Notes
        -----
        The tensor network follows MSB convention for compatibility
        with quimb. To avoid circular imports, this functionality is
        only accessible through the `QuimbCircuit` class.

        Returns
        -------
        `tensor_network` : qtn.TensorNetwork
            The tensor network representation of the circuit.

        Usage
        -----
        >>> circuit = QuimbCircuit(num_qubits=2)
        >>> circuit.get_tensor_network()
        """
        tensor_network = qtn.TensorNetwork(self.circuit.psi)

        for gate in tensor_network:
            gate.drop_tags([f"I{i}" for i in range(self.num_qubits)])

        # Apply the global phase to the tensor network
        tensor_network *= np.exp(1j * self.global_phase)

        return tensor_network

    def reset_qubit(
            self,
            qubit_indices: int | Sequence[int]
        ) -> None:

        raise NotImplementedError("Quimb currently does not support resetting qubits.")

    def to_qasm(
            self,
            qasm_version: int = 2
        ) -> str:

        from quick.circuit import QiskitCircuit

        return self.convert(QiskitCircuit).to_qasm(qasm_version=qasm_version)

    def draw(self) -> None:
        self.circuit.draw(figsize=(3, 3))