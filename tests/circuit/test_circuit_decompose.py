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

from __future__ import annotations

__all__ = ["TestDecompose"]

import numpy as np
import pytest

from quick.circuit import Circuit

from tests.circuit import CIRCUIT_FRAMEWORKS

# Define the primitive gate types to be tested
GATE_TYPES = [
    (lambda circuit: circuit.U3([np.pi, 0, np.pi], 0), 1),
    (lambda circuit: circuit.CX(0, 1), 2),
    (lambda circuit: circuit.GlobalPhase(np.pi), 1),
    (lambda circuit: circuit.measure(0), 1)
]


class TestDecompose:
    """ `tests.circuit.TestDecompose` is the tester class for `quick.circuit.Circuit`'s
    logging and decomposition to lower-level gates.
    """
    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    def test_X(
            self,
            framework: type[Circuit]
        ) -> None:
        """ Test the decomposition of the X gate.

        Notes
        -----
        We test the decomposition of a single qubit gate with a
        direct implementation to U3 within one rep.

        Parameters
        ----------
        framework : type[quick.circuit.Circuit]
            The framework to be tested.
        """
        # Define the circuit and apply the X gate
        circuit: Circuit = framework(1)
        circuit.X(0)

        # Decompose the circuit
        decomposed_circuit = circuit.decompose()

        # Check the decomposed circuit
        checker_circuit: Circuit = framework(1)
        checker_circuit.U3([np.pi, 0, np.pi], 0)

        assert decomposed_circuit == checker_circuit

    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    def test_Z(
            self,
            framework: type[Circuit]
        ) -> None:
        """ Test the decomposition of the Z gate.

        Notes
        -----
        We test the decomposition of a single qubit gate which
        is indirectly implemented by U3. This test covers the
        `reps` parameter in the `decompose` method.

        Parameters
        ----------
        framework : type[quick.circuit.Circuit]
            The framework to be tested.
        """
        # Define the circuit and apply the Z gate
        circuit: Circuit = framework(1)
        circuit.Z(0)

        # Decompose the circuit
        decomposed_circuit = circuit.decompose()

        # Check the decomposed circuit
        checker_circuit: Circuit = framework(1)
        checker_circuit.Phase(np.pi, 0)

        assert decomposed_circuit == checker_circuit

        # Decompose the circuit with 2 reps
        decomposed_circuit = circuit.decompose(reps=2)

        # Check the decomposed circuit
        checker_circuit = framework(1)
        checker_circuit.U3([0, 0, np.pi], 0)

        assert decomposed_circuit == checker_circuit

        # Decompose the circuit with `full` set to True
        decomposed_circuit = circuit.decompose(full=True)

        # Check the decomposed circuit
        checker_circuit = framework(1)
        checker_circuit.U3([0, 0, np.pi], 0)

        assert decomposed_circuit == checker_circuit

    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    def test_CZ(
            self,
            framework: type[Circuit]
        ) -> None:
        """ Test the decomposition of the CZ gate.

        Notes
        -----
        We test the decomposition of a two-qubit gate which is
        indirectly implemented by CX and Z gates.

        Parameters
        ----------
        framework : type[quick.circuit.Circuit]
            The framework to be tested.
        """
        # Define the circuit and apply the CZ gate
        circuit: Circuit = framework(2)
        circuit.CZ(0, 1)

        # Decompose the circuit
        decomposed_circuit = circuit.decompose()

        # Check the decomposed circuit
        checker_circuit: Circuit = framework(2)
        checker_circuit.H(1)
        checker_circuit.CX(0, 1)
        checker_circuit.H(1)

        assert decomposed_circuit == checker_circuit

        # Decompose the circuit with 2 reps
        decomposed_circuit = circuit.decompose(reps=2)

        # Check the decomposed circuit
        checker_circuit = framework(2)
        checker_circuit.U3([np.pi/2, 0, np.pi], 1)
        checker_circuit.CX(0, 1)
        checker_circuit.U3([np.pi/2, 0, np.pi], 1)

        assert decomposed_circuit == checker_circuit

        # Decompose the circuit with `full` set to True
        decomposed_circuit = circuit.decompose(full=True)

        # Check the decomposed circuit
        checker_circuit = framework(2)
        checker_circuit.U3([np.pi/2, 0, np.pi], 1)
        checker_circuit.CX(0, 1)
        checker_circuit.U3([np.pi/2, 0, np.pi], 1)

        assert decomposed_circuit == checker_circuit

    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    def test_MCX(
            self,
            framework: type[Circuit]
        ) -> None:
        """ Test the decomposition of the MCX gate.

        Notes
        -----
        We test the decomposition of a multi-qubit gate which is
        indirectly implemented by CX and Z gates.

        Parameters
        ----------
        framework : type[quick.circuit.Circuit]
            The framework to be tested.
        """
        # Define the circuit and apply the MCX gate
        circuit: Circuit = framework(4)
        circuit.MCX([0, 1], [2, 3])

        # Decompose the circuit
        decomposed_circuit = circuit.decompose()

        # Check the decomposed circuit
        checker_circuit: Circuit = framework(4)
        checker_circuit.H([2, 3])
        checker_circuit.MCPhase(np.pi, [0, 1], [2, 3])
        checker_circuit.H([2, 3])

        assert decomposed_circuit == checker_circuit

        # Decompose the circuit with 2 reps
        decomposed_circuit = circuit.decompose(reps=2)

        # Check the decomposed circuit
        checker_circuit = framework(4)
        checker_circuit.U3([np.pi/2, 0, np.pi], [2, 3])
        checker_circuit.MCRZ(np.pi, [0, 1], 2)
        checker_circuit.MCRZ(np.pi/2, 0, 1)
        checker_circuit.Phase(np.pi/4, 0)
        checker_circuit.MCRZ(np.pi, [0, 1], 3)
        checker_circuit.MCRZ(np.pi/2, 0, 1)
        checker_circuit.Phase(np.pi/4, 0)
        checker_circuit.U3([np.pi/2, 0, np.pi], [2, 3])

        assert decomposed_circuit == checker_circuit

        # Decompose the circuit with `full` set to True
        decomposed_circuit = circuit.decompose(full=True)

        # Check the decomposed circuit
        # We use 0.3926990816987242 to account for floating point errors after 10th decimal
        checker_circuit = framework(4)
        checker_circuit.U3([np.pi/2, 0, np.pi], [2, 3])
        checker_circuit.CX(0, 2)
        checker_circuit.U3([0.0, -0.3926990816987242, -0.3926990816987242], 2)
        checker_circuit.GlobalPhase(0.3926990816987242)
        checker_circuit.CX(1, 2)
        checker_circuit.GlobalPhase(-0.3926990816987242)
        checker_circuit.U3([0.0, 0.3926990816987242, 0.3926990816987242], 2)
        checker_circuit.CX(0, 2)
        checker_circuit.U3([0.0, -0.3926990816987242, -0.3926990816987242], 2)
        checker_circuit.GlobalPhase(0.3926990816987242)
        checker_circuit.CX(1, 2)
        checker_circuit.GlobalPhase(-0.3926990816987242)
        checker_circuit.U3([0.0, 0.3926990816987242, 0.3926990816987242], 2)
        checker_circuit.U3([np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.U3([-np.pi/4, 0, 0], 1)
        checker_circuit.U3([-np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.CX(0, 1)
        checker_circuit.U3([np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.U3([np.pi/4, 0, 0], 1)
        checker_circuit.U3([-np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.CX(0, 1)
        checker_circuit.U3([0, 0, np.pi/4], 0)
        checker_circuit.CX(0, 3)
        checker_circuit.U3([0.0, -0.3926990816987242, -0.3926990816987242], 3)
        checker_circuit.GlobalPhase(0.3926990816987242)
        checker_circuit.CX(1, 3)
        checker_circuit.GlobalPhase(-0.3926990816987242)
        checker_circuit.U3([0.0, 0.3926990816987242, 0.3926990816987242], 3)
        checker_circuit.CX(0, 3)
        checker_circuit.U3([0.0, -0.3926990816987242, -0.3926990816987242], 3)
        checker_circuit.GlobalPhase(0.3926990816987242)
        checker_circuit.CX(1, 3)
        checker_circuit.GlobalPhase(-0.3926990816987242)
        checker_circuit.U3([0.0, 0.3926990816987242, 0.3926990816987242], 3)
        checker_circuit.U3([np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.U3([-np.pi/4, 0, 0], 1)
        checker_circuit.U3([-np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.CX(0, 1)
        checker_circuit.U3([np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.U3([np.pi/4, 0, 0], 1)
        checker_circuit.U3([-np.pi/2, -np.pi/2, np.pi/2], 1)
        checker_circuit.CX(0, 1)
        checker_circuit.U3([0, 0, np.pi/4], 0)
        checker_circuit.U3([np.pi/2, 0, np.pi], [2, 3])

        assert decomposed_circuit == checker_circuit

    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("gate_func, num_qubits", GATE_TYPES)
    def test_primitive_gates(
            self,
            framework: type[Circuit],
            gate_func,
            num_qubits: int
        ) -> None:
        """ Test the decomposition of the primitive gates.

        Parameters
        ----------
        `framework` : type[quick.circuit.Circuit]
            The circuit framework to use.
        `gate_func` : Callable[[Circuit], None]
            The gate function to apply.
        `num_qubits` : int
            The number of qubits in the circuit.
        """
        # Define the circuit and apply the gate
        circuit: Circuit = framework(num_qubits)
        gate_func(circuit)

        assert circuit.circuit_log[-1]["definition"] == []

        # Decompose the circuit
        decomposed_circuit = circuit.decompose()

        # Check the decomposed circuit
        # Assert it's unchanged
        checker_circuit: Circuit = framework(num_qubits)
        gate_func(checker_circuit)

        assert decomposed_circuit == checker_circuit

    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    def test_decompose_gates(
            self,
            framework: type[Circuit]
        ) -> None:
        """ Test specific gate decomposition using `decompose_gates` method.

        Parameters
        ----------
        `framework` : type[quick.circuit.Circuit]
        """
        circuit: Circuit = framework(3)
        circuit.UCRZ([0.1, 0.2, 0.3, 0.4], [0, 1], 2)
        circuit.H(0)
        circuit.Z(0)

        decomposed_circuit = circuit.decompose_gate(["UCRZ", "RZ", "Z"])

        checker_circuit: Circuit = framework(3)

        checker_circuit.RX(angle=1.5707963267948966, qubit_indices=2)
        checker_circuit.RY(angle=-0.25, qubit_indices=2)
        checker_circuit.RX(angle=-1.5707963267948966, qubit_indices=2)
        checker_circuit.CX(control_index=0, target_index=2)
        checker_circuit.RX(angle=1.5707963267948966, qubit_indices=2)
        checker_circuit.RY(angle=0.05000000000000002, qubit_indices=2)
        checker_circuit.RX(angle=-1.5707963267948966, qubit_indices=2)
        checker_circuit.CX(control_index=1, target_index=2)
        checker_circuit.RX(angle=1.5707963267948966, qubit_indices=2)
        checker_circuit.RX(angle=-1.5707963267948966, qubit_indices=2)
        checker_circuit.CX(control_index=0, target_index=2)
        checker_circuit.RX(angle=1.5707963267948966, qubit_indices=2)
        checker_circuit.RY(angle=0.1, qubit_indices=2)
        checker_circuit.RX(angle=-1.5707963267948966, qubit_indices=2)
        checker_circuit.CX(control_index=1, target_index=2)
        checker_circuit.H(qubit_indices=0)
        checker_circuit.Phase(angle=3.141592653589793, qubit_indices=0)

        assert decomposed_circuit == checker_circuit

    @pytest.mark.parametrize("framework", CIRCUIT_FRAMEWORKS)
    @pytest.mark.parametrize("gate_func, num_qubits", GATE_TYPES)
    def test_decompose_gates_primitive_gates(
            self,
            framework: type[Circuit],
            gate_func,
            num_qubits: int
        ) -> None:
        """ Test the decomposition of the primitive gates with `decompose_gates` method.

        Parameters
        ----------
        `framework` : type[quick.circuit.Circuit]
            The circuit framework to use.
        `gate_func` : Callable[[Circuit], None]
            The gate function to apply.
        `num_qubits` : int
            The number of qubits in the circuit.
        """
        # Define the circuit and apply the gate
        circuit: Circuit = framework(num_qubits)
        gate_func(circuit)

        assert circuit.circuit_log[-1]["definition"] == []

        # Decompose the circuit
        decomposed_circuit = circuit.decompose_gate(circuit.circuit_log[-1]["gate"])

        # Check the decomposed circuit
        # Assert it's unchanged
        checker_circuit: Circuit = framework(num_qubits)
        gate_func(checker_circuit)

        assert decomposed_circuit == checker_circuit