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

__all__ = ["TestTwoQubitDecomposition"]

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_almost_equal
import pytest
from scipy.stats import unitary_group

from quick.circuit import QiskitCircuit
from quick.synthesis.gate_decompositions.two_qubit_decomposition.weyl import TwoQubitWeylDecomposition
from quick.synthesis.gate_decompositions import TwoQubitDecomposition


class TestTwoQubitDecomposition:
    """ `tests.synthesis.gate_decompositions.TestTwoQubitDecomposition` is the tester
    class for `quick.synthesis.gate_decompositions.TwoQubitDecomposition` class.
    """
    def test_two_qubit_decomposition(self) -> None:
        """ Test the two qubit decomposition.
        """
        for _ in range(10):
            # Generate a random unitary matrix
            unitary_matrix = unitary_group.rvs(4).astype(complex)

            # Define a circuit
            circuit = QiskitCircuit(2)

            # Create a two qubit decomposition object
            two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

            # Apply the two qubit decomposition
            two_qubit_decomposition.apply_unitary(circuit, unitary_matrix, [0, 1])

            # Check that the circuit is equivalent to the original unitary matrix
            assert_almost_equal(circuit.get_unitary(), unitary_matrix, decimal=8)

    def test_two_qubit_decomposition_up_to_diagonal(self) -> None:
        """ Test the two qubit decomposition up to diagonal.
        """
        for _ in range(10):
            # Generate a random unitary matrix
            unitary_matrix = unitary_group.rvs(4).astype(complex)

            # Define a circuit
            circuit = QiskitCircuit(2)

            # Create a two qubit decomposition object
            two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

            # Apply the two qubit decomposition up to diagonal
            circuit, diagonal = two_qubit_decomposition.apply_unitary_up_to_diagonal(
                two_qubit_decomposition.output_framework(2),
                unitary_matrix, [0, 1]
            )

            circuit.add(two_qubit_decomposition.prepare_unitary(diagonal), [0, 1])

            # Check that the circuit is equivalent to the original unitary matrix
            assert_almost_equal(circuit.get_unitary(), unitary_matrix, decimal=8)

    def test_decomp0(self) -> None:
        """ Test `TwoQubitDecomposition._decomp0()`.
        """
        # Initialize a circuit
        circuit = QiskitCircuit(2)

        # Apply a one qubit gate (no CX gates)
        circuit.H(0)

        # Extract the unitary matrix
        unitary_matrix = circuit.get_unitary()

        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # First part of the test is to ensure the best number of basis is 0
        target_decomposed = TwoQubitWeylDecomposition(unitary_matrix)
        traces = two_qubit_decomposition.traces(target_decomposed)
        expected_fidelities = [TwoQubitDecomposition.trace_to_fidelity(traces[i]) for i in range(4)]
        best_num_basis = int(np.argmax(expected_fidelities))

        assert best_num_basis == 0

        # The second part of the test is to ensure the unitary encoded is correct
        # Initialize a new circuit
        new_circuit = QiskitCircuit(2)

        # Apply the decomposition
        two_qubit_decomposition.apply_unitary(new_circuit, unitary_matrix, [0, 1])

        # Check that the circuit is equivalent to the original unitary matrix
        assert_almost_equal(new_circuit.get_unitary(), unitary_matrix, decimal=8)

        # The third part of the test is to ensure the circuit created has no CX gates
        # Get the number of CX gates
        num_cx_gates = new_circuit.count_ops().get("CX", 0)

        # Check that the number of CX gates is 0
        assert num_cx_gates == 0

    def test_decomp1(self) -> None:
        """ Test `TwoQubitDecomposition._decomp1()`.
        """
        # Initialize a circuit
        circuit = QiskitCircuit(2)

        # Create a GHZ state (one CX gate)
        circuit.H(0)
        circuit.CX(0, 1)

        # Extract the unitary matrix
        unitary_matrix = circuit.get_unitary()

        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # First part of the test is to ensure the best number of basis is 1
        target_decomposed = TwoQubitWeylDecomposition(unitary_matrix)
        traces = two_qubit_decomposition.traces(target_decomposed)
        expected_fidelities = [TwoQubitDecomposition.trace_to_fidelity(traces[i]) for i in range(4)]
        best_num_basis = int(np.argmax(expected_fidelities))

        assert best_num_basis == 1

        # The second part of the test is to ensure the unitary encoded is correct
        # Initialize a new circuit
        new_circuit = QiskitCircuit(2)

        # Apply the decomposition
        two_qubit_decomposition.apply_unitary(new_circuit, unitary_matrix, [0, 1])

        # Check that the circuit is equivalent to the original unitary matrix
        assert_almost_equal(new_circuit.get_unitary(), unitary_matrix, decimal=8)

        # The third part of the test is to ensure the circuit created has one or no CX gates
        # Get the number of CX gates
        num_cx_gates = new_circuit.count_ops().get("CX", 0)

        # Check that the number of CX gates is 1 or 0
        assert num_cx_gates <= 1

    def test_decomp2_supercontrolled(self) -> None:
        """ Test `TwoQubitDecomposition._decomp2_supercontrolled()`.
        """
        # Initialize a circuit
        circuit = QiskitCircuit(2)

        # Define a two qubit gate (two CX gates)
        circuit.H(0)
        circuit.CX(0, 1)
        circuit.RX(1/3 * np.pi, 0)
        circuit.CX(0, 1)

        # Extract the unitary matrix
        unitary_matrix = circuit.get_unitary()

        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # First part of the test is to ensure the best number of basis is 2
        target_decomposed = TwoQubitWeylDecomposition(unitary_matrix)
        traces = two_qubit_decomposition.traces(target_decomposed)
        expected_fidelities = [TwoQubitDecomposition.trace_to_fidelity(traces[i]) for i in range(4)]
        best_num_basis = int(np.argmax(expected_fidelities))

        assert best_num_basis == 2

        # The second part of the test is to ensure the unitary encoded is correct
        # Initialize a new circuit
        new_circuit = QiskitCircuit(2)

        # Apply the decomposition
        two_qubit_decomposition.apply_unitary(new_circuit, unitary_matrix, [0, 1])

        # Check that the circuit is equivalent to the original unitary matrix
        assert_almost_equal(new_circuit.get_unitary(), unitary_matrix, decimal=8)

        # The third part of the test is to ensure the circuit created has two or less CX gates
        # Get the number of CX gates
        num_cx_gates = new_circuit.count_ops().get("CX", 0)

        # Check that the number of CX gates is 2 or less
        assert num_cx_gates <= 2

    def test_decomp3_supercontrolled(self) -> None:
        """ Test `TwoQubitDecomposition._decomp3_supercontrolled()`.
        """
        # Initialize a circuit
        circuit = QiskitCircuit(2)

        # Define a two qubit gate (three CX gates)
        circuit.circuit_log = [
            {'gate': 'U3', 'angles': [1.4795141072922737, 4.022999026161385, -1.4643238780690337], 'qubit_indices': 0},
            {'gate': 'U3', 'angles': [0.99368034324659, -0.4046788743527383, 5.325053604136903], 'qubit_indices': 1},
            {'gate': 'CX', 'control_index': 0, 'target_index': 1},
            {'gate': 'U3', 'angles': [0.8960711613534293, 0.0, -3.141592653589793], 'qubit_indices': 0},
            {'gate': 'U3', 'angles': [0.26210799778546756, 1.5707963267948966, 3.141592653589793], 'qubit_indices': 1},
            {'gate': 'CX', 'control_index': 0, 'target_index': 1},
            {'gate': 'U3', 'angles': [0.147522578941486, -4.71238898038469, 0.0], 'qubit_indices': 0},
            {'gate': 'U3', 'angles': [1.5707963267948966, -1.5707963267948966, -3.141592653589793], 'qubit_indices': 1},
            {'gate': 'CX', 'control_index': 0, 'target_index': 1},
            {'gate': 'U3', 'angles': [1.4795141073495195, -1.4643238779322805, 4.022999026024632], 'qubit_indices': 0},
            {'gate': 'U3', 'angles': [1.2101762902010682, -2.7812036037162535, -0.7484456350052462], 'qubit_indices': 1},
            {'gate': 'GlobalPhase', 'angle': 6.1707400671456965}
        ]

        circuit.update()

        # Extract the unitary matrix
        unitary_matrix = circuit.get_unitary()

        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # First part of the test is to ensure the best number of basis is 3
        target_decomposed = TwoQubitWeylDecomposition(unitary_matrix)
        traces = two_qubit_decomposition.traces(target_decomposed)
        expected_fidelities = [TwoQubitDecomposition.trace_to_fidelity(traces[i]) for i in range(4)]
        best_num_basis = int(np.argmax(expected_fidelities))

        assert best_num_basis == 3

        # The second part of the test is to ensure the unitary encoded is correct
        # Initialize a new circuit
        new_circuit = QiskitCircuit(2)

        # Apply the decomposition
        two_qubit_decomposition.apply_unitary(new_circuit, unitary_matrix, [0, 1])

        # Check that the circuit is equivalent to the original unitary matrix
        assert_almost_equal(new_circuit.get_unitary(), unitary_matrix, decimal=8)

        # The third part of the test is to ensure the circuit created has three or less CX gates
        # Get the number of CX gates
        num_cx_gates = new_circuit.count_ops().get("CX", 0)

        # Check that the number of CX gates is 3 or less
        assert num_cx_gates <= 3

    @pytest.mark.parametrize("unitary", [
        np.array([
            [2./3, 1./3 + 1.j/3, 7 * np.sqrt(17)/51, (-1 - 1.j) * np.sqrt(17)/51],
            [1./3 - 1.j/3, -1./3, (-1 + 1.j) * np.sqrt(17)/51, 10 * np.sqrt(17)/51],
            [7*np.sqrt(17)/51, (-1 - 1.j) * np.sqrt(17)/51, -2./3, -1./3 - 1.j/3],
            [(-1 + 1.j) * np.sqrt(17)/51, 10 * np.sqrt(17)/51, -1./3 + 1.j/3, 1./3]
        ]),
        np.array([
            [
                0.043602684126304955 + -0.4986216208233326j, -0.22461079447224863 + -0.09679517443671315j,
                0.15592626697527698 + 0.13943283228059802j, -0.803224008021238 + 0.027067469117215155j
            ],
            [
                -0.09679517443669944 + 0.22461079447226426j, 0.4986216208232763 + 0.043602684126298856j,
                0.02706746911718458 + 0.8032240080213412j, -0.13943283228056091 + 0.15592626697531453j
            ],
            [
                0.13943283228059844 + -0.15592626697527806j, 0.027067469117214762 + 0.8032240080212403j,
                0.49862162082333095 + 0.0436026841263046j, 0.09679517443671384 + -0.22461079447224833j
            ],
            [
                0.8032240080213434 + -0.027067469117184207j, 0.1559262669753156 + 0.13943283228056125j,
                -0.2246107944722639 + -0.09679517443670013j, -0.043602684126298495 + 0.4986216208232746j
            ]
        ]),
        np.array([
            [-0.62695238, -0.1993407 , -0.63291226, -0.40818632],
            [-0.62695237, -0.42741604,  0.61214869,  0.2225314 ],
            [-0.32700263,  0.43412304, -0.31468734,  0.77818914],
            [ 0.32700264, -0.76753892, -0.35449671,  0.42223851]
        ]),
        np.array([
            [
                -0.3812064266367201 + 0.38120642663672005j, -0.08953682318096808 + 0.08953682318096806j,
                -0.5214184531408846 + 0.5214184531408846j, -0.27347323579354943 + 0.2734732357935494j,
            ],
            [
                0.11105398348218393 - 0.11105398348218391j, 0.145430219977459 - 0.14543021997745897j,
                0.23096365206101888 - 0.23096365206101882j, -0.6427852354467597 + 0.6427852354467596j,
            ],
            [
                -0.544932087007239 + 0.5449320870072389j, -0.16786656959437854 + 0.16786656959437854j,
                0.41778724529336947 - 0.4177872452933694j, 0.01799033088883041 - 0.017990330888830407j,
            ],
            [
                -0.213067345160641 + 0.21306734516064096j, 0.6653224962721628 - 0.6653224962721627j,
                -0.0152448403255109 + 0.015244840325510897j, 0.10823991063233353 - 0.10823991063233351j,
            ],
        ])
    ])
    def test_m2_correctness(
            self,
            unitary: NDArray[np.complex128]
        ) -> None:
        """ Test the correctness of the M2 decomposition. This tests
        many recorded cases of failure of two qubit decomposition on
        non-Ubuntu OS.

        Parameters
        ----------
        `unitary` : NDArray[np.complex128]
            The two qubit unitary to encode.
        """
        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # Initialize a circuit
        circuit = QiskitCircuit(2)

        # Apply the decomposition
        two_qubit_decomposition.apply_unitary(circuit, unitary, [0, 1])

        # Check that the circuit is equivalent to the original unitary matrix
        assert_almost_equal(circuit.get_unitary(), unitary, decimal=8)

    def test_invalid_indices_fail(self) -> None:
        """ Test that invalid indices fail.
        """
        # Generate a random unitary matrix
        unitary_matrix = unitary_group.rvs(4).astype(complex)

        # Define a circuit
        circuit = QiskitCircuit(2)

        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # Apply the two qubit decomposition
        with pytest.raises(ValueError):
            two_qubit_decomposition.apply_unitary(circuit, unitary_matrix, [0, 1, 2])

    def test_invalid_unitary_fail(self) -> None:
        """ Test that invalid unitaries fail.
        """
        # Generate a random unitary matrix
        unitary_matrix = unitary_group.rvs(8).astype(complex)

        # Define a circuit
        circuit = QiskitCircuit(2)

        # Create a two qubit decomposition object
        two_qubit_decomposition = TwoQubitDecomposition(output_framework=QiskitCircuit)

        # Apply the two qubit decomposition
        with pytest.raises(ValueError):
            two_qubit_decomposition.apply_unitary(circuit, unitary_matrix, [0, 1])