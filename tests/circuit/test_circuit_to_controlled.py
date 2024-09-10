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

__all__ = ["TestControlled"]

from typing import Type

from qickit.circuit import Circuit, CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit


class TestControlled:
    """ `tests.circuit.TestControlled` is the tester for the `.control()` method.
    """
    def x_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with X gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply X gate with both single index and multiple indices variations
        circuit.X(0)
        circuit.X([0, 1])

        # Define controlled-X gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCX(0, 1)
        check_single_controlled_circuit.MCX(0, [1, 2])

        check_multiple_controlled_circuit.MCX([0, 1], 2)
        check_multiple_controlled_circuit.MCX([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def y_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Y gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply Y gate with both single index and multiple indices variations
        circuit.Y(0)
        circuit.Y([0, 1])

        # Define controlled-Y gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCY(0, 1)
        check_single_controlled_circuit.MCY(0, [1, 2])

        check_multiple_controlled_circuit.MCY([0, 1], 2)
        check_multiple_controlled_circuit.MCY([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def z_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Z gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply Z gate with both single index and multiple indices variations
        circuit.Z(0)
        circuit.Z([0, 1])

        # Define controlled-Z gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCZ(0, 1)
        check_single_controlled_circuit.MCZ(0, [1, 2])

        check_multiple_controlled_circuit.MCZ([0, 1], 2)
        check_multiple_controlled_circuit.MCZ([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def h_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with H gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply H gate with both single index and multiple indices variations
        circuit.H(0)
        circuit.H([0, 1])

        # Define controlled-H gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCH(0, 1)
        check_single_controlled_circuit.MCH(0, [1, 2])

        check_multiple_controlled_circuit.MCH([0, 1], 2)
        check_multiple_controlled_circuit.MCH([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def s_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with S gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply S gate with both single index and multiple indices variations
        circuit.S(0)
        circuit.S([0, 1])

        # Define controlled-S gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCS(0, 1)
        check_single_controlled_circuit.MCS(0, [1, 2])

        check_multiple_controlled_circuit.MCS([0, 1], 2)
        check_multiple_controlled_circuit.MCS([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def sdg_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Sdg gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply Sdg gate with both single index and multiple indices variations
        circuit.Sdg(0)
        circuit.Sdg([0, 1])

        # Define controlled-Sdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCSdg(0, 1)
        check_single_controlled_circuit.MCSdg(0, [1, 2])

        check_multiple_controlled_circuit.MCSdg([0, 1], 2)
        check_multiple_controlled_circuit.MCSdg([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def t_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with T gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply T gate with both single index and multiple indices variations
        circuit.T(0)
        circuit.T([0, 1])

        # Define controlled-T gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCT(0, 1)
        check_single_controlled_circuit.MCT(0, [1, 2])

        check_multiple_controlled_circuit.MCT([0, 1], 2)
        check_multiple_controlled_circuit.MCT([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def tdg_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Tdg gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply Tdg gate with both single index and multiple indices variations
        circuit.Tdg(0)
        circuit.Tdg([0, 1])

        # Define controlled-Tdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCTdg(0, 1)
        check_single_controlled_circuit.MCTdg(0, [1, 2])

        check_multiple_controlled_circuit.MCTdg([0, 1], 2)
        check_multiple_controlled_circuit.MCTdg([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def rx_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with RX gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply RX gate with both single index and multiple indices variations
        circuit.RX(0.5, 0)
        circuit.RX(0.5, [0, 1])

        # Define controlled-Rx gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCRX(0.5, 0, 1)
        check_single_controlled_circuit.MCRX(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCRX(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCRX(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def ry_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with RY gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply RY gate with both single index and multiple indices variations
        circuit.RY(0.5, 0)
        circuit.RY(0.5, [0, 1])

        # Define controlled-Ry gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCRY(0.5, 0, 1)
        check_single_controlled_circuit.MCRY(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCRY(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCRY(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def rz_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with RZ gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply RZ gate with both single index and multiple indices variations
        circuit.RZ(0.5, 0)
        circuit.RZ(0.5, [0, 1])

        # Define controlled-Rz gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCRZ(0.5, 0, 1)
        check_single_controlled_circuit.MCRZ(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def u3_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with U3 gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply U3 gate with both single index and multiple indices variations
        circuit.U3([0.1, 0.2, 0.3], 0)

        # Define controlled-U3 gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCU3([0.1, 0.2, 0.3], 0, 1)

        check_multiple_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], 2)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def swap_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with SWAP gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply SWAP gate with both single index and multiple indices variations
        circuit.SWAP(0, 1)

        # Define controlled-SWAP gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCSWAP([0], 1, 2)

        check_multiple_controlled_circuit.MCSWAP([0, 1], 2, 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cx_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CX gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CX gate with both single index and multiple indices variations
        circuit.CX(0, 1)

        # Define controlled-CX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCX([0, 1], 2)

        check_multiple_controlled_circuit.MCX([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cy_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CY gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CY gate with both single index and multiple indices variations
        circuit.CY(0, 1)

        # Define controlled-CY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCY([0, 1], 2)

        check_multiple_controlled_circuit.MCY([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cz_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CZ gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CZ gate with both single index and multiple indices variations
        circuit.CZ(0, 1)

        # Define controlled-CZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCZ([0, 1], 2)

        check_multiple_controlled_circuit.MCZ([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def ch_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CH gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CH gate with both single index and multiple indices variations
        circuit.CH(0, 1)

        # Define controlled-CH gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCH([0, 1], 2)

        check_multiple_controlled_circuit.MCH([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cs_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CS gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CS gate with both single index and multiple indices variations
        circuit.CS(0, 1)

        # Define controlled-CS gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCS([0, 1], 2)

        check_multiple_controlled_circuit.MCS([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def csdg_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CSdg gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CSdg gate with both single index and multiple indices variations
        circuit.CSdg(0, 1)

        # Define controlled-CSdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCSdg([0, 1], 2)

        check_multiple_controlled_circuit.MCSdg([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def ct_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CT gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CT gate with both single index and multiple indices variations
        circuit.CT(0, 1)

        # Define controlled-CT gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCT([0, 1], 2)

        check_multiple_controlled_circuit.MCT([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def ctdg_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CTdg gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply CTdg gate with both single index and multiple indices variations
        circuit.CTdg(0, 1)

        # Define controlled-CTdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCTdg([0, 1], 2)

        check_multiple_controlled_circuit.MCTdg([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def crx_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CRX gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=3)

        # Apply CRX gate with both single index and multiple indices variations
        circuit.CRX(0.5, 0, 1)

        # Define controlled-CRX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=4)
        check_multiple_controlled_circuit = framework(num_qubits=5)

        check_single_controlled_circuit.MCRX(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCRX(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cry_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CRY gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=3)

        # Apply CRY gate with both single index and multiple indices variations
        circuit.CRY(0.5, 0, 1)

        # Define controlled-CRY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=4)
        check_multiple_controlled_circuit = framework(num_qubits=5)

        check_single_controlled_circuit.MCRY(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCRY(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def crz_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CRZ gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=3)

        # Apply CRZ gate with both single index and multiple indices variations
        circuit.CRZ(0.5, 0, 1)

        # Define controlled-CRZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=4)
        check_multiple_controlled_circuit = framework(num_qubits=5)

        check_single_controlled_circuit.MCRZ(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cu3_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CU3 gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=3)

        # Apply CU3 gate with both single index and multiple indices variations
        circuit.CU3([0.1, 0.2, 0.3], 0, 1)

        # Define controlled-CU3 gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=4)
        check_multiple_controlled_circuit = framework(num_qubits=5)

        check_single_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], 2)

        check_multiple_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcx_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCX gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCX gate with both single index and multiple indices variations
        circuit.MCX([0, 1], [2, 3])

        # Define controlled-MCX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCX([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCX([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcy_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCY gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCY gate with both single index and multiple indices variations
        circuit.MCY([0, 1], [2, 3])

        # Define controlled-MCY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCY([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCY([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcz_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCZ gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCZ gate with both single index and multiple indices variations
        circuit.MCZ([0, 1], [2, 3])

        # Define controlled-MCZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCZ([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCZ([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mch_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCH gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCH gate with both single index and multiple indices variations
        circuit.MCH([0, 1], [2, 3])

        # Define controlled-MCH gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCH([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCH([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcs_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCS gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCS gate with both single index and multiple indices variations
        circuit.MCS([0, 1], [2, 3])

        # Define controlled-MCS gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCS([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCS([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcsdg_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCSdg gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCSdg gate with both single index and multiple indices variations
        circuit.MCSdg([0, 1], [2, 3])

        # Define controlled-MCSdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCSdg([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCSdg([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mct_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCT gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCT gate with both single index and multiple indices variations
        circuit.MCT([0, 1], [2, 3])

        # Define controlled-MCT gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCT([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCT([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mctdg_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCTdg gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCTdg gate with both single index and multiple indices variations
        circuit.MCTdg([0, 1], [2, 3])

        # Define controlled-MCTdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCTdg([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCTdg([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcrx_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCRX gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=5)

        # Apply MCRX gate with both single index and multiple indices variations
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # Define controlled-MCRX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=6)
        check_multiple_controlled_circuit = framework(num_qubits=7)

        check_single_controlled_circuit.MCRX(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCRX(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcry_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCRY gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=5)

        # Apply MCRY gate with both single index and multiple indices variations
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # Define controlled-MCRY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=6)
        check_multiple_controlled_circuit = framework(num_qubits=7)

        check_single_controlled_circuit.MCRY(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCRY(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcrz_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCRZ gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=5)

        # Apply MCRZ gate with both single index and multiple indices variations
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Define controlled-MCRZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=6)
        check_multiple_controlled_circuit = framework(num_qubits=7)

        check_single_controlled_circuit.MCRZ(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcu3_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCU3 gate.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=5)

        # Apply MCU3 gate with both single index and multiple indices variations
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        # Define controlled-MCU3 gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=6)
        check_multiple_controlled_circuit = framework(num_qubits=7)

        check_single_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def test_x_control(self) -> None:
        """ Test the `.control()` method with X gate. """
        self.x_control(CirqCircuit)
        self.x_control(PennylaneCircuit)
        self.x_control(QiskitCircuit)
        self.x_control(TKETCircuit)

    def test_y_control(self) -> None:
        """ Test the `.control()` method with Y gate. """
        self.y_control(CirqCircuit)
        self.y_control(PennylaneCircuit)
        self.y_control(QiskitCircuit)
        self.y_control(TKETCircuit)

    def test_z_control(self) -> None:
        """ Test the `.control()` method with Z gate. """
        self.z_control(CirqCircuit)
        self.z_control(PennylaneCircuit)
        self.z_control(QiskitCircuit)
        self.z_control(TKETCircuit)

    def test_h_control(self) -> None:
        """ Test the `.control()` method with H gate. """
        self.h_control(CirqCircuit)
        self.h_control(PennylaneCircuit)
        self.h_control(QiskitCircuit)
        self.h_control(TKETCircuit)

    def test_s_control(self) -> None:
        """ Test the `.control()` method with S gate. """
        self.s_control(CirqCircuit)
        self.s_control(PennylaneCircuit)
        self.s_control(QiskitCircuit)
        self.s_control(TKETCircuit)

    def test_sdg_control(self) -> None:
        """ Test the `.control()` method with Sdg gate. """
        self.sdg_control(CirqCircuit)
        self.sdg_control(PennylaneCircuit)
        self.sdg_control(QiskitCircuit)
        self.sdg_control(TKETCircuit)

    def test_t_control(self) -> None:
        """ Test the `.control()` method with T gate. """
        self.t_control(CirqCircuit)
        self.t_control(PennylaneCircuit)
        self.t_control(QiskitCircuit)
        self.t_control(TKETCircuit)

    def test_tdg_control(self) -> None:
        """ Test the `.control()` method with Tdg gate. """
        self.tdg_control(CirqCircuit)
        self.tdg_control(PennylaneCircuit)
        self.tdg_control(QiskitCircuit)
        self.tdg_control(TKETCircuit)

    def test_rx_control(self) -> None:
        """ Test the `.control()` method with RX gate. """
        self.rx_control(CirqCircuit)
        self.rx_control(PennylaneCircuit)
        self.rx_control(QiskitCircuit)
        self.rx_control(TKETCircuit)

    def test_ry_control(self) -> None:
        """ Test the `.control()` method with RY gate. """
        self.ry_control(CirqCircuit)
        self.ry_control(PennylaneCircuit)
        self.ry_control(QiskitCircuit)
        self.ry_control(TKETCircuit)

    def test_rz_control(self) -> None:
        """ Test the `.control()` method with RZ gate. """
        self.rz_control(CirqCircuit)
        self.rz_control(PennylaneCircuit)
        self.rz_control(QiskitCircuit)
        self.rz_control(TKETCircuit)

    def test_u3_control(self) -> None:
        """ Test the `.control()` method with U3 gate. """
        self.u3_control(CirqCircuit)
        self.u3_control(PennylaneCircuit)
        self.u3_control(QiskitCircuit)
        self.u3_control(TKETCircuit)

    def test_swap_control(self) -> None:
        """ Test the `.control()` method with SWAP gate. """
        self.swap_control(CirqCircuit)
        self.swap_control(PennylaneCircuit)
        self.swap_control(QiskitCircuit)
        self.swap_control(TKETCircuit)

    def test_cx_control(self) -> None:
        """ Test the `.control()` method with CX gate. """
        self.cx_control(CirqCircuit)
        self.cx_control(PennylaneCircuit)
        self.cx_control(QiskitCircuit)
        self.cx_control(TKETCircuit)

    def test_cy_control(self) -> None:
        """ Test the `.control()` method with CY gate. """
        self.cy_control(CirqCircuit)
        self.cy_control(PennylaneCircuit)
        self.cy_control(QiskitCircuit)
        self.cy_control(TKETCircuit)

    def test_cz_control(self) -> None:
        """ Test the `.control()` method with CZ gate. """
        self.cz_control(CirqCircuit)
        self.cz_control(PennylaneCircuit)
        self.cz_control(QiskitCircuit)
        self.cz_control(TKETCircuit)

    def test_ch_control(self) -> None:
        """ Test the `.control()` method with CH gate. """
        self.ch_control(CirqCircuit)
        self.ch_control(PennylaneCircuit)
        self.ch_control(QiskitCircuit)
        self.ch_control(TKETCircuit)

    def test_cs_control(self) -> None:
        """ Test the `.control()` method with CS gate. """
        self.cs_control(CirqCircuit)
        self.cs_control(PennylaneCircuit)
        self.cs_control(QiskitCircuit)
        self.cs_control(TKETCircuit)

    def test_csdg_control(self) -> None:
        """ Test the `.control()` method with CSdg gate. """
        self.csdg_control(CirqCircuit)
        self.csdg_control(PennylaneCircuit)
        self.csdg_control(QiskitCircuit)
        self.csdg_control(TKETCircuit)

    def test_ct_control(self) -> None:
        """ Test the `.control()` method with CT gate. """
        self.ct_control(CirqCircuit)
        self.ct_control(PennylaneCircuit)
        self.ct_control(QiskitCircuit)
        self.ct_control(TKETCircuit)

    def test_ctdg_control(self) -> None:
        """ Test the `.control()` method with CTdg gate. """
        self.ctdg_control(CirqCircuit)
        self.ctdg_control(PennylaneCircuit)
        self.ctdg_control(QiskitCircuit)
        self.ctdg_control(TKETCircuit)

    def test_crx_control(self) -> None:
        """ Test the `.control()` method with CRX gate. """
        self.crx_control(CirqCircuit)
        self.crx_control(PennylaneCircuit)
        self.crx_control(QiskitCircuit)
        self.crx_control(TKETCircuit)

    def test_cry_control(self) -> None:
        """ Test the `.control()` method with CRY gate. """
        self.cry_control(CirqCircuit)
        self.cry_control(PennylaneCircuit)
        self.cry_control(QiskitCircuit)
        self.cry_control(TKETCircuit)

    def test_crz_control(self) -> None:
        """ Test the `.control()` method with CRZ gate. """
        self.crz_control(CirqCircuit)
        self.crz_control(PennylaneCircuit)
        self.crz_control(QiskitCircuit)
        self.crz_control(TKETCircuit)

    def test_cu3_control(self) -> None:
        """ Test the `.control()` method with CU3 gate. """
        self.cu3_control(CirqCircuit)
        self.cu3_control(PennylaneCircuit)
        self.cu3_control(QiskitCircuit)
        self.cu3_control(TKETCircuit)

    def test_mcx_control(self) -> None:
        """ Test the `.control()` method with MCX gate. """
        self.mcx_control(CirqCircuit)
        self.mcx_control(PennylaneCircuit)
        self.mcx_control(QiskitCircuit)
        self.mcx_control(TKETCircuit)

    def test_mcy_control(self) -> None:
        """ Test the `.control()` method with MCY gate. """
        self.mcy_control(CirqCircuit)
        self.mcy_control(PennylaneCircuit)
        self.mcy_control(QiskitCircuit)
        self.mcy_control(TKETCircuit)

    def test_mcz_control(self) -> None:
        """ Test the `.control()` method with MCZ gate. """
        self.mcz_control(CirqCircuit)
        self.mcz_control(PennylaneCircuit)
        self.mcz_control(QiskitCircuit)
        self.mcz_control(TKETCircuit)

    def test_mch_control(self) -> None:
        """ Test the `.control()` method with MCH gate. """
        self.mch_control(CirqCircuit)
        self.mch_control(PennylaneCircuit)
        self.mch_control(QiskitCircuit)
        self.mch_control(TKETCircuit)

    def test_mcs_control(self) -> None:
        """ Test the `.control()` method with MCS gate. """
        self.mcs_control(CirqCircuit)
        self.mcs_control(PennylaneCircuit)
        self.mcs_control(QiskitCircuit)
        self.mcs_control(TKETCircuit)

    def test_mcsdg_control(self) -> None:
        """ Test the `.control()` method with MCSdg gate. """
        self.mcsdg_control(CirqCircuit)
        self.mcsdg_control(PennylaneCircuit)
        self.mcsdg_control(QiskitCircuit)
        self.mcsdg_control(TKETCircuit)

    def test_mct_control(self) -> None:
        """ Test the `.control()` method with MCT gate. """
        self.mct_control(CirqCircuit)
        self.mct_control(PennylaneCircuit)
        self.mct_control(QiskitCircuit)
        self.mct_control(TKETCircuit)

    def test_mctdg_control(self) -> None:
        """ Test the `.control()` method with MCTdg gate. """
        self.mctdg_control(CirqCircuit)
        self.mctdg_control(PennylaneCircuit)
        self.mctdg_control(QiskitCircuit)
        self.mctdg_control(TKETCircuit)

    def test_mcrx_control(self) -> None:
        """ Test the `.control()` method with MCRX gate. """
        self.mcrx_control(CirqCircuit)
        self.mcrx_control(PennylaneCircuit)
        self.mcrx_control(QiskitCircuit)
        self.mcrx_control(TKETCircuit)

    def test_mcry_control(self) -> None:
        """ Test the `.control()` method with MCRY gate. """
        self.mcry_control(CirqCircuit)
        self.mcry_control(PennylaneCircuit)
        self.mcry_control(QiskitCircuit)
        self.mcry_control(TKETCircuit)

    def test_mcrz_control(self) -> None:
        """ Test the `.control()` method with MCRZ gate. """
        self.mcrz_control(CirqCircuit)
        self.mcrz_control(PennylaneCircuit)
        self.mcrz_control(QiskitCircuit)
        self.mcrz_control(TKETCircuit)

    def test_mcu3_control(self) -> None:
        """ Test the `.control()` method with MCU3 gate. """
        self.mcu3_control(CirqCircuit)
        self.mcu3_control(PennylaneCircuit)
        self.mcu3_control(QiskitCircuit)
        self.mcu3_control(TKETCircuit)