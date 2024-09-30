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


# The quantum circuit frameworks
CIRCUIT_FRAMEWORKS = [CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit]


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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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

    def phase_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Phase gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=2)

        # Apply Phase gate with both single index and multiple indices variations
        circuit.Phase(0.5, 0)
        circuit.Phase(0.5, [0, 1])

        # Define controlled-Phase gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=3)
        check_multiple_controlled_circuit = framework(num_qubits=4)

        check_single_controlled_circuit.MCPhase(0.5, 0, 1)
        check_single_controlled_circuit.MCPhase(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def u3_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with U3 gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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

    def cphase_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CPhase gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=3)

        # Apply CPhase gate with both single index and multiple indices variations
        circuit.CPhase(0.5, 0, 1)

        # Define controlled-CPhase gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=4)
        check_multiple_controlled_circuit = framework(num_qubits=5)

        check_single_controlled_circuit.MCPhase(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cswap_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CSWAP gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=3)

        # Apply CSWAP gate with both single index and multiple indices variations
        circuit.CSWAP(0, 1, 2)

        # Define controlled-CSWAP gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=4)
        check_multiple_controlled_circuit = framework(num_qubits=5)

        check_single_controlled_circuit.MCSWAP([0, 1], 2, 3)

        check_multiple_controlled_circuit.MCSWAP([0, 1, 2], 3, 4)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def cu3_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CU3 gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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
        `framework` : type[qickit.circuit.Circuit]
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

    def mcphase_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCPhase gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=5)

        # Apply MCPhase gate with both single index and multiple indices variations
        circuit.MCPhase(0.5, [0, 1], [2, 3])

        # Define controlled-MCPhase gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=6)
        check_multiple_controlled_circuit = framework(num_qubits=7)

        check_single_controlled_circuit.MCPhase(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def mcu3_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCU3 gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
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

    def mcswap_control(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCSWAP gate.

        Parameters
        ----------
        `framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=4)

        # Apply MCSWAP gate with both single index and multiple indices variations
        circuit.MCSWAP([0, 1], 2, 3)

        # Define controlled-MCSWAP gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = framework(num_qubits=5)
        check_multiple_controlled_circuit = framework(num_qubits=6)

        check_single_controlled_circuit.MCSWAP([0, 1, 2], 3, 4)

        check_multiple_controlled_circuit.MCSWAP([0, 1, 2, 3], 4, 5)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    def test_x_control(self) -> None:
        """ Test the `.control()` method with X gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.x_control(circuit_framework)

    def test_y_control(self) -> None:
        """ Test the `.control()` method with Y gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.y_control(circuit_framework)

    def test_z_control(self) -> None:
        """ Test the `.control()` method with Z gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.z_control(circuit_framework)

    def test_h_control(self) -> None:
        """ Test the `.control()` method with H gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.h_control(circuit_framework)

    def test_s_control(self) -> None:
        """ Test the `.control()` method with S gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.s_control(circuit_framework)

    def test_sdg_control(self) -> None:
        """ Test the `.control()` method with Sdg gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.sdg_control(circuit_framework)

    def test_t_control(self) -> None:
        """ Test the `.control()` method with T gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.t_control(circuit_framework)

    def test_tdg_control(self) -> None:
        """ Test the `.control()` method with Tdg gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.tdg_control(circuit_framework)

    def test_rx_control(self) -> None:
        """ Test the `.control()` method with RX gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.rx_control(circuit_framework)

    def test_ry_control(self) -> None:
        """ Test the `.control()` method with RY gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.ry_control(circuit_framework)

    def test_rz_control(self) -> None:
        """ Test the `.control()` method with RZ gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.rz_control(circuit_framework)

    def test_phase_control(self) -> None:
        """ Test the `.control()` method with Phase gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.phase_control(circuit_framework)

    def test_u3_control(self) -> None:
        """ Test the `.control()` method with U3 gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.u3_control(circuit_framework)

    def test_swap_control(self) -> None:
        """ Test the `.control()` method with SWAP gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.swap_control(circuit_framework)

    def test_cx_control(self) -> None:
        """ Test the `.control()` method with CX gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cx_control(circuit_framework)

    def test_cy_control(self) -> None:
        """ Test the `.control()` method with CY gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cy_control(circuit_framework)

    def test_cz_control(self) -> None:
        """ Test the `.control()` method with CZ gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cz_control(circuit_framework)

    def test_ch_control(self) -> None:
        """ Test the `.control()` method with CH gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.ch_control(circuit_framework)

    def test_cs_control(self) -> None:
        """ Test the `.control()` method with CS gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cs_control(circuit_framework)

    def test_csdg_control(self) -> None:
        """ Test the `.control()` method with CSdg gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.csdg_control(circuit_framework)

    def test_ct_control(self) -> None:
        """ Test the `.control()` method with CT gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.ct_control(circuit_framework)

    def test_ctdg_control(self) -> None:
        """ Test the `.control()` method with CTdg gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.ctdg_control(circuit_framework)

    def test_crx_control(self) -> None:
        """ Test the `.control()` method with CRX gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.crx_control(circuit_framework)

    def test_cry_control(self) -> None:
        """ Test the `.control()` method with CRY gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cry_control(circuit_framework)

    def test_crz_control(self) -> None:
        """ Test the `.control()` method with CRZ gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.crz_control(circuit_framework)

    def test_cphase_control(self) -> None:
        """ Test the `.control()` method with CPhase gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cphase_control(circuit_framework)

    def test_cu3_control(self) -> None:
        """ Test the `.control()` method with CU3 gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cu3_control(circuit_framework)

    def test_cswap_control(self) -> None:
        """ Test the `.control()` method with CSWAP gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.cswap_control(circuit_framework)

    def test_mcx_control(self) -> None:
        """ Test the `.control()` method with MCX gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcx_control(circuit_framework)

    def test_mcy_control(self) -> None:
        """ Test the `.control()` method with MCY gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcy_control(circuit_framework)

    def test_mcz_control(self) -> None:
        """ Test the `.control()` method with MCZ gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcz_control(circuit_framework)

    def test_mch_control(self) -> None:
        """ Test the `.control()` method with MCH gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mch_control(circuit_framework)

    def test_mcs_control(self) -> None:
        """ Test the `.control()` method with MCS gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcs_control(circuit_framework)

    def test_mcsdg_control(self) -> None:
        """ Test the `.control()` method with MCSdg gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcsdg_control(circuit_framework)

    def test_mct_control(self) -> None:
        """ Test the `.control()` method with MCT gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mct_control(circuit_framework)

    def test_mctdg_control(self) -> None:
        """ Test the `.control()` method with MCTdg gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mctdg_control(circuit_framework)

    def test_mcrx_control(self) -> None:
        """ Test the `.control()` method with MCRX gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcrx_control(circuit_framework)

    def test_mcry_control(self) -> None:
        """ Test the `.control()` method with MCRY gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcry_control(circuit_framework)

    def test_mcrz_control(self) -> None:
        """ Test the `.control()` method with MCRZ gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcrz_control(circuit_framework)

    def test_mcphase_control(self) -> None:
        """ Test the `.control()` method with MCPhase gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcphase_control(circuit_framework)

    def test_mcu3_control(self) -> None:
        """ Test the `.control()` method with MCU3 gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcu3_control(circuit_framework)

    def test_mcswap_control(self) -> None:
        """ Test the `.control()` method with MCSWAP gate.
        """
        for circuit_framework in CIRCUIT_FRAMEWORKS:
            self.mcswap_control(circuit_framework)