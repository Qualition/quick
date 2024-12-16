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

import pytest
from typing import Type

from qickit.circuit import Circuit

from tests.circuit import CIRCUIT_FRAMEWORKS


class TestControlled:
    """ `tests.circuit.TestControlled` is the tester for the `.control()` method.
    """
    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_x_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with X gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply X gate with both single index and multiple indices variations
        circuit.X(0)
        circuit.X([0, 1])

        # Define controlled-X gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCX(0, 1)
        check_single_controlled_circuit.MCX(0, [1, 2])

        check_multiple_controlled_circuit.MCX([0, 1], 2)
        check_multiple_controlled_circuit.MCX([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_y_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Y gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply Y gate with both single index and multiple indices variations
        circuit.Y(0)
        circuit.Y([0, 1])

        # Define controlled-Y gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCY(0, 1)
        check_single_controlled_circuit.MCY(0, [1, 2])

        check_multiple_controlled_circuit.MCY([0, 1], 2)
        check_multiple_controlled_circuit.MCY([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_z_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Z gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply Z gate with both single index and multiple indices variations
        circuit.Z(0)
        circuit.Z([0, 1])

        # Define controlled-Z gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCZ(0, 1)
        check_single_controlled_circuit.MCZ(0, [1, 2])

        check_multiple_controlled_circuit.MCZ([0, 1], 2)
        check_multiple_controlled_circuit.MCZ([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_h_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with H gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply H gate with both single index and multiple indices variations
        circuit.H(0)
        circuit.H([0, 1])

        # Define controlled-H gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCH(0, 1)
        check_single_controlled_circuit.MCH(0, [1, 2])

        check_multiple_controlled_circuit.MCH([0, 1], 2)
        check_multiple_controlled_circuit.MCH([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_s_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with S gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply S gate with both single index and multiple indices variations
        circuit.S(0)
        circuit.S([0, 1])

        # Define controlled-S gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCS(0, 1)
        check_single_controlled_circuit.MCS(0, [1, 2])

        check_multiple_controlled_circuit.MCS([0, 1], 2)
        check_multiple_controlled_circuit.MCS([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_sdg_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Sdg gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply Sdg gate with both single index and multiple indices variations
        circuit.Sdg(0)
        circuit.Sdg([0, 1])

        # Define controlled-Sdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCSdg(0, 1)
        check_single_controlled_circuit.MCSdg(0, [1, 2])

        check_multiple_controlled_circuit.MCSdg([0, 1], 2)
        check_multiple_controlled_circuit.MCSdg([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_t_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with T gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply T gate with both single index and multiple indices variations
        circuit.T(0)
        circuit.T([0, 1])

        # Define controlled-T gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCT(0, 1)
        check_single_controlled_circuit.MCT(0, [1, 2])

        check_multiple_controlled_circuit.MCT([0, 1], 2)
        check_multiple_controlled_circuit.MCT([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_tdg_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Tdg gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply Tdg gate with both single index and multiple indices variations
        circuit.Tdg(0)
        circuit.Tdg([0, 1])

        # Define controlled-Tdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCTdg(0, 1)
        check_single_controlled_circuit.MCTdg(0, [1, 2])

        check_multiple_controlled_circuit.MCTdg([0, 1], 2)
        check_multiple_controlled_circuit.MCTdg([0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_rx_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with RX gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply RX gate with both single index and multiple indices variations
        circuit.RX(0.5, 0)
        circuit.RX(0.5, [0, 1])

        # Define controlled-Rx gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCRX(0.5, 0, 1)
        check_single_controlled_circuit.MCRX(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCRX(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCRX(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_ry_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with RY gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply RY gate with both single index and multiple indices variations
        circuit.RY(0.5, 0)
        circuit.RY(0.5, [0, 1])

        # Define controlled-Ry gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCRY(0.5, 0, 1)
        check_single_controlled_circuit.MCRY(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCRY(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCRY(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_rz_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with RZ gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply RZ gate with both single index and multiple indices variations
        circuit.RZ(0.5, 0)
        circuit.RZ(0.5, [0, 1])

        # Define controlled-Rz gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCRZ(0.5, 0, 1)
        check_single_controlled_circuit.MCRZ(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1], [2, 3])

        print(single_controlled_circuit.circuit_log)
        print(check_single_controlled_circuit.circuit_log)

        print(multiple_controlled_circuit.circuit_log)
        print(check_multiple_controlled_circuit.circuit_log)
        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_phase_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with Phase gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply Phase gate with both single index and multiple indices variations
        circuit.Phase(0.5, 0)
        circuit.Phase(0.5, [0, 1])

        # Define controlled-Phase gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCPhase(0.5, 0, 1)
        check_single_controlled_circuit.MCPhase(0.5, 0, [1, 2])

        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1], 2)
        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1], [2, 3])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_u3_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with U3 gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply U3 gate with both single index and multiple indices variations
        circuit.U3([0.1, 0.2, 0.3], 0)

        # Define controlled-U3 gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCU3([0.1, 0.2, 0.3], 0, 1)

        check_multiple_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], 2)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_swap_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with SWAP gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply SWAP gate with both single index and multiple indices variations
        circuit.SWAP(0, 1)

        # Define controlled-SWAP gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCSWAP([0], 1, 2)

        check_multiple_controlled_circuit.MCSWAP([0, 1], 2, 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cx_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CX gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CX gate with both single index and multiple indices variations
        circuit.CX(0, 1)

        # Define controlled-CX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCX([0, 1], 2)

        check_multiple_controlled_circuit.MCX([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cy_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CY gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CY gate with both single index and multiple indices variations
        circuit.CY(0, 1)

        # Define controlled-CY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCY([0, 1], 2)

        check_multiple_controlled_circuit.MCY([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cz_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CZ gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CZ gate with both single index and multiple indices variations
        circuit.CZ(0, 1)

        # Define controlled-CZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCZ([0, 1], 2)

        check_multiple_controlled_circuit.MCZ([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_ch_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CH gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CH gate with both single index and multiple indices variations
        circuit.CH(0, 1)

        # Define controlled-CH gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCH([0, 1], 2)

        check_multiple_controlled_circuit.MCH([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cs_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CS gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CS gate with both single index and multiple indices variations
        circuit.CS(0, 1)

        # Define controlled-CS gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCS([0, 1], 2)

        check_multiple_controlled_circuit.MCS([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_csdg_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CSdg gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CSdg gate with both single index and multiple indices variations
        circuit.CSdg(0, 1)

        # Define controlled-CSdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCSdg([0, 1], 2)

        check_multiple_controlled_circuit.MCSdg([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_ct_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CT gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CT gate with both single index and multiple indices variations
        circuit.CT(0, 1)

        # Define controlled-CT gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCT([0, 1], 2)

        check_multiple_controlled_circuit.MCT([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_ctdg_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CTdg gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=2)

        # Apply CTdg gate with both single index and multiple indices variations
        circuit.CTdg(0, 1)

        # Define controlled-CTdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=3)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=4)

        check_single_controlled_circuit.MCTdg([0, 1], 2)

        check_multiple_controlled_circuit.MCTdg([0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_crx_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CRX gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=3)

        # Apply CRX gate with both single index and multiple indices variations
        circuit.CRX(0.5, 0, 1)

        # Define controlled-CRX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=4)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=5)

        check_single_controlled_circuit.MCRX(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCRX(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cry_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CRY gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=3)

        # Apply CRY gate with both single index and multiple indices variations
        circuit.CRY(0.5, 0, 1)

        # Define controlled-CRY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=4)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=5)

        check_single_controlled_circuit.MCRY(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCRY(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_crz_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CRZ gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=3)

        # Apply CRZ gate with both single index and multiple indices variations
        circuit.CRZ(0.5, 0, 1)

        # Define controlled-CRZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=4)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=5)

        check_single_controlled_circuit.MCRZ(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cphase_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CPhase gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=3)

        # Apply CPhase gate with both single index and multiple indices variations
        circuit.CPhase(0.5, 0, 1)

        # Define controlled-CPhase gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=4)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=5)

        check_single_controlled_circuit.MCPhase(0.5, [0, 1], 2)

        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cswap_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CSWAP gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=3)

        # Apply CSWAP gate with both single index and multiple indices variations
        circuit.CSWAP(0, 1, 2)

        # Define controlled-CSWAP gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=4)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=5)

        check_single_controlled_circuit.MCSWAP([0, 1], 2, 3)

        check_multiple_controlled_circuit.MCSWAP([0, 1, 2], 3, 4)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_cu3_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with CU3 gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=3)

        # Apply CU3 gate with both single index and multiple indices variations
        circuit.CU3([0.1, 0.2, 0.3], 0, 1)

        # Define controlled-CU3 gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=4)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=5)

        check_single_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1], 2)

        check_multiple_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1, 2], 3)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcx_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCX gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCX gate with both single index and multiple indices variations
        circuit.MCX([0, 1], [2, 3])

        # Define controlled-MCX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCX([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCX([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcy_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCY gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCY gate with both single index and multiple indices variations
        circuit.MCY([0, 1], [2, 3])

        # Define controlled-MCY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCY([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCY([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcz_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCZ gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCZ gate with both single index and multiple indices variations
        circuit.MCZ([0, 1], [2, 3])

        # Define controlled-MCZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCZ([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCZ([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mch_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCH gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCH gate with both single index and multiple indices variations
        circuit.MCH([0, 1], [2, 3])

        # Define controlled-MCH gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCH([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCH([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcs_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCS gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCS gate with both single index and multiple indices variations
        circuit.MCS([0, 1], [2, 3])

        # Define controlled-MCS gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCS([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCS([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcsdg_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCSdg gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCSdg gate with both single index and multiple indices variations
        circuit.MCSdg([0, 1], [2, 3])

        # Define controlled-MCSdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCSdg([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCSdg([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mct_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCT gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCT gate with both single index and multiple indices variations
        circuit.MCT([0, 1], [2, 3])

        # Define controlled-MCT gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCT([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCT([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mctdg_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCTdg gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCTdg gate with both single index and multiple indices variations
        circuit.MCTdg([0, 1], [2, 3])

        # Define controlled-MCTdg gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCTdg([0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCTdg([0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcrx_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCRX gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=5)

        # Apply MCRX gate with both single index and multiple indices variations
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # Define controlled-MCRX gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=6)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=7)

        check_single_controlled_circuit.MCRX(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCRX(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcry_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCRY gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=5)

        # Apply MCRY gate with both single index and multiple indices variations
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # Define controlled-MCRY gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=6)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=7)

        check_single_controlled_circuit.MCRY(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCRY(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcrz_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCRZ gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=5)

        # Apply MCRZ gate with both single index and multiple indices variations
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Define controlled-MCRZ gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=6)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=7)

        check_single_controlled_circuit.MCRZ(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCRZ(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcphase_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCPhase gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=5)

        # Apply MCPhase gate with both single index and multiple indices variations
        circuit.MCPhase(0.5, [0, 1], [2, 3])

        # Define controlled-MCPhase gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=6)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=7)

        check_single_controlled_circuit.MCPhase(0.5, [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCPhase(0.5, [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcu3_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCU3 gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=5)

        # Apply MCU3 gate with both single index and multiple indices variations
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        # Define controlled-MCU3 gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=6)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=7)

        check_single_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1, 2], [3, 4])

        check_multiple_controlled_circuit.MCU3([0.1, 0.2, 0.3], [0, 1, 2, 3], [4, 5])

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_mcswap_control(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with MCSWAP gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=4)

        # Apply MCSWAP gate with both single index and multiple indices variations
        circuit.MCSWAP([0, 1], 2, 3)

        # Define controlled-MCSWAP gates
        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=5)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=6)

        check_single_controlled_circuit.MCSWAP([0, 1, 2], 3, 4)

        check_multiple_controlled_circuit.MCSWAP([0, 1, 2, 3], 4, 5)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_global_phase_in_target(
            self,
            circuit_framework: Type[Circuit]
        ) -> None:
        """ Test the `.control()` method with global phase in target gate.

        Parameters
        ----------
        `circuit_framework` : type[qickit.circuit.Circuit]
            The framework to convert the circuit to.
        """
        circuit = circuit_framework(num_qubits=1)

        # Create a circuit with global phase
        circuit.X(0)
        circuit.GlobalPhase(0.5)

        single_controlled_circuit = circuit.control(1)
        multiple_controlled_circuit = circuit.control(2)

        # Define checkers
        check_single_controlled_circuit = circuit_framework(num_qubits=2)
        check_multiple_controlled_circuit = circuit_framework(num_qubits=3)

        check_single_controlled_circuit.MCX(0, 1)
        check_single_controlled_circuit.Phase(0.5, 0)

        check_multiple_controlled_circuit.MCX([0, 1], 2)
        check_multiple_controlled_circuit.MCPhase(0.5, 0, 1)

        assert single_controlled_circuit == check_single_controlled_circuit
        assert multiple_controlled_circuit == check_multiple_controlled_circuit