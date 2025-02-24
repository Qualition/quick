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

__all__ = ["TestControlState"]

import numpy as np
import pytest
from typing import Type

from quick.circuit import Circuit
from quick.circuit.gate_matrix import RX

from tests.circuit import CIRCUIT_FRAMEWORKS


class TestControlState:
    """ `tests.circuit.TestControlState` is the tester class for testing the
    `control_state` parameter of `quick.circuit.Circuit` controlled gate methods.
    """
    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CX(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CX(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CY(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CY(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CZ(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CZ(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CH_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CH` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CH(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CH(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CS_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CS` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CS(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CS(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CSdg_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CSdg` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CSdg(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CSdg(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CT_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CT` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CT(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CT(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CTdg_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CTdg` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CTdg(0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CTdg(0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CRX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CRX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CRX(0.1, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CRX(0.1, 0, 1)
        checker_circuit.X(0)

        print(repr(circuit))
        print(repr(checker_circuit))
        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CRY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CRY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CRY(0.1, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CRY(0.1, 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CRZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CRZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CRZ(0.1, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CRZ(0.1, 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CPhase_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CPhase` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CPhase(0.1, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CPhase(0.1, 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CXPow_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CXPow` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CXPow(0.1, 0, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CXPow(0.1, 0, 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CYPow_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CYPow` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CYPow(0.1, 0, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CYPow(0.1, 0, 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CZPow_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CZPow` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CZPow(0.1, 0, 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CZPow(0.1, 0, 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CRXX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CRXX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.CRXX(0.1, 0, 1, 2, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.CRXX(0.1, 0, 1, 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CRYY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CRYY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.CRYY(0.1, 0, 1, 2, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.CRYY(0.1, 0, 1, 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CRZZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CRZZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.CRZZ(0.1, 0, 1, 2, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.CRZZ(0.1, 0, 1, 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CU3_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CU3` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1
        phi = 0.2
        lam = 0.3

        # Given control state "0"
        circuit = circuit_framework(2)
        circuit.CU3([theta, phi, lam], 0, 1, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(2)
        checker_circuit.X(0)
        checker_circuit.CU3([theta, phi, lam], 0, 1)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_CSWAP_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `CSWAP` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.CSWAP(0, 1, 2, control_state="0")

        # Ensure the control qubit is sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.CSWAP(0, 1, 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCX([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCX([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCX([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCX([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCY([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCY([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCY([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCY([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCZ([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCZ([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCZ([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCZ([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCH_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCH` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCH([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCH([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCH([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCH([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCS_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCS` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCS([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCS([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCS([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCS([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCSdg_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCSdg` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCSdg([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCSdg([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCSdg([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCSdg([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCT_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCT` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCT([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCT([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCT([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCT([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCTdg_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCTdg` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCTdg([0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCTdg([0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCTdg([0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCTdg([0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCRX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCRX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCRX(theta, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCRX(theta, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCRX(theta, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCRX(theta, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCRY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCRY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCRY(theta, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCRY(theta, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCRY(theta, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCRY(theta, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCRZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCRZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCRZ(theta, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCRZ(theta, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCRZ(theta, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCRZ(theta, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCPhase_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCPhase` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCPhase(theta, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCPhase(theta, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCPhase(theta, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCPhase(theta, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCXPow_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCXPow` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1
        global_phase = 0.2

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCXPow(theta, global_phase, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCXPow(theta, global_phase, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCXPow(theta, global_phase, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCXPow(theta, global_phase, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCYPow_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCYPow` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1
        global_phase = 0.2

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCYPow(theta, global_phase, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCYPow(theta, global_phase, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCYPow(theta, global_phase, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCYPow(theta, global_phase, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCZPow_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCZPow` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1
        global_phase = 0.2

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCZPow(theta, global_phase, [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCZPow(theta, global_phase, [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCZPow(theta, global_phase, [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCZPow(theta, global_phase, [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCRXX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCRXX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(6)
        circuit.MCRXX(theta, [0, 1, 2, 3], 4, 5, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X(3)
        checker_circuit.MCRXX(theta, [0, 1, 2, 3], 4, 5)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(6)
        circuit.MCRXX(theta, [0, 1, 2, 3], 4, 5, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCRXX(theta, [0, 1, 2, 3], 4, 5)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCRYY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCRYY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(6)
        circuit.MCRYY(theta, [0, 1, 2, 3], 4, 5, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X(3)
        checker_circuit.MCRYY(theta, [0, 1, 2, 3], 4, 5)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(6)
        circuit.MCRYY(theta, [0, 1, 2, 3], 4, 5, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCRYY(theta, [0, 1, 2, 3], 4, 5)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCRZZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCRZZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1

        # Given control state "0"
        circuit = circuit_framework(6)
        circuit.MCRZZ(theta, [0, 1, 2, 3], 4, 5, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X(3)
        checker_circuit.MCRZZ(theta, [0, 1, 2, 3], 4, 5)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(6)
        circuit.MCRZZ(theta, [0, 1, 2, 3], 4, 5, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCRZZ(theta, [0, 1, 2, 3], 4, 5)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCU3_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCU3` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = 0.1
        phi = 0.2
        lam = 0.3

        # Given control state "0"
        circuit = circuit_framework(5)
        circuit.MCU3([theta, phi, lam], [0, 1, 2, 3], 4, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X(3)
        checker_circuit.MCU3([theta, phi, lam], [0, 1, 2, 3], 4)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(5)
        circuit.MCU3([theta, phi, lam], [0, 1, 2, 3], 4, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(5)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCU3([theta, phi, lam], [0, 1, 2, 3], 4)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_MCSWAP_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `MCSWAP` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        # Given control state "0"
        circuit = circuit_framework(6)
        circuit.MCSWAP([0, 1, 2, 3], 4, 5, control_state="1110")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X(3)
        checker_circuit.MCSWAP([0, 1, 2, 3], 4, 5)
        checker_circuit.X(3)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(6)
        circuit.MCSWAP([0, 1, 2, 3], 4, 5, control_state="0001")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(6)
        checker_circuit.X([0, 1, 2])
        checker_circuit.MCSWAP([0, 1, 2, 3], 4, 5)
        checker_circuit.X([0, 1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UCRX_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `UCRX` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = [0.1, 0.2, 0.3, 0.4]

        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.UCRX(theta, [0, 1], 2, control_state="10")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(1)
        checker_circuit.UCRX(theta, [0, 1], 2)
        checker_circuit.X(1)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(3)
        circuit.UCRX(theta, [0, 1], 2, control_state="01")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.UCRX(theta, [0, 1], 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UCRY_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `UCRY` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = [0.1, 0.2, 0.3, 0.4]

        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.UCRY(theta, [0, 1], 2, control_state="10")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(1)
        checker_circuit.UCRY(theta, [0, 1], 2)
        checker_circuit.X(1)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(3)
        circuit.UCRY(theta, [0, 1], 2, control_state="01")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.UCRY(theta, [0, 1], 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_UCRZ_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `UCRZ` method of the circuit
        framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        theta = [0.1, 0.2, 0.3, 0.4]

        # Given control state "0"
        circuit = circuit_framework(3)
        circuit.UCRZ(theta, [0, 1], 2, control_state="10")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(1)
        checker_circuit.UCRZ(theta, [0, 1], 2)
        checker_circuit.X(1)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(3)
        circuit.UCRZ(theta, [0, 1], 2, control_state="01")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(3)
        checker_circuit.X(0)
        checker_circuit.UCRZ(theta, [0, 1], 2)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_Multiplexor_control_state(
            self,
            circuit_framework: Type[Circuit],
        ) -> None:
        """ Test the `control_state` parameter of the `Multiplexor` method of the
        circuit framework.

        Parameters
        ----------
        `circuit_framework` : Type[quick.circuit.Circuit]
            The circuit framework to test.
        """
        gates = [
            RX(np.pi/2).matrix,
            RX(np.pi/3).matrix,
            RX(np.pi/4).matrix,
            RX(np.pi/5).matrix,
            RX(np.pi/6).matrix,
            RX(np.pi/7).matrix,
            RX(np.pi/8).matrix,
            RX(np.pi/9).matrix
        ]

        # Given control state "0"
        circuit = circuit_framework(4)
        circuit.Multiplexor(gates, [0, 1, 2], 3, control_state="011")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(4)
        checker_circuit.X(0)
        checker_circuit.Multiplexor(gates, [0, 1, 2], 3)
        checker_circuit.X(0)

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit

        circuit = circuit_framework(4)
        circuit.Multiplexor(gates, [0, 1, 2], 3, control_state="100")

        # Ensure the control qubits are sandwiched between X gates
        checker_circuit = circuit_framework(4)
        checker_circuit.X([1, 2])
        checker_circuit.Multiplexor(gates, [0, 1, 2], 3)
        checker_circuit.X([1, 2])

        # Check the circuit is equivalent to the checker circuit
        assert circuit == checker_circuit