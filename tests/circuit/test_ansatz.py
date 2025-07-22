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

__all__ = ["TestAnsatz"]

import pytest
from quick.circuit import Ansatz, Circuit

from tests.circuit import CIRCUIT_FRAMEWORKS


class TestAnsatz:
    def test_init_type_error(self) -> None:
        """ Test TypeError raised when invalid type is passed to Ansatz.
        """
        with pytest.raises(TypeError):
            Ansatz(2) # type: ignore

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_init_value_error(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test ValueError raised when the circuit used is not
        parameterized.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(2)

        circuit.CX(0, 1)

        with pytest.raises(ValueError):
            Ansatz(circuit)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_thetas(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test thetas property of Ansatz.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(1)

        circuit.RY(0.3, 0)

        ansatz = Ansatz(circuit)

        assert ansatz.thetas == [0.3]

        circuit = circuit_framework(2)
        circuit.RY(0.3, 0)
        circuit.RY(0.2, 1)

        ansatz = Ansatz(circuit)

        assert ansatz.thetas == [0.3, 0.2]

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_set_thetas(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test thetas setter of Ansatz.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(1)

        circuit.RY(0.3, 0)

        ansatz = Ansatz(circuit)

        ansatz.thetas = [0.3]

        assert ansatz.thetas == [0.3]
        assert circuit.circuit_log[0]["angle"] == 0.3

        circuit = circuit_framework(2)
        circuit.RY(0.3, 0)
        circuit.RY(0.2, 1)

        ansatz = Ansatz(circuit)

        ansatz.thetas = [0.3, 0.2]

        assert ansatz.thetas == [0.3, 0.2]
        assert circuit.circuit_log[0]["angle"] == 0.3
        assert circuit.circuit_log[1]["angle"] == 0.2

        circuit = circuit_framework(1)

        circuit.RY(0.3, 0)
        circuit.U3([0.1, 0.2, 0.3], 0)
        circuit.RX(0.2, 0)

        ansatz = Ansatz(circuit)

        ansatz.thetas = [0.1, [0.2, 0.3, 0.4], 0.5]

        assert circuit.circuit_log[0]["angle"] == 0.1
        assert circuit.circuit_log[1]["angles"] == [0.2, 0.3, 0.4]
        assert circuit.circuit_log[2]["angle"] == 0.5

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_num_params(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test num_params property of Ansatz.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(1)

        circuit.RY(0.3, 0)

        ansatz = Ansatz(circuit)

        assert ansatz.num_params == 1

        circuit = circuit_framework(2)
        circuit.RY(0.3, 0)
        circuit.RY(0.2, 1)

        ansatz = Ansatz(circuit)

        assert ansatz.num_params == 2

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_num_parameterized_gates(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test num_parameterized_gates property of Ansatz.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(1)

        circuit.RY(0.3, 0)

        ansatz = Ansatz(circuit)

        assert ansatz.num_parameterized_gates == 1

        circuit = circuit_framework(2)
        circuit.RY(0.3, 0)
        circuit.RY(0.2, 1)

        ansatz = Ansatz(circuit)

        assert ansatz.num_parameterized_gates == 2

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_is_parameterized(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test is_parameterized property of Ansatz.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(1)

        circuit.RY(0.3, 0)

        ansatz = Ansatz(circuit)

        assert ansatz.is_parameterized

        circuit = circuit_framework(2)
        circuit.RY(0.3, 0)
        circuit.RY(0.2, 1)

        ansatz = Ansatz(circuit)

        assert ansatz.is_parameterized

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_is_parameterized_error(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test is_parameterized property of Ansatz when circuit is not
        parameterized.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(2)

        circuit.CX(0, 1)

        with pytest.raises(ValueError):
            Ansatz(circuit)

    @pytest.mark.parametrize("circuit_framework", CIRCUIT_FRAMEWORKS)
    def test_is_parameterized_with_ignore_global_phase(
            self,
            circuit_framework: type[Circuit]
        ) -> None:
        """ Test is_parameterized property of Ansatz with ignore_global_phase
        set to True.

        Parameters
        ----------
        `circuit_framework`: type[quick.circuit.Circuit]
            The circuit framework to use for testing.
        """
        circuit = circuit_framework(1)

        circuit.GlobalPhase(0.3)

        ansatz = Ansatz(circuit, ignore_global_phase=False)

        assert ansatz.is_parameterized

        with pytest.raises(ValueError):
            Ansatz(circuit, ignore_global_phase=True)