# Copyright 2023-2024 Qualition Computing LLC.
#
# Licensed under the GNU Version 3.0 (the "License");
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

from abc import ABC, abstractmethod


class Template(ABC):
    """ `qickit.Template` is the template for creating circuit testers.
    """
    @abstractmethod
    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        pass

    @abstractmethod
    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        pass

    @abstractmethod
    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        pass

    @abstractmethod
    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        pass

    @abstractmethod
    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """
        pass

    @abstractmethod
    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """
        pass

    @abstractmethod
    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        pass

    @abstractmethod
    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        pass

    @abstractmethod
    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        pass

    @abstractmethod
    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        pass

    @abstractmethod
    def test_CX(self) -> None:
        """ Test the Controlled Pauli-X gate.
        """
        pass

    @abstractmethod
    def test_CY(self) -> None:
        """ Test the Controlled Pauli-Y gate.
        """
        pass

    @abstractmethod
    def test_CZ(self) -> None:
        """ Test the Controlled Pauli-Z gate.
        """
        pass

    @abstractmethod
    def test_CH(self) -> None:
        """ Test the Controlled Hadamard gate.
        """
        pass

    @abstractmethod
    def test_CS(self) -> None:
        """ Test the Controlled Clifford-S gate.
        """
        pass

    @abstractmethod
    def test_CT(self) -> None:
        """ Test the Controlled Clifford-T gate.
        """
        pass

    @abstractmethod
    def test_CRX(self) -> None:
        """ Test the Controlled RX gate.
        """
        pass

    @abstractmethod
    def test_CRY(self) -> None:
        """ Test the Controlled RY gate.
        """
        pass

    @abstractmethod
    def test_CRZ(self) -> None:
        """ Test the Controlled RZ gate.
        """
        pass

    @abstractmethod
    def test_CU3(self) -> None:
        """ Test the Controlled U3 gate.
        """
        pass

    @abstractmethod
    def test_MCX(self) -> None:
        """ Test the Multi-Controlled Pauli-X gate.
        """
        pass

    @abstractmethod
    def test_MCY(self) -> None:
        """ Test the Multi-Controlled Pauli-Y gate.
        """
        pass

    @abstractmethod
    def test_MCZ(self) -> None:
        """ Test the Multi-Controlled Pauli-Z gate.
        """
        pass

    @abstractmethod
    def test_MCH(self) -> None:
        """ Test the Multi-Controlled Hadamard gate.
        """
        pass

    @abstractmethod
    def test_MCS(self) -> None:
        """ Test the Multi-Controlled Clifford-S gate.
        """
        pass

    @abstractmethod
    def test_MCT(self) -> None:
        """ Test the Multi-Controlled Clifford-T gate.
        """
        pass

    @abstractmethod
    def test_MCRX(self) -> None:
        """ Test the Multi-Controlled RX gate.
        """
        pass

    @abstractmethod
    def test_MCRY(self) -> None:
        """ Test the Multi-Controlled RY gate.
        """
        pass

    @abstractmethod
    def test_MCRZ(self) -> None:
        """ Test the Multi-Controlled RZ gate.
        """
        pass

    @abstractmethod
    def test_MCU3(self) -> None:
        """ Test the Multi-Controlled U3 gate.
        """
        pass

    @abstractmethod
    def test_measure(self) -> None:
        """ Test the measurement gate.
        """
        pass