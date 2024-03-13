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

__all__ = ['TestDwaveCircuit']

import numpy as np

# D-wave imports
from dwave.gate import Circuit as DWCircuit
from dwave.gate.operations import *

# QICKIT imports
from qickit.circuit import DwaveCircuit
from tester.circuit.test_circuit import TestCircuit


class TestDwaveCircuit(TestCircuit):
    """ `qickit.TestDwaveCircuit` is the tester class for `qickit.DwaveCircuit` class.
    """
    def test_circuit_initialization(self) -> None:
        """ Test the initialization of the D-wave circuit.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        assert circuit.circuit == DWCircuit(1, 1)

    def test_X(self) -> None:
        """ Test the Pauli-X gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the Pauli-X gate
        circuit.X(0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            X(q[0])

        assert circuit.circuit == dwave_circuit

    def test_Y(self) -> None:
        """ Test the Pauli-Y gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the Pauli-Y gate
        circuit.Y(0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            Y(q[0])

        assert circuit.circuit == dwave_circuit

    def test_Z(self) -> None:
        """ Test the Pauli-Z gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the Pauli-Z gate
        circuit.Z(0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            Z(q[0])

        assert circuit.circuit == dwave_circuit

    def test_H(self) -> None:
        """ Test the Hadamard gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the Hadamard gate
        circuit.H(0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            Hadamard(q[0])

        assert circuit.circuit == dwave_circuit

    def test_S(self) -> None:
        """ Test the Clifford-S gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the Clifford-S gate
        circuit.S(0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            S(q[0])

        assert circuit.circuit == dwave_circuit

    def test_T(self) -> None:
        """ Test the Clifford-T gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the Clifford-T gate
        circuit.T(0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            T(q[0])

        assert circuit.circuit == dwave_circuit

    def test_RX(self) -> None:
        """ Test the RX gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the RX gate
        circuit.RX(0.5, 0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            RX(0.5)(q[0])

        assert circuit.circuit == dwave_circuit

    def test_RY(self) -> None:
        """ Test the RY gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the RY gate
        circuit.RY(0.5, 0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            RY(0.5)(q[0])

        assert circuit.circuit == dwave_circuit

    def test_RZ(self) -> None:
        """ Test the RZ gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the RZ gate
        circuit.RZ(0.5, 0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            RZ(0.5)(q[0])

        assert circuit.circuit == dwave_circuit

    def test_U3(self) -> None:
        """ Test the U3 gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(1, 1)

        # Apply the U3 gate
        circuit.U3([0.5, 0.5, 0.5], 0)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(1, 1)

        with dwave_circuit.context as (q, c):
            Rotation(np.pi, np.pi, np.pi)(q[0])

        assert circuit.circuit == dwave_circuit

    def test_CX(self) -> None:
        """ Test the CX gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CX gate
        circuit.CX(0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        with dwave_circuit.context as (q, c):
            CNOT(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CY(self) -> None:
        """ Test the CY gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CY gate
        circuit.CY(0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        with dwave_circuit.context as (q, c):
            CY(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CZ(self) -> None:
        """ Test the CZ gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CZ gate
        circuit.CZ(0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        with dwave_circuit.context as (q, c):
            CZ(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CH(self) -> None:
        """ Test the CH gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CH gate
        circuit.CH(0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        with dwave_circuit.context as (q, c):
            CHadamard(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CS(self) -> None:
        """ Test the CS gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CS gate
        circuit.CS(0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        # As of now, dwave.gate does not support CS gates. We can use the ControlledOperation class
        # to create a custom CS gate
        class CSControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = S()

        with dwave_circuit.context as (q, c):
            CSControlledOp(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CT(self) -> None:
        """ Test the CT gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CT gate
        circuit.CT(0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        # As of now, dwave.gate does not support CT gates. We can use the ControlledOperation class
        # to create a custom CT gate
        class CTControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = T()

        with dwave_circuit.context as (q, c):
            CTControlledOp(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CRX(self) -> None:
        """ Test the CRX gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CRX gate
        circuit.CRX(0.5, 0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        # As of now, dwave.gate does not support CRX gates. We can use the ControlledOperation class
        # to create a custom CRX gate
        class CRXControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = RX(0.5)

        with dwave_circuit.context as (q, c):
            CRXControlledOp(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CRY(self) -> None:
        """ Test the CRY gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CRY gate
        circuit.CRY(0.5, 0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        # As of now, dwave.gate does not support CRY gates. We can use the ControlledOperation class
        # to create a custom CRY gate
        class CRYControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = RY(0.5)

        with dwave_circuit.context as (q, c):
            CRYControlledOp(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CRZ(self) -> None:
        """ Test the CRZ gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CRZ gate
        circuit.CRZ(0.5, 0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        # As of now, dwave.gate does not support CRZ gates. We can use the ControlledOperation class
        # to create a custom CRZ gate
        class CRZControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = RZ(0.5)

        with dwave_circuit.context as (q, c):
            CRZControlledOp(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_CU3(self) -> None:
        """ Test the CU3 gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(2, 2)

        # Apply the CU3 gate
        circuit.CU3([0.5, 0.5, 0.5], 0, 1)

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(2, 2)

        # As of now, dwave.gate does not support CU3 gates. We can use the ControlledOperation class
        # to create a custom CU3 gate
        class CU3ControlledOp(ControlledOperation):
            _num_control = 1
            _num_target = 1
            _target_operation = Rotation(0.5, 0.5, 0.5)

        with dwave_circuit.context as (q, c):
            CU3ControlledOp(q[0], q[1])

        assert circuit.circuit == dwave_circuit

    def test_MCX(self) -> None:
        """ Test the MCX gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCX gate
        circuit.MCX([0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        class MCXControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = X()

        with dwave_circuit.context as (q, c):
            MCXControlledOp(q[0], q[1], q[2])
            MCXControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCY(self) -> None:
        """ Test the MCY gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCY gate
        circuit.MCY([0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        class MCYControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = Y()

        with dwave_circuit.context as (q, c):
            MCYControlledOp(q[0], q[1], q[2])
            MCYControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCZ(self) -> None:
        """ Test the MCZ gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCZ gate
        circuit.MCZ([0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        class MCZControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = Z()

        with dwave_circuit.context as (q, c):
            MCZControlledOp(q[0], q[1], q[2])
            MCZControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCH(self) -> None:
        """ Test the MCH gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCH gate
        circuit.MCH([0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        class MCHControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = Hadamard()

        with dwave_circuit.context as (q, c):
            MCHControlledOp(q[0], q[1], q[2])
            MCHControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCS(self) -> None:
        """ Test the MCS gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCS gate
        circuit.MCS([0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        # As of now, dwave.gate does not support MCS gates. We can use the ControlledOperation class
        # to create a custom MCS gate
        class MCSControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = S()

        with dwave_circuit.context as (q, c):
            MCSControlledOp(q[0], q[1], q[2])
            MCSControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCT(self) -> None:
        """ Test the MCT gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCT gate
        circuit.MCT([0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        # As of now, dwave.gate does not support MCT gates. We can use the ControlledOperation class
        # to create a custom MCT gate
        class MCTControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = T()

        with dwave_circuit.context as (q, c):
            MCTControlledOp(q[0], q[1], q[2])
            MCTControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCRX(self) -> None:
        """ Test the MCRX gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCRX gate
        circuit.MCRX(0.5, [0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        # As of now, dwave.gate does not support MCRX gates. We can use the ControlledOperation class
        # to create a custom MCRX gate
        class MCRXControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = RX(0.5)

        with dwave_circuit.context as (q, c):
            MCRXControlledOp(q[0], q[1], q[2])
            MCRXControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCRY(self) -> None:
        """ Test the MCRY gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCRY gate
        circuit.MCRY(0.5, [0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        # As of now, dwave.gate does not support MCRY gates. We can use the ControlledOperation class
        # to create a custom MCRY gate
        class MCRYControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = RY(0.5)

        with dwave_circuit.context as (q, c):
            MCRYControlledOp(q[0], q[1], q[2])
            MCRYControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCRZ(self) -> None:
        """ Test the MCRZ gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCRZ gate
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        # As of now, dwave.gate does not support MCRZ gates. We can use the ControlledOperation class
        # to create a custom MCRZ gate
        class MCRZControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = RZ(0.5)

        with dwave_circuit.context as (q, c):
            MCRZControlledOp(q[0], q[1], q[2])
            MCRZControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit

    def test_MCU3(self) -> None:
        """ Test the MCU3 gate.
        """
        # Define the `qickit.DwaveCircuit` instance
        circuit = DwaveCircuit(4, 4)

        # Apply the MCU3 gate
        circuit.MCU3([0.5, 0.5, 0.5], [0, 1], [2, 3])

        # Define the equivalent `dwave.gate.Circuit` instance, and
        # ensure they are equivalent
        dwave_circuit = DWCircuit(4, 4)

        # As of now, dwave.gate does not support MCU3 gates. We can use the ControlledOperation class
        # to create a custom MCU3 gate
        class MCU3ControlledOp(ControlledOperation):
            _num_control = len(2)
            _num_target = 1
            _target_operation = Rotation(0.5, 0.5, 0.5)

        with dwave_circuit.context as (q, c):
            MCU3ControlledOp(q[0], q[1], q[2])
            MCU3ControlledOp(q[0], q[1], q[3])

        assert circuit.circuit == dwave_circuit