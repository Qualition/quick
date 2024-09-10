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

__all__ = ["TestConvert"]

from typing import Type

from qickit.circuit import Circuit, CirqCircuit, PennylaneCircuit, QiskitCircuit, TKETCircuit


class TestConvert:
    """ `tests.circuit.TestConvert` is the tester for the `.convert()` method.
    """
    def from_circuit(
            self,
            framework: Type[Circuit]
        ) -> None:
        """ Test the conversion from `qickit.circuit.CirqCircuit`.

        Parameters
        ----------
        framework : Type[Circuit]
            The framework to convert the circuit to.
        """
        circuit = framework(num_qubits=5)

        # Apply single qubit gates with both single index and multiple indices variations
        circuit.X(0)
        circuit.X([0, 1])
        circuit.Y(0)
        circuit.Y([0, 1])
        circuit.Z(0)
        circuit.Z([0, 1])
        circuit.H(0)
        circuit.H([0, 1])
        circuit.S(0)
        circuit.S([0, 1])
        circuit.Sdg(0)
        circuit.Sdg([0, 1])
        circuit.T(0)
        circuit.T([0, 1])
        circuit.Tdg(0)
        circuit.Tdg([0, 1])
        circuit.RX(0.5, 0)
        circuit.RX(0.5, [0, 1])
        circuit.RY(0.5, 0)
        circuit.RY(0.5, [0, 1])
        circuit.RZ(0.5, 0)
        circuit.RZ(0.5, [0, 1])
        circuit.U3([0.1, 0.2, 0.3], 0)
        circuit.SWAP(0, 1)

        # Apply controlled gates
        circuit.CX(0, 1)
        circuit.CY(0, 1)
        circuit.CZ(0, 1)
        circuit.CH(0, 1)
        circuit.CS(0, 1)
        circuit.CSdg(0, 1)
        circuit.CT(0, 1)
        circuit.CTdg(0, 1)
        circuit.CRX(0.5, 0, 1)
        circuit.CRY(0.5, 0, 1)
        circuit.CRZ(0.5, 0, 1)
        circuit.CU3([0.1, 0.2, 0.3], 0, 1)
        circuit.CSWAP(0, 1, 2)

        # Apply multi-controlled gates with both single index and multiple indices variations
        circuit.MCX(0, 1)
        circuit.MCX([0, 1], 2)
        circuit.MCX(0, [1, 2])
        circuit.MCX([0, 1], [2, 3])

        circuit.MCY(0, 1)
        circuit.MCY([0, 1], 2)
        circuit.MCY(0, [1, 2])
        circuit.MCY([0, 1], [2, 3])

        circuit.MCZ(0, 1)
        circuit.MCZ([0, 1], 2)
        circuit.MCZ(0, [1, 2])
        circuit.MCZ([0, 1], [2, 3])

        circuit.MCH(0, 1)
        circuit.MCH([0, 1], 2)
        circuit.MCH(0, [1, 2])
        circuit.MCH([0, 1], [2, 3])

        circuit.MCS(0, 1)
        circuit.MCS([0, 1], 2)
        circuit.MCS(0, [1, 2])
        circuit.MCS([0, 1], [2, 3])

        circuit.MCSdg(0, 1)
        circuit.MCSdg([0, 1], 2)
        circuit.MCSdg(0, [1, 2])
        circuit.MCSdg([0, 1], [2, 3])

        circuit.MCT(0, 1)
        circuit.MCT([0, 1], 2)
        circuit.MCT(0, [1, 2])
        circuit.MCT([0, 1], [2, 3])

        circuit.MCTdg(0, 1)
        circuit.MCTdg([0, 1], 2)
        circuit.MCTdg(0, [1, 2])
        circuit.MCTdg([0, 1], [2, 3])

        circuit.MCRX(0.5, 0, 1)
        circuit.MCRX(0.5, [0, 1], 2)
        circuit.MCRX(0.5, 0, [1, 2])
        circuit.MCRX(0.5, [0, 1], [2, 3])

        circuit.MCRY(0.5, 0, 1)
        circuit.MCRY(0.5, [0, 1], 2)
        circuit.MCRY(0.5, 0, [1, 2])
        circuit.MCRY(0.5, [0, 1], [2, 3])

        circuit.MCRZ(0.5, 0, 1)
        circuit.MCRZ(0.5, [0, 1], 2)
        circuit.MCRZ(0.5, 0, [1, 2])
        circuit.MCRZ(0.5, [0, 1], [2, 3])

        circuit.MCU3([0.1, 0.2, 0.3], 0, 1)
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], 2)
        circuit.MCU3([0.1, 0.2, 0.3], 0, [1, 2])
        circuit.MCU3([0.1, 0.2, 0.3], [0, 1], [2, 3])

        circuit.MCSWAP(0, 1, 2)
        circuit.MCSWAP([0, 1], 2, 3)

        # Convert the circuit
        converted_circuit_cirq = circuit.convert(CirqCircuit)
        converted_circuit_pennylane = circuit.convert(PennylaneCircuit)
        converted_circuit_qiskit = circuit.convert(QiskitCircuit)
        converted_circuit_tket = circuit.convert(TKETCircuit)

        # Check the converted circuit
        assert circuit == converted_circuit_cirq
        assert circuit == converted_circuit_pennylane
        assert circuit == converted_circuit_qiskit
        assert circuit == converted_circuit_tket

    def test_from_cirq(self) -> None:
        """ Test the conversion from `qickit.circuit.CirqCircuit`.
        """
        self.from_circuit(CirqCircuit)

    def test_from_pennylane(self) -> None:
        """ Test the conversion from `qickit.circuit.PennylaneCircuit`.
        """
        self.from_circuit(PennylaneCircuit)

    def test_from_qiskit(self) -> None:
        """ Test the conversion from `qickit.circuit.QiskitCircuit`.
        """
        self.from_circuit(QiskitCircuit)

    def test_from_tket(self) -> None:
        """ Test the conversion from `qickit.circuit.TKETCircuit`.
        """
        self.from_circuit(TKETCircuit)