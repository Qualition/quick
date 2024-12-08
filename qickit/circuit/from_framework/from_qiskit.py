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

""" Converter for quantum circuits from Qiskit to Qickit.
"""

from __future__ import annotations

__all__ = ["FromQiskit"]

from qiskit import QuantumCircuit, transpile # type: ignore
from qiskit._accelerate.circuit import CircuitInstruction # type: ignore
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.circuit.from_framework import FromFramework


class FromQiskit(FromFramework):
    """ `qickit.circuit.from_framework.FromQiskit` is a class for converting quantum circuits from
    Qiskit to `qickit.circuit.Circuit` class.

    Notes
    -----
    The conversion is done by first transpiling the circuit to u3 and cx gates, and then extracting
    the parameters of the gates in the Qiskit circuit. We perform transpilation to the minimal gateset
    of [u3, cx, global phase] to allow for support of future Qiskit gates, as well as custom ones that
    are not native to Qiskit.

    This is done to ensure that the conversion is as general as possible without having to update the
    converter for every new gate that is added to Qiskit, or for gates that are not currently implemented
    in Qickit.

    Kindly note that the conversion is not perfect, given that classical control flow operations such as
    `IfElse`, `ForLoop`, etc., are not currently supported in Qickit. Furthermore, parametric gates where
    the parameter is a variable (not a constant) are not supported at the moment.

    The conversion is limited to the unitary quantum gates, global phase, and measurement gates.

    Parameters
    ----------
    `output_framework` : type[Circuit]
        The quantum computing framework to convert the quantum circuit to.

    Attributes
    ----------
    `output_framework` : type[Circuit]
        The quantum computing framework to convert the quantum circuit to.
    `gate_mapping` : dict[str, Callable]
        The mapping of the gates in Qiskit to the gates in Qickit.
    `skip_gates` : list[str]
        The list of gates to skip while converting the quantum circuit.

    Raises
    ------
    `TypeError`
        - If the `output_framework` is not a subclass of `qickit.circuit.Circuit`.

    Usage
    -----
    >>> qiskit_converter = FromQiskit(output_framework=QiskitCircuit)
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:

        super().__init__(output_framework=output_framework)

        self.gate_mapping = {
            "cx": self._extract_cx_gate_params,
            "u3": self._extract_u3_gate_params,
            "u": self._extract_u3_gate_params,
            "measure": self._extract_measure_gate_params
        }

        self.skip_gates = ["barrier", "reset"]

    def _extract_u3_gate_params(
            self,
            gate: CircuitInstruction,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a U3 gate.

        Parameters
        ----------
        `gate` : qiskit._accelerate.circuit.CircuitInstruction
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "U3",
            "qubit_indices": int(gate.qubits[0]._index),
            "angles": gate.operation.params
        })

    def _extract_cx_gate_params(
            self,
            gate: CircuitInstruction,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a CX gate.

        Parameters
        ----------
        `gate` : qiskit._accelerate.circuit.CircuitInstruction
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "CX",
            "control_index": int(gate.qubits[0]._index),
            "target_index": int(gate.qubits[1]._index),
        })

    def _extract_measure_gate_params(
            self,
            gate: CircuitInstruction,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a measurement gate.

        Parameters
        ----------
        `gate` : qiskit._accelerate.circuit.CircuitInstruction
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "measure",
            "qubit_indices": [int(qubit._index) for qubit in gate.qubits]
        })

    def extract_params(
            self,
            circuit: QuantumCircuit
        ) -> list[dict]:
        """Extract the parameters of the gates in the Qiskit circuit.

        Parameters
        ----------
        `circuit` : qiskit.QuantumCircuit
            The quantum circuit to extract the parameters from.

        Returns
        -------
        `params` : list[dict]
            The list of parameters of the gates in the Qiskit circuit.

        Raises
        ------
        NotImplementedError
            - If the gate is not found in the gate mapping.
        """
        params: list[dict] = []

        for gate in circuit.data:
            gate_name = gate.operation.name

            if gate_name in self.skip_gates:
                continue

            self.gate_mapping[gate_name](gate, params)

        return params

    def convert(
            self,
            circuit: QuantumCircuit
        ) -> Circuit:

        # Define a circuit
        num_qubits = circuit.num_qubits
        qickit_circuit = self.output_framework(num_qubits=num_qubits)

        # We first transpile the circuit to the minimal gateset of [u3, cx, global phase]
        # This allows for support of future Qiskit gates, as well as custom ones
        # that are not native to Qiskit
        circuit = transpile(
            circuit,
            basis_gates=["u3", "cx"]
        )

        # Extract the parameters of the gates in the Qiskit circuit
        params = self.extract_params(circuit)

        # Add the gates to the Qickit circuit
        for param in params:
            gate_name = param.pop("gate")
            getattr(qickit_circuit, gate_name)(**param)

        qickit_circuit.GlobalPhase(circuit.global_phase)

        return qickit_circuit