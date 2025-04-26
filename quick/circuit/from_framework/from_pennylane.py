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


""" Converter for quantum circuits from Pennylane to quick.
"""

from __future__ import annotations

__all__ = ["FromPennyLane"]

import pennylane as qml  # type: ignore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quick.circuit import Circuit
from quick.circuit.from_framework import FromFramework


class FromPennyLane(FromFramework):
    """ `quick.circuit.from_framework.FromPennylane` is a class for converting quantum circuits from
    Pennylane to `quick.circuit.Circuit` class.

    Notes
    -----
    The conversion is done by first transpiling the circuit to rx, rz and cx gates, and then extracting
    the parameters of the gates in the Pennylane circuit. We perform transpilation to the minimal gateset
    of [rx, rz, cx, global phase] to allow for support of future Pennylane gates, as well as custom ones
    that are not native to Pennylane.

    This is done to ensure that the conversion is as general as possible without having to update the
    converter for every new gate that is added to Pennylane, or for gates that are not currently implemented
    in quick.

    The conversion is limited to the unitary quantum gates, global phase, and measurement gates.

    Parameters
    ----------
    `output_framework` : type[quick.circuit.Circuit]
        The quantum computing framework to convert the quantum circuit to.

    Attributes
    ----------
    `output_framework` : type[quick.circuit.Circuit]
        The quantum computing framework to convert the quantum circuit to.
    `gate_mapping` : dict[str, Callable]
        The mapping of the gate names between Pennylane and quick.

    Raises
    ------
    `TypeError`
        - If the `output_framework` is not a subclass of `quick.circuit.Circuit`.

    Usage
    -----
    >>> pennylane_converter = FromPennylane(output_framework=CirqCircuit)
    """
    def __init__(
            self,
            output_framework: type[Circuit]
        ) -> None:

        super().__init__(output_framework=output_framework)

        self.gate_mapping = {
            "RX": self._extract_rx_gate_params,
            "RZ": self._extract_rz_gate_params,
            "CNOT": self._extract_cx_gate_params,
            "GlobalPhase": self._extract_global_phase_gate_params,
            "MidMeasureMP": self._extract_measure_gate_params
        }

    def _extract_rx_gate_params(
            self,
            gate: qml.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a RX gate.

        Parameters
        ----------
        `gate` : qml.Operation
            The RX gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters to extract the parameters to.
        """
        params.append({
            "gate": "RX",
            "angle": gate.data[0],
            "qubit_indices": gate.wires.tolist()
        })

    def _extract_rz_gate_params(
            self,
            gate: qml.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a RZ gate.

        Parameters
        ----------
        `gate` : qml.Operation
            The RZ gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters to extract the parameters to.
        """
        params.append({
            "gate": "RZ",
            "angle": gate.data[0],
            "qubit_indices": gate.wires.tolist()
        })

    def _extract_cx_gate_params(
            self,
            gate: qml.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a CX gate.

        Parameters
        ----------
        `gate` : qml.Operation
            The CX gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters to extract the parameters to.
        """
        params.append({
            "gate": "CX",
            "control_index": gate.wires.tolist()[0],
            "target_index": gate.wires.tolist()[1]
        })

    def _extract_global_phase_gate_params(
            self,
            gate: qml.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a global phase gate.

        Parameters
        ----------
        `gate` : qml.Operation
            The global phase gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters to extract the parameters to.
        """
        params.append({
            "gate": "GlobalPhase",
            "angle": -gate.data[0] # type: ignore
        })

    def _extract_measure_gate_params(
            self,
            gate: qml.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a measurement gate.

        Parameters
        ----------
        `gate` : qml.Operation
            The measurement gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters to extract the parameters to.
        """
        params.append({
            "gate": "measure",
            "qubit_indices": gate.wires.tolist()
        })

    def extract_params(
            self,
            circuit: qml.QNode
        ) -> list[dict]:
        """Extract the parameters of the gates in the Pennylane
        circuit.

        Parameters
        ----------
        `circuit` : qml.QNode
            The quantum circuit to extract the parameters from.

        Returns
        -------
        `params` : list[dict]
            The list of parameters of the gates in the Pennylane
            circuit.

        Raises
        ------
        NotImplementedError
            - If the gate is not found in the gate mapping.
        """
        params: list[dict] = []

        tape = qml.workflow.construct_tape(circuit)()

        print(tape.operations)

        for gate in tape.operations:
            gate_name = gate.name
            self.gate_mapping[gate_name](gate, params)

        return params

    def convert(
            self,
            circuit: qml.QNode
        ) -> Circuit:

        num_qubits = len(circuit.device.wires)
        quick_circuit = self.output_framework(num_qubits=num_qubits)

        # We first transpile the circuit to the minimal gateset of [u3, rz, rx, global phase]
        # This allows for support of future Pennylane gates, as well as custom ones
        # that are not native to Pennylane
        circuit = qml.transforms.decompose(
            circuit,
            gate_set=[qml.CNOT, qml.RZ, qml.RX, qml.GlobalPhase]
        )

        for param in self.extract_params(circuit):
            gate_name = param.pop("gate")
            getattr(quick_circuit, gate_name)(**param)

        return quick_circuit