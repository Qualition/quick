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

""" Converter for quantum circuits from TKET to Qickit.
"""

from __future__ import annotations

__all__ = ["FromTKET"]

import numpy as np
from pytket import Circuit as TKCircuit
from pytket._tket.circuit import Command
from pytket import OpType
from pytket.passes import AutoRebase
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.circuit.from_framework import FromFramework


class FromTKET(FromFramework):
    """ `qickit.circuit.from_framework.FromTKET` is a class for converting quantum circuits from
    TKET to `qickit.circuit.Circuit` class.

    Notes
    -----
    The conversion is done by first transpiling the circuit to u3 and cx gates, and then extracting
    the parameters of the gates in the TKET circuit. We perform transpilation to the minimal gateset
    of [u3, cx, global phase] to allow for support of future TKET gates, as well as custom ones that
    are not native to TKET.

    This is done to ensure that the conversion is as general as possible without having to update the
    converter for every new gate that is added to TKET, or for gates that are not currently implemented
    in Qickit.

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
        The mapping of the gate names between TKET and Qickit.

    Raises
    ------
    `TypeError`
        - If the `output_framework` is not a subclass of `qickit.circuit.Circuit`.

    Usage
    -----
    >>> tket_converter = FromTKET(output_framework=CirqCircuit)
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:

        super().__init__(output_framework=output_framework)

        self.gate_mapping = {
            "OpType.U3": self._extract_u3_gate_params,
            "OpType.CX": self._extract_cx_gate_params,
            "OpType.Measure": self._extract_measure_gate_params
        }

    def _extract_u3_gate_params(
            self,
            gate: Command,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a U3 gate.

        Parameters
        ----------
        `gate` : pytket._tket.circuit.Command
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "U3",
            "qubit_indices": gate.qubits[0].index[0],
            "angles": [float(param)*np.pi for param in gate.op.params]
        })

    def _extract_cx_gate_params(
            self,
            gate: Command,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a CX gate.

        Parameters
        ----------
        `gate` : pytket._tket.circuit.Command
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "CX",
            "control_index": gate.qubits[0].index[0],
            "target_index": gate.qubits[1].index[0]
        })

    def _extract_measure_gate_params(
            self,
            gate: Command,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a measurement gate.

        Parameters
        ----------
        `gate` : pytket._tket.circuit.Command
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "measure",
            "qubit_indices": gate.qubits[0].index[0]
        })

    def extract_params(
            self,
            circuit: TKCircuit,
        ) -> list[dict]:
        """ Extract the parameters of a gate.

        Parameters
        ----------
        `circuit` : pytket.Circuit
            The quantum circuit to extract the parameters from.

        Returns
        -------
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params: list[dict] = []

        # Iterate over the gates in the TKET circuit
        for gate in circuit:
            gate_name = str(gate.op.type)

            self.gate_mapping[gate_name](gate, params)

        return params

    def convert(
            self,
            circuit: TKCircuit,
        ) -> Circuit:

        # Define a circuit
        num_qubits = circuit.n_qubits
        qickit_circuit = self.output_framework(num_qubits=num_qubits)

        # Transpile the TKET circuit to u3 and cx gates
        tket_pass = AutoRebase({OpType.U3, OpType.CX})
        tket_pass.apply(circuit)

        # Extract the parameters of the gates in the TKET circuit
        params = self.extract_params(circuit)

        # Add the gates to the Qickit circuit
        for param in params:
            gate_name = param.pop("gate")
            getattr(qickit_circuit, gate_name)(**param)

        qickit_circuit.GlobalPhase(float(circuit.phase) * np.pi)

        return qickit_circuit