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

""" Converter for quantum circuits from Cirq to Qickit.
"""

from __future__ import annotations

__all__ = ["FromCirq"]

import cirq # type: ignore
import numpy as np
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from qickit.circuit import Circuit
from qickit.circuit.from_framework import FromFramework


class FromCirq(FromFramework):
    """ `qickit.circuit.from_framework.FromCirq` is a class for converting quantum circuits from
    Cirq to `qickit.circuit.Circuit` class.

    Notes
    -----
    The conversion is done by first transpiling the circuit to PhasedXZ and CZ gates, and then extracting
    the parameters of the gates in the Qiskit circuit. We perform transpilation to the minimal gateset
    of [PhasedXZ, CZ] to allow for support of future Cirq gates, as well as custom ones that are not
    native to Cirq. Note that PhasedXZ is comprised of RX, RZ, and global phase gates.

    PhasedXZ(x, y, z) = RZ(-z*pi) RX(x*pi) RZ((z+y)*pi) GlobalPhase(-z*pi/2 + x*pi/2 + (z+y)*pi/2)

    This is analogous to transpiling to U3, CX, and GlobalPhase gates.

    This is done to ensure that the conversion is as general as possible without having to update the
    converter for every new gate that is added to Cirq, or for gates that are not currently implemented
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
        The mapping of the gates in Qiskit to the gates in Qickit.

    Raises
    ------
    `TypeError`
        - If the `output_framework` is not a subclass of `qickit.circuit.Circuit`.

    Usage
    -----
    >>> cirq_converter = FromCirq(output_framework=QiskitCircuit)
    """
    def __init__(
            self,
            output_framework: Type[Circuit]
        ) -> None:

        super().__init__(output_framework=output_framework)

        self.gate_mapping = {
            "CZPowGate": self._extract_cz_gate_params,
            "PhasedXZGate": self._extract_phasedzx_gate_params,
            "MeasurementGate": self._extract_measure_gate_params
        }

    def _extract_phasedzx_gate_params(
            self,
            gate: cirq.ops.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a PhasedZX gate.

        Parameters
        ----------
        `gate` : cirq.ops.Operation
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        gate_params = gate.gate._json_dict_() # type: ignore

        # Decompose PhasedXZGate(x, y, z) to RZ(-z*pi) RX(x*pi) RZ((y+z)*pi)
        params.append({
            "gate": "RZ",
            "qubit_indices": gate.qubits[0].x, # type: ignore
            "angle": -gate_params["axis_phase_exponent"] * np.pi
        })
        params.append({
            "gate": "RX",
            "qubit_indices": gate.qubits[0].x, # type: ignore
            "angle": gate_params["x_exponent"] * np.pi
        })
        params.append({
            "gate": "RZ",
            "qubit_indices": gate.qubits[0].x, # type: ignore
            "angle": (gate_params["axis_phase_exponent"] + gate_params["z_exponent"]) * np.pi
        })
        params.append({
            "gate": "GlobalPhase",
            "angle": (gate_params["x_exponent"] + gate_params["z_exponent"]) * np.pi/2
        })

    def _extract_cz_gate_params(
            self,
            gate: cirq.ops.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a CZ gate.

        Parameters
        ----------
        `gate` : cirq.ops.Operation
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "CZ",
            "control_index": gate.qubits[0].x, # type: ignore
            "target_index": gate.qubits[1].x # type: ignore
        })

    def _extract_measure_gate_params(
            self,
            gate: cirq.ops.Operation,
            params: list[dict]
        ) -> None:
        """ Extract the parameters of a measurement gate.

        Parameters
        ----------
        `gate` : cirq.ops.Operation
            The quantum gate to extract the parameters from.
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params.append({
            "gate": "measure",
            "qubit_indices": [qubit.x for qubit in gate.qubits] # type: ignore
        })

    def extract_params(
            self,
            circuit: cirq.Circuit,
        ) -> list[dict]:
        """ Extract the parameters of a gate.

        Parameters
        ----------
        `circuit` : cirq.Circuit
            The quantum circuit to extract the parameters from.

        Returns
        -------
        `params` : list[dict]
            The list of parameters for mapping the gates to Qickit.
        """
        params: list[dict] = []

        # Define the list of all circuit operations
        ops = list(circuit.all_operations())

        # Iterate over the operations in the Cirq circuit
        for operation in ops:
            op_name = type(operation.gate).__name__

            self.gate_mapping[op_name](operation, params)

        return params

    def convert(
            self,
            circuit: cirq.Circuit,
        ) -> Circuit:

        # Define a circuit
        num_qubits = len(circuit.all_qubits())
        qickit_circuit = self.output_framework(num_qubits=num_qubits)

        # Transpile the circuit to PhasedXZ and CZ gates
        circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq.CZTargetGateset())

        # Extract the parameters of the gates in the Qiskit circuit
        params = self.extract_params(circuit)

        # Add the gates to the Qickit circuit
        for param in params:
            gate_name = param.pop("gate")
            getattr(qickit_circuit, gate_name)(**param)

        return qickit_circuit