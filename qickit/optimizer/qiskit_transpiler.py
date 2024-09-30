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

""" Wrapper class for using the Qiskit transpiler in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["QiskitTranspiler"]

from qiskit.transpiler import PassManager # type: ignore

from qickit.circuit import Circuit, QiskitCircuit
from qickit.optimizer.optimizer import Optimizer


class QiskitTranspiler(Optimizer):
    """ `qickit.optimizer.QiskitTranspiler` is the wrapper class for the LightSABRE optimizer
    provided by the `qiskit` library. This optimizer utilizes the `qiskit` transpiler to optimize
    the circuit.

    Notes
    -----
    The `qiskit.transpiler` library is a quantum circuit optimization library developed by IBM Quantum
    to optimize, route, and transpile quantum circuits given a set of constraints. The implementation
    utilizes LightSABRE approach.

    For more information on Qiskit transpiler:
    - Documentation:
    https://qiskit.org/documentation/apidoc/transpiler.html.
    - Source code:
    https://github.com/Qiskit/qiskit/tree/main/qiskit/transpiler
    - Publication:
    https://arxiv.org/pdf/2409.08368

    Usage
    -----
    >>> optimizer = QiskitTranspiler()
    """
    def optimize(
            self,
            circuit: Circuit
        ) -> Circuit:
        """ Optimize the given circuit

        Parameters
        ----------
        `circuit` : qickit.circuit.Circuit
            The circuit to be optimized

        Returns
        -------
        `optimized_circuit` : qickit.circuit.Circuit
            The optimized circuit
        """
        circuit_type = type(circuit)

        if not isinstance(circuit, QiskitCircuit):
            circuit = circuit.convert(QiskitCircuit)

        # TODO: Speculate the role of the pass manager,
        # and how it is used to optimize the circuit
        # Create a pass manager to optimize the circuit
        pass_manager = PassManager()

        # Apply the transpilation pass to optimize the circuit
        transpiled_circuit = pass_manager.run(circuit.circuit)

        optimized_circuit = Circuit.from_qiskit(transpiled_circuit, circuit_type)

        return optimized_circuit