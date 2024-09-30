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

""" Wrapper class for using the optimizer provided by the `tket2` library in Qickit SDK.
"""

from __future__ import annotations

__all__ = ["TKET2Optimizer"]

from tket2.passes import badger_pass

from qickit.circuit import Circuit, TKETCircuit
from qickit.optimizer.optimizer import Optimizer


class TKET2Optimizer(Optimizer):
    """ `qickit.optimizer.TKET2Optimizer` is the wrapper class for the optimizer provided by
    the `tket2` library. This optimizer utilizes the `tket2` rewrite rules to optimize the
    circuit.

    Notes
    -----
    The `tket2` library is a quantum circuit optimization library developed by Cambridge Quantum
    Computing. The library is written in Rust and provides a Python interface, providing a faster
    and more efficient optimization compared to the default optimizer provided by `tket`.

    The Badger compiler pass optimizes the circuit by applying multiple rewrite rules simultaneously
    and searching for the best sequence. Though computationally expensive, this process can significantly
    simplify large circuits.

    For more information, see https://github.com/CQCL/tket2.

    Usage
    -----
    >>> optimizer = TKET2Optimizer()
    """
    def optimize(
            self,
            circuit: Circuit
        ) -> Circuit:
        """ Optimize the given circuit using the `tket2` optimizer.

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

        if not isinstance(circuit, TKETCircuit):
            circuit = circuit.convert(TKETCircuit)

        # Apply the Badger compiler pass to optimize the circuit
        badger_pass().apply(circuit.circuit)

        optimized_circuit = Circuit.from_tket(circuit.circuit, circuit_type)

        return optimized_circuit