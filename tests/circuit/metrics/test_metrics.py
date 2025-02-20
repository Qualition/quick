# Copyright 2023-2024 Qualition Computing LLC.
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

__all__ = ["TestMetrics"]

from quick.circuit import QiskitCircuit
from quick.circuit.metrics import get_entanglements


class TestMetrics:
    """ `tests.circuit.metrics.TestMetrics` class to test the `quick.circuit.metrics`
    module.
    """
    def test_get_entanglements(self) -> None:
        """ Test the `get_entanglements` method.
        """
        qc = QiskitCircuit(6)
        qc.H(0)
        qc.CX(0, 1)

        qc.H(3)
        qc.CX(3, 4)
        qc.CX(4, 5)

        entanglements = get_entanglements(qc)
        assert entanglements == [(0, 1), (2, 2), (3, 5)]