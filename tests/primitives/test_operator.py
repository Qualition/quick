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

from __future__ import annotations

__all__ = ["TestOperator"]

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from quick.primitives import Statevector, Operator


class TestOperator:
    """ `tests.primitives.test_operator.TestOperator` is the tester class
    for `quick.primitives.Operator`.
    """
    def test_init(self) -> None:
        """ Test the initialization of the `quick.primitives.Operator` class.
        """
        operator = Operator(
            np.array([
                [1, 0],
                [0, 1]
            ]), label="A"
        )
        assert_almost_equal(operator.data, np.array([[1+0j, 0+0j], [0+0j, 1+0j]]))
        assert operator.shape == (2, 2)
        assert operator.num_qubits == 1
        assert operator.label == "A"

    def test_from_scalar_fail(self) -> None:
        """ Test the failure of defining a `quick.primitives.Operator` object from a scalar.
        """
        with pytest.raises(ValueError):
            Operator(1) # type: ignore

    def test_from_statevector_fail(self) -> None:
        """ Test the failure of defining a `quick.primitives.Operator` object from a statevector.
        """
        with pytest.raises(ValueError):
            Operator(np.array([1, 0, 0, 0]))

    def test_reverse_bits(self) -> None:
        """ Test the MSB to LSB (vice versa) conversion of the `quick.primitives.Operator` object.
        """
        cx_msb = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        cx_lsb = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ])
        operator = Operator(cx_msb)
        operator.reverse_bits()
        checker_operator = Operator(cx_lsb)
        assert_almost_equal(operator.data, checker_operator.data)

        operator.reverse_bits()
        checker_operator = Operator(cx_msb)
        assert_almost_equal(operator.data, checker_operator.data)

    def test_check_mul(self) -> None:
        """ Test the multiplication of two `quick.primitives.Operator` objects.
        """
        op1 = Operator(np.array([[1, 0], [0, 1]]))
        op2 = Operator(np.array([[0, 1], [1, 0]]))
        state = Statevector(np.array([1, 0]))
        op1._check__mul__(op2)
        op1._check__mul__(state)

    def test_check_mul_fail(self) -> None:
        """ Test the failure of the multiplication of two `quick.primitives.Operator` objects.
        """
        op1 = Operator(np.array([[1, 0], [0, 1]]))
        with pytest.raises(ValueError):
            # Mismatched number of qubits
            op1._check__mul__(Statevector(np.array([1, 0, 0, 0])))

        with pytest.raises(TypeError):
            # Incompatible type
            op1._check__mul__("not an operator or statevector or scalar")

    def test_eq(self) -> None:
        """ Test the equality of two `quick.primitives.Operator` objects.
        """
        op1 = Operator(np.array([[1, 0], [0, 1]]))
        op2 = Operator(np.array([[1, 0], [0, 1]]))

        assert op1 == op2

    def test_eq_fail(self) -> None:
        """ Test the failure of the equality of two `quick.primitives.Operator` objects.
        """
        op1 = Operator(np.array([[1, 0], [0, 1]]))
        op2 = Operator(np.array([[0, 1], [1, 0]]))
        assert op1 != op2

        with pytest.raises(TypeError):
            # Incompatible type
            op1 == "not an operator" # type: ignore

    def test_mul_statevector(self) -> None:
        """ Test the multiplication of a `quick.primitives.Operator` with a `quick.primitives.Statevector`.
        """
        operator = Operator(np.array([[1, 0], [0, 1]]))
        state = Statevector(np.array([1, 0]))
        result = operator * state
        assert_almost_equal(result.data, np.array([1, 0]))

    def test_mul_operator(self) -> None:
        """ Test the multiplication of two `quick.primitives.Operator` objects.
        """
        op1 = Operator(np.array([[1, 0], [0, 1]]))
        op2 = Operator(np.array([[0, 1], [1, 0]]))
        result = op1 * op2
        assert_almost_equal(result.data, np.array([[0, 1], [1, 0]]))

    def test_mul_fail(self) -> None:
        """ Test the failure of the multiplication of a `quick.primitives.Operator` with an incompatible type.
        """
        operator = Operator(np.array([[1, 0], [0, 1]]))

        with pytest.raises(ValueError):
            # Incompatible dimensions
            operator * Statevector(np.array([1, 0, 0, 0])) # type: ignore

        with pytest.raises(TypeError):
            # Incompatible type
            operator * "not an operator or statevector" # type: ignore

    def test_matmul(self) -> None:
        """ Test the tensor product operation of the `quick.primitives.Operator` object.
        """
        op1 = Operator(np.array([
            [1, 0],
            [0, 1]
        ]))
        op2 = Operator(np.array([
            [0, 1],
            [1, 0]
        ]))

        op3 = op1 @ op2
        op3_checker = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        assert_almost_equal(op3.data, op3_checker)

        op4 = op2 @ op1
        op4_checker = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        assert_almost_equal(op4.data, op4_checker)

    def test_str(self) -> None:
        """ Test the string representation of the `quick.primitives.Operator` object.
        """
        operator = Operator(np.array([[1, 0], [0, 1]]), label="Identity")
        assert str(operator) == "Identity"

    def test_repr(self) -> None:
        """ Test the string representation of the `quick.primitives.Operator` object.
        """
        operator = Operator(np.array([[1, 0], [0, 1]]), label="Identity")
        assert repr(operator) == "Operator(data=[[1 0]\n [0 1]], label=Identity)"