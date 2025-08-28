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

""" Contraction logic for `quick.primitives` classes.
"""

from __future__ import annotations

__all__ = ["contract"]

import numpy as np
from numpy.typing import NDArray
from quick.primitives import Statevector, Operator


def _einsum_tensor_contract(
        tensor: NDArray[np.complex128],
        op: Operator,
        contract_indices: list[int]
    ) -> NDArray[np.complex128]:
    """ Perform tensor contraction using Einstein's summation convention.

    Notes
    -----
    The implementation is based on LSB convention.

    Parameters
    ----------
    `tensor` : NDArray[np.complex128]
        The tensor the operator is being contracted with.
    `op` : quick.primitives.Operator
        The operator to contract with the tensor.
    `contract_indices` : list[int]
        The indices to contract over.

    Returns
    -------
    NDArray[np.complex128]
        The result of the tensor contraction.
    """
    rank = tensor.ndim

    # We tag the tensor indices that will be contracted with respect to
    # the order in contract indices
    tensor_indices = list(range(rank))
    for i, index in enumerate(contract_indices):
        tensor_indices[index] = rank + i

    # Since the order is already taken into account in tensor indices, we
    # only need to ensure the indices are present in a constant order
    # regardless of different permutations
    # The reason we reverse the order is because the gates themselves
    # are in LSB convention as well
    op_contract_indices = list(
        range(rank + len(contract_indices) - 1, rank - 1, -1)
    )

    # The reason we reverse the order is because the gates themselves
    # are in LSB convention as well
    op_free_indices = list(reversed(contract_indices))

    op_indices = op_free_indices + op_contract_indices

    # Reshape the passed operator `op` to be a tensor with 2M indices
    # of dimension 2 to represent the qubits, resulting in a
    # rank-2M tensor where M <= N and 2N is the rank of `tensor`
    op_tensor = np.reshape(op.data, op.tensor_shape)

    return np.einsum(tensor, tensor_indices, op_tensor, op_indices) # type: ignore

def contract(
        tensor: Statevector | Operator,
        op: NDArray[np.complex128] | Operator,
        qubit_indices: list[int]
    ) -> None:
    """ Contract the operator with an operator on the specified
    qubits in place using Einstein's Summation convention.

    Parameters
    ----------
    `tensor` : quick.primitives.Statevector | quick.primitives.Operator
        The tensor to apply `op` to.
    `op` : NDArray[np.complex128] | quick.primitives.Operator
        The operator or matrix to contract with.
    `qubit_indices` : list[int]
        The qubit indices to contract over.

    Raises
    ------
    ValueError
        ValueError
        - If the operator is not unitary.
        - If the number of indices is less than the number of qubits for `op`.
        - If the number of qubit indices exceeds the number of qubits in `tensor`.
        - If any of the qubit indices are out of range of `self`.

    Usage
    -----
    >>> tensor.contract(op, [0, 1])
    """
    op_tensor = Operator(np.array(op))

    num_indices = len(qubit_indices)

    if op_tensor.num_qubits != num_indices:
        raise ValueError(
            f"Operator requires {op_tensor.num_qubits} qubits. ",
            f"Received {num_indices} instead."
        )

    if num_indices > tensor.num_qubits:
        raise ValueError(
            f"{type(tensor).__name__} supports operators with at most {tensor.num_qubits} qubits."
            f"Received an operator with {num_indices} instead."
        )

    if any(i >= tensor.num_qubits for i in qubit_indices):
        raise ValueError(
            f"Invalid qubit index in {qubit_indices}. "
            f"Valid indices are in range(0, {tensor.num_qubits})."
        )

    # Modify the indices to be MSB for correct alignment given how numpy does broadcasting
    op_contract_indices = [tensor.num_qubits - 1 - i for i in qubit_indices]

    # Reshape the current operator to be a tensor with X indices
    # of dimension 2 to represent the qubits, resulting in a
    # rank-X tensor
    # For statevectors X is the number of qubits N whereas for operators
    # X is 2N
    reshaped_tensor = np.reshape(tensor.data, tensor.tensor_shape)

    tensor.data = np.reshape(
        _einsum_tensor_contract(reshaped_tensor, op_tensor, op_contract_indices),
        tensor.shape
    )