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

from quick.predicates.predicates import (
    is_diagonal_matrix as is_diagonal_matrix,
    is_hermitian_matrix as is_hermitian_matrix,
    is_identity_matrix as is_identity_matrix,
    is_isometry as is_isometry,
    is_positive_semidefinite_matrix as is_positive_semidefinite_matrix,
    is_square_matrix as is_square_matrix,
    is_symmetric_matrix as is_symmetric_matrix,
    is_unitary_matrix as is_unitary_matrix
)

__all__ = [
    "is_square_matrix",
    "is_diagonal_matrix",
    "is_symmetric_matrix",
    "is_identity_matrix",
    "is_unitary_matrix",
    "is_hermitian_matrix",
    "is_positive_semidefinite_matrix",
    "is_isometry"
]
