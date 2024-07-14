import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
from qickit.backend import Backend
from qickit.primitives import Bra, Ket, Operator
from qickit.synthesis.statepreparation import StatePreparation
from qickit.synthesis.unitarypreparation import UnitaryPreparation

__all__ = ['QuantumExpression']

class QuantumExpression:
    backend: Incomplete
    state_preparation_method: Incomplete
    unitary_preparation_method: Incomplete
    def __init__(self, expression: list[Bra | Ket | Operator], backend: Backend, state_preparation_method: type[StatePreparation] | None = None, unitary_preparation_method: type[UnitaryPreparation] | None = None) -> None: ...
    expression: Incomplete
    def check_expression(self, expression: list[Bra | Ket | Operator]) -> None: ...
    def evaluate(self) -> NDArray[np.complex128]: ...
