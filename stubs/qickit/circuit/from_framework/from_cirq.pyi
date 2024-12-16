import cirq
from qickit.circuit import Circuit
from qickit.circuit.from_framework import FromFramework
from typing import Callable

__all__ = ["FromCirq"]

class FromCirq(FromFramework):
    gate_mapping: dict[str, Callable]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    def extract_params(self, circuit: cirq.Circuit) -> list[dict]: ...
    def convert(self, circuit: cirq.Circuit) -> Circuit: ...
