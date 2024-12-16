from qickit.circuit import Circuit
from qickit.circuit.from_framework import FromFramework
from qiskit import QuantumCircuit
from typing import Callable

__all__ = ["FromQiskit"]

class FromQiskit(FromFramework):
    gate_mapping: dict[str, Callable]
    skip_gates: list[str]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    def extract_params(self, circuit: QuantumCircuit) -> list[dict]: ...
    def convert(self, circuit: QuantumCircuit) -> Circuit: ...
