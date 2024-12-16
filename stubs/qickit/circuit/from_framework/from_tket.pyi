from pytket import Circuit as TKCircuit
from qickit.circuit import Circuit
from qickit.circuit.from_framework import FromFramework
from typing import Callable

__all__ = ["FromTKET"]

class FromTKET(FromFramework):
    gate_mapping: dict[str, Callable]
    def __init__(self, output_framework: type[Circuit]) -> None: ...
    def extract_params(self, circuit: TKCircuit) -> list[dict]: ...
    def convert(self, circuit: TKCircuit) -> Circuit: ...
