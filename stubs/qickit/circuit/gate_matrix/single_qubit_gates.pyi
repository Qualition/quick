from qickit.circuit.gate_matrix import Gate

__all__ = [
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "S",
    "T",
    "RX",
    "RY",
    "RZ",
    "U3",
    "Phase"
]

class PauliX(Gate):
    def __init__(self) -> None: ...

class PauliY(Gate):
    def __init__(self) -> None: ...

class PauliZ(Gate):
    def __init__(self) -> None: ...

class Hadamard(Gate):
    def __init__(self) -> None: ...

class S(Gate):
    def __init__(self) -> None: ...

class T(Gate):
    def __init__(self) -> None: ...

class RX(Gate):
    def __init__(self, theta: float) -> None: ...

class RY(Gate):
    def __init__(self, theta: float) -> None: ...

class RZ(Gate):
    def __init__(self, theta: float) -> None: ...

class U3(Gate):
    def __init__(self, theta: float, phi: float, lam: float) -> None: ...

class Phase(Gate):
    def __init__(self, theta: float) -> None: ...
