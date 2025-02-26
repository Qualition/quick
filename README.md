# Quick

[![PyPI version](https://img.shields.io/pypi/v/quick-core)](//pypi.org/project/quick-core)
[![License](https://img.shields.io/github/license/Qualition/quick.svg?)](https://opensource.org/licenses/Apache-2.0) <!--- long-description-skip-begin -->
[![Tests](https://github.com/Qualition/quick/actions/workflows/tests.yml/badge.svg)](https://github.com/qualition/quick/actions/workflows/tests.yml)
[![codecov](https://codecov.io/github/Qualition/quick/branch/main/graph/badge.svg?token=IHWJZG8VJT)](https://codecov.io/github/Qualition/quick)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e287a2eed9e24d5e9d4a3ffe911ce6a5)](https://app.codacy.com?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![Quick Logo (Light Mode)](https://github.com/user-attachments/assets/f49e2684-7cff-4ced-807c-add616bb6527#gh-light-mode-only)

![Quick Logo (Dark Mode)](https://github.com/user-attachments/assets/8ac68d33-3be2-4620-8160-e0d8a5888813#gh-dark-mode-only)

`quick` is an agnostic gate-based circuit SDK, providing an integrated interface for using any supported quantum circuit framework seamlessly, and provides a complete pipeline from construction of circuits to mapping on actual hardware.

## Getting Started

### Prerequisites

- python 3.10, 3.11, 3.12
- Ubuntu

Currently, due to this [issue](https://github.com/Qualition/quick/issues/11) `quick` works reliably only on Ubuntu.

### Quick Installation

`quick` can be installed with the command:

```
pip install quick-core
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source:

```
pip install git+https://github.com/Qualition/quick
```

## Usage

The docs/examples are a good way for understanding how `quick` works. Depending on your preference, you may use the package as end-to-end, or use it in parts for low-level modifications.

The `/notebooks` directory contains pedagogical material for utilizing `quick`:

- [`Creating and Manipulating Circuits`](https://github.com/Qualition/quick/blob/main/notebooks/Creating%20and%20Manipulating%20Circuits.ipynb)
: This notebook demonstrates the basics of creating quantum circuits, and how different existing frameworks have been integrated within `quick`.
- [`Preparing Quantum States and Operators`](https://github.com/Qualition/quick/blob/main/notebooks/Preparing%20Quantum%20States%20and%20Operators.ipynb)
: This notebook demonstrates how to prepare arbitrary statevectors and unitary operators to quantum circuits using exact encoding schema.
- [`Running Quantum Circuits`](https://github.com/Qualition/quick/blob/main/notebooks/Running%20Quantum%20Circuits.ipynb)
: This notebook demonstrates the different backends available for running circuits, ranging from simulators to QPUs.

## Testing

Run tests with the command:

```
pytest tests
```

To run all tests including slow ones, use:

```
pytest tests --runslow
```

## Linting

Run lint checks with the commands:

```
mypy quick
ruff check quick
```

## Contribution Guidelines

If you'd like to contribute to `quick`, please take a look at our [`contribution guidelines`](CONTRIBUTING). By participating, you are expected to uphold our code of conduct.

We use [`GitHub issues`](https://github.com/Qualition/quick/issues) for tracking requests and bugs.

## Citation

If you wish to attribute/distribute our work, please cite the accompanying paper:
```
@article{malekaninezhad2024quick,
   title={quick: {A} {H}igh-{L}evel {P}ython {L}ibrary for {I}ntegrating {Q}uantum {G}ate-based {F}rameworks},
   author={Amir Ali Malekani Nezhad},
   year={2024},
   journal={arXiv preprint arXiv:TBD},
}
```

## License

Distributed under Apache v2.0 License. See [`LICENSE`](LICENSE) for details.
