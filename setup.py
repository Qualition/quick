import codecs
import os
from setuptools import setup, find_packages # type: ignore


here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.0rc'
DESCRIPTION = 'Framework-agnostic quantum circuit library.'
LONG_DESCRIPTION = '`qickit` is an agnostic gate-based circuit SDK, providing an integrated interface for using any supported quantum circuit framework seamlessly.'

# Setting up
setup(
    name="qickit",
    version=VERSION,
    author="Amir Ali Malekani Nezhad",
    author_email="<amiralimlk07@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    keywords=[
        "quantum computing", "quantum circuit", "quantum compiler"
    ],
    python_requires=">=3.10,<=3.12",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers"
    ],
    project_urls = {
        "Bug Tracker": "https://github.com/qualition/qickit/issues",
    }
)