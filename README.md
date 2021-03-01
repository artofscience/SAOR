# SAO

SAO framework

## Installation

To install the package for development, first clone the repository.
Then, from the project's root run `make`. This will setup a virtual
environment inside the project at `./venv`. After activating this
environment, `source ./venv/bin/active`, the `sao` package and its
dependencies are available.

The tests for `sao` are evaluated with `pytest`:

```
# minimal reporting on the test status
pytest tests/

# explicit coverage report with missing lines highlighted
pytest tests/ --cov sao --cov-report term-missing
```

## Documentation

Documentation is generated using `Sphinx`. To build and clean docs:

```
# create documentation
make docs

# clean documentation
make docsclean
```

The documentation is generated from the source at `docs/source/` and the
documentation strings throughout the Python code. The output
is placed at `docs/build/`. Open `docs/build/index.html` to browse locally.

Requirements to building the docs can be setup in the local environment by
running

```
pip install -e .[docs]
```

The extra `docs` dependencies are pulled from the `setup.py`, which installs
`sphinx`, `sphix_rtd_theme` and `sphinxcontrib-napolean` into the local
(virtual)environment.
