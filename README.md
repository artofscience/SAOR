# SAOR

The Sequential Approximate Optimization Repository, or SAOR, provides an concise
implementation of multiple sequential approximate optimization routines. The
package aims to provide a simple, modular implementation that enables users to
investigate and modify the optimization routines to match their optimization
problems.

SAOR provides multiple well-known algorithms, e.g. *Optimality Criteria*,
[*ConLin*](https://mecheng.iisc.ac.in/suresh/me256/conlin.pdf), and the
[*Method of Moving Asymptotes (MMA)*](https://fab.cba.mit.edu/classes/865.18/design/optimization/mma.pdf).
Although these are implemented in a modular fashion, simple wrappers are
provided to use a standard implementation with corresponding default settings.

## Usage

A problem can be setup using predefined wrappers

```python
problem = Square()
x, f = sao.solvers.method_of_moving_asymptoptes.mma(problem, x0=x0)
print(f'Final design: {x} with objective value: {f}')
```

Or by constructing the full optimization strategy

```python
problem = Square()
x = problem.x0
intervening_variables = sao.intervening_variables.MMA()
approximation = sao.approximation.Taylor1(intervening_variables)
sub_problem = sao.problems.Subproblem(problem)
criterion = sao.convergence_criteria.VariableChange(x, tolerance=1e-2)

while not criterion.converged:
    f = problem.g(x)
    df = problem.dg(x)
    sub_problem.build(x, f, df)
    x[:] = sao.solvers.primal_dual_interior_point.pdip(sub_problem)

print(f'Final design: {x} with objective value: {f}')
```

Many alternative variations are illustrated in
[`examples/optimization_problem_setup.py`](https://github.com/artofscience/SAOR/blob/main/examples/optimization_problem_setup.py).


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
