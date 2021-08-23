from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

_sao_packages = find_packages(include=['sao/'])

# additional requirements besides `install_requires` for development
_dev = ['yapf', 'flake8', 'tox', 'pytest', 'pytest-cov']

# additional requirements to generate package documentation
_docs = ['sphinx', 'sphinx_rtd_theme', 'sphinxcontrib-napoleon']

# combined dependencies
_all = list(set(_dev) | set(_docs))

setup(
    name="sao",
    version="0.0.1",
    description='Sequential Approximate Optimisation',
    author_email='s.koppen@tudelft.nl',
    packages=_sao_packages,
    keywords=['sao', 'optimisation'],
    python_requires='>=3.8, <4',
    install_requires=[
        'numpy',
        'scipy',
        'cvxopt',
    ],
    extras_require={
        'all': _all,
        'dev': _dev,
        'doc': _docs,
    },
)
