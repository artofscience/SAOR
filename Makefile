export VENV := ./venv

.PHONY: install
install: python

.PHONY: python
python:  venv
	. $(VENV)/bin/activate && pip install -e .[dev]

venv:
	test -d $(VENV) || python3 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel

.PHONY: test
test:
	tox

.PHONY: docs
docs:
	sphinx-build -b html docs/source/ docs/build/

.PHONY: docsclean
docsclean:
	rm -rf docs/build/

.PHONY: distclean
distclean:
	rm -rf $(VENV)/
	rm -rf .tox/
	rm -rf isct.egg-info/
	rm -rf cov_html/
