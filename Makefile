.PHONY: clean-pyc clean-build docs clean

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "sdist - package"

clean: clean-build clean-pyc
	rm -fr htmlcov/

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -fr tmp*

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	flake8 lib/astunparse tests

test:
	tox

coverage:
	tox -e coverage

docs:
	tox -e docs

dist: clean install-build-deps
	python -m build
	ls -l dist

PIP ?= pip
install-deps:
	$(PIP) install -r requirements.txt

install-test-deps:
	$(PIP) install -r test_requirements.txt

install: install-deps
	$(PIP) install .
