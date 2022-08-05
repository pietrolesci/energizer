sources = energizer

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources) tests examples
	black $(sources) tests examples
	nbqa isort examples
	nbqa black examples

lint:
	flake8 $(sources) tests examples
	mypy $(sources) tests examples

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
	rm -rf */lightning_logs/
