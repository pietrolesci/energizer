sources = energizer

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix
	torchfix $(sources) --fix

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
	rm -rf site

serve_docs:
	mkdocs serve --watch .

clean-poetry-cache:
	rm -rf ~/.cache/pypoetry/virtualenvs/
