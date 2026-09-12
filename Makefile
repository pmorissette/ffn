.DEFAULT_GOAL := help
.PHONY: develop requirements build install lint-py lint-docs fix-py fix-docs lint lints fix format check-dist check-types checks check test tests coverage benchmark show-version patch minor major dist dist-build dist-check test-dist publish upload docs docs-develop serve notebooks clean help

develop:  ## install development dependencies and library
	uv pip install -e '.[develop]'

requirements:  ## install prerequisite Python build requirements
	uv pip install -r pyproject.toml --extra develop

build:  ## build the Python library
	python -m build -n

install:  ## install library
	uv pip install .

lint-py:  ## lint Python with ruff
	python -m ruff check ffn .github/scripts docs/build.py
	python -m ruff format --check ffn .github/scripts docs/build.py

lint-docs:  ## lint contributor documentation
	python -m mdformat --check README.md docs/development.md docs/source/*.md
	python -m codespell_lib README.md docs/development.md docs/source/*.md

fix-py:  ## autoformat Python code
	python -m ruff check --fix ffn .github/scripts docs/build.py
	python -m ruff format ffn .github/scripts docs/build.py

fix-docs:  ## autoformat contributor documentation
	python -m mdformat README.md docs/development.md docs/source/*.md
	python -m codespell_lib --write README.md docs/development.md docs/source/*.md

lint: lint-py lint-docs  ## run all linters
lints: lint
fix: fix-py fix-docs  ## run all autoformatters
format: fix

check-dist:  ## check sdist and wheel contents
	check-dist -v --rebuild

check-types:  ## check Python types (advisory)
	ty check ffn

checks: check-dist  ## run distribution checks
check: checks

test:  ## run Python tests
	python -m pytest tests

tests: test

coverage:  ## run tests with coverage
	python -m pytest tests --cov=ffn --cov-report term-missing --cov-report xml

benchmark:  ## run performance benchmarks
	python -m pytest benchmarks --benchmark-only

show-version:  ## show current library version
	@bump-my-version show current_version

patch:  ## bump a patch version
	@bump-my-version bump patch

minor:  ## bump a minor version
	@bump-my-version bump minor

major:  ## bump a major version
	@bump-my-version bump major

dist-build:  ## build Python distributions
	python -m build -w -s

dist-check:  ## check distribution metadata
	python -m twine check dist/*

test-dist:  ## test installed wheels and source distributions
	python .github/scripts/test-distributions.py

dist:  ## build and check distributions
	$(MAKE) clean
	$(MAKE) dist-build
	$(MAKE) dist-check

publish: dist

upload: dist  ## upload distributions to PyPI
	python -m twine upload dist/* --skip-existing

docs-develop:  ## install documentation dependencies
	uv pip install -e '.[develop]' -r docs/requirements.txt

docs:  ## build documentation with Yardang and Klink
	python docs/build.py

serve:  ## serve built documentation on port 9087
	python -m http.server 9087 --directory docs/html

notebooks:  ## edit documentation notebooks (requires Jupyter)
	cd docs/source && jupyter notebook --no-browser

clean:  ## remove distribution build output
	rm -rf build dist ffn.egg-info

help:
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "%-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
