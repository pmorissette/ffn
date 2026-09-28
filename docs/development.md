# How to develop ffn

Use Python 3.11 for development and documentation tools. ffn's runtime supports Python 3.9 and later. Install uv, then create an environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
make develop
make lint
make checks
make coverage
make build
```

On Windows, activate with `.venv\Scripts\activate`. Run `make help` for available targets. `make test` runs the unit tests; `make benchmark` runs the separate performance benchmarks. Type checking (`make check-types`) is advisory and does not gate CI. The `dev` extra remains an alias for `develop`.

CI uses the template's `actions-ext/python/test-wheel` and `actions-ext/python/test-sdist` actions to check distribution installation.

## Build documentation

```bash
make develop
make docs
make serve
```

Open <http://localhost:9087>. Yardang reads configuration from `pyproject.toml` and uses `README.md` as the homepage. Klink supplies the theme; generated HTML goes into `docs/html`. Builds treat warnings as errors.

Documentation dependencies live in `pyproject.toml` under the `develop` extra. Both local setup and documentation CI install that extra, including `klink>=0.1.13` and `yardang>=0.10.0`. There is no separate documentation requirements file.

`make docs` runs `yardang build --warning-is-error`, then copies `docs/source/_static` into the built site's `_static` directory. No custom Python build wrapper is needed.

Edit the MyST Markdown installation guide, quickstart, and API reference under `docs/source`. Keep API reference separate from the tutorials and contributor instructions. Existing page URLs are preserved by redirects in `[tool.yardang.redirects]`.

Keep autodoc directives inside `{eval-rst}` fences in the Markdown API reference: autodoc generates reStructuredText from Python docstrings.

The notebook Markdown exports and images are checked in. Builds use saved outputs without executing notebooks or downloading market data. After editing a notebook, regenerate its Markdown export and images without rerunning it:

```bash
uv pip install nbconvert mdformat-myst
cd docs/source
jupyter nbconvert --to markdown --NbConvertApp.output_files_dir=_static intro.ipynb quickstart.ipynb
python -m mdformat intro.md quickstart.md
```

Use Markdown cells and MyST roles such as `{py:func}` for API links in notebooks. Pandoc is not required for Markdown exports. Review and commit the changed notebooks, Markdown exports, and images together. Run `make docs` and check navigation, images, and API links before submitting changes.

Pull requests build an HTML artifact. Successful documentation builds on `master` publish to the existing `gh-pages` branch.

## Update the template

From a clean branch with development dependencies installed, run:

```bash
copier update --answers-file .copier-answers.yaml --trust
```

Resolve conflicts and review the generated diff before running the checks above. Keep ffn's MIT license, version, Python runtime floor, package metadata, top-level tests, benchmarks, and Ruff line length when accepting template changes. `.copier-answers.yaml` records the pure-Python variant of `python-project-templates/base` and the pinned template revision.

The Python Templates Copier Update GitHub App can propose updates once installed for this repository. The answers file also supports manual updates with the command above.
