TMPREPO=/tmp/docs/ffn

.PHONY: clean dist docs pages serve notebooks klink test lint fix develop

develop:
	python -m pip install -e .[dev]

test:
	python -m pytest -vvv tests --cov=ffn --junitxml=python_junit.xml --cov-report=xml --cov-branch --cov-report term

lint:
	python -m flake8 ffn setup.py docs/source/conf.py

fix:
	python -m black ffn setup.py docs/source/conf.py

clean:
	- rm -rf dist
	- rm -rf ffn.egg-info

dist:
	python setup.py sdist

upload: clean dist
	twine upload dist/*

docs: 
	$(MAKE) -C docs/ clean
	$(MAKE) -C docs/ html

pages: 
	rm -rf $(TMPREPO)
	git clone -b gh-pages git@github.com:pmorissette/ffn.git $(TMPREPO)
	rm -rf $(TMPREPO)/*
	cp -r docs/build/html/* $(TMPREPO)
	cd $(TMPREPO);\
	git add -A;\
	git commit -a -m 'auto-updating docs';\
	git push;\

serve: 
	cd docs/build/html; \
	python -m http.server 9087

notebooks:
	cd docs/source; \
	jupyter notebook --no-browser --ip=*

klink:
	git subtree pull --prefix=docs/source/_themes/klink --squash klink master
