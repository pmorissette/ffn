TMPREPO=/tmp/docs/ffn

.PHONY: clean dist docs pages serve notebooks

clean:
	- rm -rf dist
	- rm -rf ffn.egg-info

dist:
	python setup.py sdist

upload: dist
	twine upload dist/*

docs: 
	$(MAKE) -C docs/ clean
	$(MAKE) -C docs/ html

pages: 
	- rm -rf $(TMPREPO)
	git clone -b gh-pages git@github.com:pmorissette/ffn.git $(TMPREPO)
	rm -rf $(TMPREPO)/*
	cp -r docs/build/html/* $(TMPREPO)
	cd $(TMPREPO); \
	git add -A ; \
	git commit -a -m 'auto-updating docs' ; \
	git push

serve: 
	cd docs/build/html; \
	python -m SimpleHTTPServer 9087

notebooks:
	cd docs/source; \
	ipython notebook --no-browser --ip=*
