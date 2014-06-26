TMPREPO=/tmp/docs/ffn

.PHONY: clean dist docs css pages serve

clean:
	rm -rf dist
	rm -rf ffn.egg-info

dist:
	python setup.py sdist upload

docs: css
	$(MAKE) -C docs/ clean
	$(MAKE) -C docs/ html

css:
	lessc --clean-css docs/source/_themes/klink/static/less/klink.less docs/source/_themes/klink/static/css/klink.css

pages: 
	rm -rf $(TMPREPO)
	git clone -b gh-pages git@github.com:pmorissette/ffn.git $(TMPREPO)
	rm -rf $(TMPREPO)/*
	cp -r docs/build/html/* $(TMPREPO)
	cd $(TMPREPO); \
	git add -A ; \
	git commit -a -m 'auto-updating docs' ; \
	git push

serve: 
	cd docs/build/html; \
	python -m SimpleHTTPServer
