TMPREPO=/tmp/docs/ffn

.PHONY: docs

dist:
	python setup.py publish

docs:
	rm -rf $(TMPREPO)
	git clone -b gh-pages git@github.com:pmorissette/ffn.git $(TMPREPO)
	rm -rf $(TMPREPO)/*
	$(MAKE) -C docs/ clean
	$(MAKE) -C docs/ html
	cp -r docs/build/html/* $(TMPREPO)
	cd $(TMPREPO)
	git add -A
	git commit -a -m 'auto-updating docs'
	git push
