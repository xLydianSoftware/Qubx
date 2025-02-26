set windows-shell := ["C:\\Program Files\\Git\\bin\\sh.exe", "-c"]

# publish *FLAGS:
# git push origin master --follow-tags

help:
	@just --list --unsorted


style-check:
	./check_style.sh


test:
	poetry run pytest -m "not integration" -v -n auto


test-verbose:
	poetry run pytest -m "not integration" -v -s


test-integration:
	poetry run pytest -m integration --env=.env.integration


test-ci:
	mkdir -p reports
	poetry run pytest -m "not integration" -v -ra --cov=src --cov-report=xml:reports/coverage.xml --cov-report=term -n auto


build:
	rm -rf build
	find src -type f -name *.pyd -exec  rm {} \;
	poetry build


build-fast:
	# Skip Cython compilation if binaries exist by setting PYO3_ONLY=true
	PYO3_ONLY=true poetry build


dev-install:
	poetry lock --no-update || true
	poetry install --with dev


update-docs:
	./update_docs.sh

publish: build test
	@if [ "$(git symbolic-ref --short -q HEAD)" = "main" ]; then rm -rf dist && rm -rf build && poetry build && twine upload dist/*; else echo ">>> Not in master branch !"; fi


dev-publish: build
	@rm -rf dist && rm -rf build && poetry build && twine upload dist/*
