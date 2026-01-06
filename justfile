set windows-shell := ["C:\\Program Files\\Git\\bin\\sh.exe", "-c"]

# publish *FLAGS:
# git push origin master --follow-tags

help:
	@just --list --unsorted


style-check:
	./check_style.sh


test:
	poetry run pytest -m "not integration and not e2e" --ignore=debug -v -n auto


test-verbose:
	poetry run pytest -m "not integration and not e2e" --ignore=debug -v -s


test-ci:
	mkdir -p reports
	poetry run pytest -m "not integration and not e2e" --ignore=debug -v -ra --cov=src --cov-report=xml:reports/coverage.xml --cov-report=term -n auto


test-integration:
	poetry run pytest -m integration --env=.env.integration


test-e2e:
	poetry run pytest -m e2e --env=.env.integration


snap TEST_PATH:
	poetry run pytest {{TEST_PATH}} -v --disable-warnings --snapshot-update


build:
	rm -rf build
	find src -type f -name *.pyd -exec  rm {} \;
	poetry build


build-fast:
	# Skip Cython compilation if binaries exist by setting PYO3_ONLY=true
	PYO3_ONLY=true poetry build


compile:
	poetry run python build.py


dev-install:
	poetry lock || true
	poetry install --with dev


update-docs:
	./update_docs.sh


update-version part="patch":
	@echo "Updating version ({{part}})..."
	poetry run python -c "from qubx.utils.version import update_project_version; import sys; sys.exit(0 if update_project_version('{{part}}') else 1)"


publish: build test
	@if [ "$(git symbolic-ref --short -q HEAD)" = "main" ]; then rm -rf dist && rm -rf build && poetry build && twine upload dist/*; else echo ">>> Not in master branch !"; fi


dev-publish: build
	@rm -rf dist && rm -rf build && poetry build && twine upload dist/*
