set windows-shell := ["C:\\Program Files\\Git\\bin\\sh.exe", "-c"]

# publish *FLAGS:
# git push origin master --follow-tags

help:
	@just --list --unsorted


style-check:
	./check_style.sh


test:
	uv run pytest -m "not integration and not e2e" --ignore=debug -v -n auto


test-verbose:
	uv run pytest -m "not integration and not e2e" --ignore=debug -v -s


test-ci:
	mkdir -p reports
	uv run pytest -m "not integration and not e2e" --ignore=debug -v -ra --cov=src --cov-report=xml:reports/coverage.xml --cov-report=term -n auto


test-integration:
	uv run pytest -m integration --env=.env.integration


test-e2e:
	uv run pytest -m e2e --env=.env.integration


snap TEST_PATH:
	uv run pytest {{TEST_PATH}} -v --disable-warnings --snapshot-update


clean:
	rm -rf .venv build dist *.egg-info
	find src -name "*.so" -delete
	find src -name "*.pyd" -delete


build:
	rm -rf build dist
	uv build


compile:
	uv run python build.py


install:
	uv sync --all-extras
	just compile


lock:
	uv lock


update-docs:
	./update_docs.sh


update-version part="patch":
	@echo "Updating version ({{part}})..."
	uv run python -c "from qubx.utils.version import update_project_version; import sys; sys.exit(0 if update_project_version('{{part}}') else 1)"


publish: build test
	@if [ "$(git symbolic-ref --short -q HEAD)" = "main" ]; then rm -rf dist && rm -rf build && uv build && twine upload dist/*; else echo ">>> Not in master branch !"; fi


dev-publish: build
	@rm -rf dist && rm -rf build && uv build && twine upload dist/*
