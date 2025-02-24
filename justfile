set windows-shell := ["C:\\Program Files\\Git\\bin\\sh.exe", "-c"]

# publish *FLAGS:
# git push origin master --follow-tags

help:
	@just --list --unsorted


style-check:
	./check_style.sh


test:
	poetry run pytest -m "not integration" -v


test-verbose:
	poetry run pytest -m "not integration" -v -s


test-integration:
	poetry run pytest -m integration --env=.env.integration


test-ci:
	poetry run pytest -m "not integration" -v -s --cov=src --cov-report=xml --cov-report=term


build:
	rm -rf build
	find src -type f -name *.pyd -exec  rm {} \;
	poetry build


dev-install:
	poetry lock --no-update || true
	poetry install
	

publish: build test
	@if [ "$(git symbolic-ref --short -q HEAD)" = "main" ]; then rm -rf dist && rm -rf build && poetry build && twine upload dist/*; else echo ">>> Not in master branch !"; fi


dev-publish: build
	@rm -rf dist && rm -rf build && poetry build && twine upload dist/*
