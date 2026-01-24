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


# Version & Release (using python-semantic-release)
version:
	@uv run python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])"

next-version:
	uv run semantic-release version --print --no-push --no-commit --no-tag

bump:
	uv run semantic-release version

bump-force part="patch":
	uv run semantic-release version --{{part}} --no-push
	@echo "Version bumped locally. Review changes, then: git push && git push --tags"

release:
	uv run semantic-release version --no-push
	git push && git push --tags

release-custom token="rc" part="patch":
	uv run semantic-release version --{{part}} --as-prerelease --prerelease-token {{token}} --no-vcs-release --no-push
	git push && git push --tags

changelog:
	uv run semantic-release changelog
