set windows-shell := ["C:\\Program Files\\Git\\bin\\sh.exe", "-c"]

# publish *FLAGS:
# git push origin master --follow-tags

help:
	@just --list --unsorted


style-check:
	./scripts/check_style.sh


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
	uv run python scripts/build.py


install:
	uv sync --all-extras
	just compile


lock:
	uv lock


update-docs:
	./scripts/update_docs.sh


# Version (tag-driven via hatch-vcs)
version:
	@git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || \
	 python -c "from qubx._version import __version__; print(__version__)" 2>/dev/null || echo "dev"

# Preview what the next auto-release version would be
next-version CHANNEL="":
	@./scripts/next-version.sh {{CHANNEL}}

# Manual release override — creates a tag and pushes (triggers release pipeline).
# Normally not needed — merging to main/dev triggers auto-release.
release CHANNEL="":
	#!/usr/bin/env bash
	set -e
	TAG=$(./scripts/next-version.sh {{CHANNEL}})
	echo "Creating tag: $TAG"
	uv run git-cliff --tag "$TAG" --output CHANGELOG.md
	git add CHANGELOG.md
	git commit -m "chore(release): ${TAG#v}"
	git tag -a "$TAG" -m "Release ${TAG#v}"
	git push origin "$(git branch --show-current)" --follow-tags
	echo "Pushed $TAG — release pipeline will build and publish."

# Preview changelog for unreleased changes
changelog:
	uv run git-cliff --unreleased

# Generate full changelog
changelog-full:
	uv run git-cliff
