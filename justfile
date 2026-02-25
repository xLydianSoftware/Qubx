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
	uv run python hatch_build.py


install:
	uv sync --all-extras
	just compile


lock:
	uv lock


update-docs:
	./update_docs.sh


# Version (tag-driven via hatch-vcs)
version:
	@python -c "from qubx._version import __version__; print(__version__)" 2>/dev/null || \
	 git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "dev"

# Create and push a release tag. Channel auto-detected from branch (main=stable, dev=dev) or set explicitly.
release BUMP CHANNEL="":
	#!/usr/bin/env bash
	set -e
	BUMP="{{BUMP}}"
	CHANNEL="{{CHANNEL}}"
	# Auto-detect channel from branch if not specified
	if [ -z "$CHANNEL" ]; then
		BRANCH=$(git branch --show-current)
		if [ "$BRANCH" = "main" ]; then
			CHANNEL="stable"
		elif [ "$BRANCH" = "dev" ]; then
			CHANNEL="dev"
		else
			echo "Error: cannot auto-detect channel from branch '$BRANCH'. Specify explicitly: just release <bump> <channel>"
			exit 1
		fi
	fi
	# Get latest stable tag (ignore rc/dev tags)
	LATEST_STABLE=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' | grep -v 'rc\|dev' | sort -V | tail -n1 || echo "v0.0.0")
	if [ -z "$LATEST_STABLE" ]; then LATEST_STABLE="v0.0.0"; fi
	# Parse version components
	VERSION=${LATEST_STABLE#v}
	MAJOR=$(echo $VERSION | cut -d. -f1)
	MINOR=$(echo $VERSION | cut -d. -f2)
	PATCH=$(echo $VERSION | cut -d. -f3)
	# Apply bump
	case $BUMP in
		major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
		minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
		patch) PATCH=$((PATCH + 1)) ;;
		*) echo "Error: invalid bump type '$BUMP'. Use: major, minor, patch"; exit 1 ;;
	esac
	BASE_VERSION="${MAJOR}.${MINOR}.${PATCH}"
	# Determine final version based on channel
	case $CHANNEL in
		stable)
			TAG="v${BASE_VERSION}"
			;;
		rc)
			EXISTING=$(git tag -l "v${BASE_VERSION}rc*" | grep -oP 'rc\K[0-9]+' | sort -n | tail -n1 || echo "0")
			if [ -z "$EXISTING" ]; then EXISTING=0; fi
			NEXT=$((EXISTING + 1))
			TAG="v${BASE_VERSION}rc${NEXT}"
			;;
		dev)
			EXISTING=$(git tag -l "v${BASE_VERSION}.dev*" | grep -oP '\.dev\K[0-9]+' | sort -n | tail -n1 || echo "0")
			if [ -z "$EXISTING" ]; then EXISTING=0; fi
			NEXT=$((EXISTING + 1))
			TAG="v${BASE_VERSION}.dev${NEXT}"
			;;
		*) echo "Error: invalid channel '$CHANNEL'. Use: stable, rc, dev"; exit 1 ;;
	esac
	echo "Creating tag: $TAG (bump=$BUMP, channel=$CHANNEL)"
	# Update changelog and commit before tagging
	uv run git-cliff --tag "$TAG" --output CHANGELOG.md
	git add CHANGELOG.md
	git commit -m "chore: update changelog for ${TAG#v}"
	# Tag and push
	git tag -a "$TAG" -m "Release ${TAG#v}"
	git push origin "$(git branch --show-current)" && git push origin "$TAG"
	uv run python -m setuptools_scm
	echo "Pushed $TAG — CI will build and publish automatically."

# Preview changelog for unreleased changes
changelog:
	uv run git-cliff --unreleased

# Generate full changelog
changelog-full:
	uv run git-cliff
