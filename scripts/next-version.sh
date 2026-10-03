#!/bin/bash
# Determine the next version based on conventional commits and the current branch.
#
# Usage:
#   scripts/next-version.sh          # auto-detect from branch
#   scripts/next-version.sh dev      # force dev channel
#   scripts/next-version.sh stable   # force stable channel
#
# On dev branch:   bumps dev suffix (1.0.1.dev2 → 1.0.1.dev3)
# On main branch:  bumps patch/minor/major based on commit types since last stable tag
#   - feat:           → minor bump
#   - fix/perf/other: → patch bump
#   - feat!/BREAKING: → major bump
set -e

CHANNEL="${1:-}"

# Auto-detect channel from branch
if [ -z "$CHANNEL" ]; then
    BRANCH=$(git branch --show-current 2>/dev/null || echo "")
    case "$BRANCH" in
        main)   CHANNEL="stable" ;;
        dev)    CHANNEL="dev" ;;
        *)      echo "Cannot auto-detect channel from branch '$BRANCH'" >&2; exit 1 ;;
    esac
fi

# Get latest stable tag (vX.Y.Z without .dev or rc suffix)
LATEST_STABLE=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n1 || echo "")
if [ -z "$LATEST_STABLE" ]; then
    LATEST_STABLE="v0.0.0"
fi

# Parse version components
VERSION=${LATEST_STABLE#v}
MAJOR=$(echo "$VERSION" | cut -d. -f1)
MINOR=$(echo "$VERSION" | cut -d. -f2)
PATCH=$(echo "$VERSION" | cut -d. -f3)

# Next stable version from conventional commits since the last stable tag:
# breaking → major, feat → minor, otherwise patch. Dev builds use the same base,
# so a dev build of a breaking change is already versioned as the next major.
HAS_BREAKING=false
HAS_FEAT=false
while IFS= read -r msg; do
    if echo "$msg" | grep -qE '^[a-z]+(\(.+\))?!:'; then
        HAS_BREAKING=true
    fi
    if echo "$msg" | grep -qi 'BREAKING CHANGE'; then
        HAS_BREAKING=true
    fi
    if echo "$msg" | grep -qE '^feat(\(.+\))?:'; then
        HAS_FEAT=true
    fi
    # tformat (not format) terminates every line — format omits the trailing
    # newline, so `while read` drops the last commit (the whole range for a squash merge).
done < <(git log "${LATEST_STABLE}..HEAD" --pretty=tformat:"%s" 2>/dev/null)

if [ "$HAS_BREAKING" = true ]; then
    MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0
elif [ "$HAS_FEAT" = true ]; then
    MINOR=$((MINOR + 1)); PATCH=0
else
    PATCH=$((PATCH + 1))
fi
BASE="${MAJOR}.${MINOR}.${PATCH}"

if [ "$CHANNEL" = "dev" ]; then
    LATEST_DEV=$(git tag -l "v${BASE}.dev*" | grep -oP '\.dev\K[0-9]+' | sort -n | tail -n1 || echo "")
    if [ -z "$LATEST_DEV" ]; then
        NEXT_DEV=1
    else
        NEXT_DEV=$((LATEST_DEV + 1))
    fi
    echo "v${BASE}.dev${NEXT_DEV}"

elif [ "$CHANNEL" = "stable" ]; then
    echo "v${BASE}"
else
    echo "Unknown channel: $CHANNEL" >&2
    exit 1
fi
