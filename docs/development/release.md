# Release Process

Guide to releasing new versions of Qubx.

## Conventional Commits

Qubx uses [Conventional Commits](https://www.conventionalcommits.org/) for changelog generation:

| Commit Type | Example | Changelog Section |
|-------------|---------|-------------------|
| `feat:` | `feat: add position sizing` | Features |
| `fix:` | `fix: correct order placement` | Bug Fixes |
| `perf:` | `perf: optimize backtest loop` | Performance |
| `docs:` | `docs: update API reference` | Documentation |
| `refactor:` | `refactor: simplify broker logic` | Refactoring |
| `test:` | `test: add backtest coverage` | Testing |

Release commits (`chore(release):`) and dependency updates (`chore(deps):`) are excluded from changelogs.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Examples:**

```bash
# Feature
git commit -m "feat(strategy): add trailing stop support"

# Bug fix
git commit -m "fix(backtest): correct position size calculation"

# Breaking change
git commit -m "feat!: redesign IStrategy interface

BREAKING CHANGE: Strategy.on_event() signature changed"
```

## Version Format (PEP 440)

| Type | Tag Format | PyPI Version | Install Command |
|------|------------|--------------|-----------------|
| Stable | `v0.7.40` | `0.7.40` | `pip install qubx` |
| RC | `v0.7.40rc1` | `0.7.40rc1` | `pip install --pre qubx` |
| Dev | `v0.7.40.dev1` | `0.7.40.dev1` | `pip install --pre qubx` |

## Creating a Release

Releases are created locally using `just release` and published automatically by CI.

### Commands

```bash
# Show current version
just version

# Create a stable release (patch/minor/major)
just release patch
just release minor
just release major

# Create a release candidate
just release patch rc
just release minor rc

# Create a dev pre-release
just release patch dev
just release minor dev

# Preview changelog for unreleased changes
just changelog

# Build package locally
just build
```

### Channel Auto-Detection

The release channel is auto-detected from the current git branch:

| Branch | Default Channel |
|--------|----------------|
| `main` | `stable` |
| `dev` | `dev` |
| other | error (must specify channel explicitly) |

You can always override with an explicit channel:

```bash
# Force stable release from any branch
just release patch stable

# Force rc from dev branch
just release patch rc
```

### Version Auto-Increment Examples

```
Given: Latest stable tag is v0.7.39

just release patch            → v0.7.40       (on main)
just release patch            → v0.7.40.dev1  (on dev)
just release patch rc         → v0.7.40rc1    (or rc2, rc3... if exists)
just release patch dev        → v0.7.40.dev1  (or dev2, dev3... if exists)
just release minor            → v0.8.0        (on main)
just release major rc         → v1.0.0rc1
```

### What Happens After Tag Push

When a `v*` tag is pushed, the **Build and Publish** workflow automatically:

1. **Build source distribution** (sdist)
2. **Build platform-specific wheels** via cibuildwheel (6 wheels in parallel):

   | Platform | Python Versions | Wheel Tag |
   |----------|-----------------|-----------|
   | Linux | 3.12, 3.13 | `manylinux_x86_64` |
   | macOS Apple Silicon | 3.12, 3.13 | `macosx_arm64` |
   | Windows | 3.12, 3.13 | `win_amd64` |

3. **Test wheel installation** - verifies Cython imports work
4. **Publish to TestPyPI** - skips if version already exists
5. **Publish to PyPI** - skips if version already exists
6. **Create GitHub Release** with artifacts and release notes (stable releases only)
7. **Deploy documentation** (stable releases only)

### What `just release` Does Locally

1. Auto-detect channel from branch (or use explicit override)
2. Calculate next version based on latest stable tag
3. Update `CHANGELOG.md` via git-cliff and commit it
4. Create annotated git tag on the changelog commit
5. Push branch and tag to origin
6. Regenerate `src/qubx/_version.py` so the local version is up to date

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        just release <bump>                          │
│                          (local command)                            │
├─────────────────────────────────────────────────────────────────────┤
│  1. Auto-detect channel from branch (or use explicit override)     │
│  2. Calculate next version                                         │
│  3. Update CHANGELOG.md and commit                                 │
│  4. Create annotated git tag                                       │
│  5. Push branch + tag to origin                                    │
│  6. Regenerate _version.py locally                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ (tag push triggers)
┌─────────────────────────────────────────────────────────────────────┐
│                       Build and Publish                              │
│                      (Triggered by v* tags)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐  ┌──────────────────────────────────────────────────┐ │
│  │  sdist   │  │              build-wheels (6 parallel)           │ │
│  └────┬─────┘  │  cp312-linux  cp313-linux  cp312-macos           │ │
│       │        │  cp313-macos  cp312-win    cp313-win             │ │
│       │        └──────────────────────┬─────────────────────────────┘ │
│       │                               │                              │
│       └───────────────┬───────────────┘                              │
│                       ▼                                              │
│               ┌──────────────┐                                       │
│               │ test-install │                                       │
│               └──────┬───────┘                                       │
│                      ▼                                              │
│            ┌──────────────────┐                                      │
│            │ publish-testpypi │                                      │
│            └────────┬─────────┘                                      │
│                     ▼                                               │
│            ┌──────────────────┐                                      │
│            │  publish-pypi    │                                      │
│            └────────┬─────────┘                                      │
│                     │                                               │
│      ┌──────────────┼──────────────┐                                │
│      ▼              ▼              ▼                                │
│ ┌─────────┐  ┌─────────────┐  ┌──────────┐                         │
│ │ github  │  │ deploy-docs │  │  (done)  │                         │
│ │ release │  │             │  │          │                         │
│ │(stable) │  │  (stable)   │  │ (rc/dev) │                         │
│ └─────────┘  └─────────────┘  └──────────┘                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## CI/CD Pipeline

### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push to main/dev, PRs | Lint, build, test |
| `build-publish.yml` | Tag push (v*) | Build wheels, publish to PyPI, GitHub Release |

### Environments

The following GitHub environments are used with OIDC trusted publishing:

- **testpypi**: For TestPyPI publishing
- **pypi**: For PyPI publishing

### Trusted Publisher Configuration

Both TestPyPI and PyPI use OIDC trusted publishing. Configure on each platform:

| Setting | Value |
|---------|-------|
| Owner | `xLydianSoftware` |
| Repository | `Qubx` |
| Workflow | `build-publish.yml` |
| Environment | `testpypi` or `pypi` |

## Changelog

The `CHANGELOG.md` in the repo root is the canonical changelog, updated automatically on every release by `just release`.

It is generated using [git-cliff](https://git-cliff.org/) based on conventional commits. The format is configured in `cliff.toml`, which groups commits by type and links to GitHub PRs/issues.

### Preview Unreleased Changes

```bash
# Preview what would be added to the changelog
just changelog

# Generate full changelog (without committing)
just changelog-full
```

## Troubleshooting

### Tag Already Exists

If a tag already exists:

```bash
# Delete local tag
git tag -d v0.8.0

# Delete remote tag (use with caution)
git push origin :refs/tags/v0.8.0
```

### Build Fails on CI

1. Check the workflow logs in GitHub Actions
2. Ensure all Cython modules compile correctly
3. Verify build dependencies in `pyproject.toml`

### Version Not Showing Correctly

The version is derived from git tags via `hatch-vcs`. Ensure:

1. You have fetched all tags: `git fetch --tags`
2. The tag follows the format `vX.Y.Z` or `vX.Y.ZrcN` or `vX.Y.Z.devN`

### Local Development Version

During development (no tags), the version will show as `0.0.0.dev0` or similar. This is expected behavior with `hatch-vcs`.

To see what version would be built:

```bash
just version
```

### PyPI Publishing Fails

If trusted publishing fails with "invalid-publisher":

1. Verify the trusted publisher is configured on PyPI/TestPyPI
2. Check that the workflow name matches exactly: `build-publish.yml`
3. Ensure the environment name matches: `pypi` or `testpypi`
