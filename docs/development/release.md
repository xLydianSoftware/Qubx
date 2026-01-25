# Release Process

Guide to releasing new versions of Qubx using CI-only releases via GitHub Actions.

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

## Local Commands

Available commands via `just`:

```bash
# Show current version
just version

# Preview changelog for unreleased changes
just changelog

# Generate full changelog
just changelog-full

# Build package locally
just build
```

## Release Workflow (CI-Only)

Releases are created exclusively through GitHub Actions. This ensures consistent builds across platforms.

### Creating a Release

1. Go to **Actions** → **Create Release Tag** workflow
2. Click **Run workflow**
3. Select options:
   - **bump_type**: `patch`, `minor`, or `major`
   - **channel**: `stable`, `rc`, or `dev`
4. Click **Run workflow**

The workflow will:

1. Calculate the next version (auto-increments rc/dev numbers)
2. Create and push the git tag
3. Generate release notes using git-cliff
4. Create a GitHub Release (marked as prerelease for rc/dev)

### Version Auto-Increment Examples

```
Given: Latest stable tag is v0.7.39

bump=patch, channel=stable  → v0.7.40
bump=patch, channel=rc      → v0.7.40rc1  (or rc2, rc3... if exists)
bump=patch, channel=dev     → v0.7.40.dev1 (or dev2, dev3... if exists)
bump=minor, channel=stable  → v0.8.0
bump=major, channel=rc      → v1.0.0rc1
```

### What Happens After Tag Creation

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
6. **Upload artifacts to GitHub Release** (stable releases only)
7. **Deploy documentation** (stable releases only)

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Create Release Tag                              │
│                    (Manual workflow dispatch)                        │
├─────────────────────────────────────────────────────────────────────┤
│  1. Calculate next version                                          │
│  2. Create git tag                                                  │
│  3. Generate release notes (git-cliff)                              │
│  4. Create GitHub Release                                           │
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
| `create-tag.yml` | Manual dispatch | Create release tags + GitHub Release |
| `build-publish.yml` | Tag push (v*) | Build wheels, publish to PyPI |

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

Changelogs are generated automatically using [git-cliff](https://git-cliff.org/) based on conventional commits.

### Preview Unreleased Changes

```bash
just changelog
```

### Configuration

The changelog format is configured in `cliff.toml`. It groups commits by type and links to GitHub PRs/issues.

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

## Migration from python-semantic-release

The previous release system used `python-semantic-release` with local commands. The new system:

- **Removed**: `just bump`, `just bump-force`, `just release`, `just release-custom`, `just next-version`
- **Added**: GitHub Actions workflow for tag creation
- **Changed**: Version source from `pyproject.toml` to git tags via `hatch-vcs`
- **Changed**: Changelog generation from semantic-release to git-cliff
- **Changed**: Build backend from poetry-core to hatchling
