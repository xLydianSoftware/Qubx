# Release Process

Guide to releasing new versions of Qubx.

## How Releases Work

Releases are **fully automatic**. Merging to `main` or `dev` triggers the release pipeline:

| Branch | What happens | Version format | Example |
|--------|-------------|----------------|---------|
| `dev` | Auto-releases a dev pre-release | `X.Y.Z.devN` | `1.0.2.dev3` |
| `main` | Auto-releases a stable version | `X.Y.Z` | `1.1.0` |

No manual tagging required. The pipeline determines the version, builds everything, and publishes.

## Version Determination

### Dev branch (pre-releases)
Each push to `dev` increments the dev suffix: `1.0.2.dev1` → `1.0.2.dev2` → `1.0.2.dev3`.

The base version is the latest stable tag + patch bump.

### Main branch (stable releases)
The bump type is determined automatically from conventional commits since the last stable tag:

| Commit pattern | Bump | Example |
|---|---|---|
| `feat!:` or `BREAKING CHANGE` | **major** | `1.0.1` → `2.0.0` |
| `feat:` | **minor** | `1.0.1` → `1.1.0` |
| `fix:`, `perf:`, `refactor:`, etc. | **patch** | `1.0.1` → `1.0.2` |

## Conventional Commits

Qubx uses [Conventional Commits](https://www.conventionalcommits.org/) for both changelog generation and version detection:

| Commit Type | Example | Changelog Section |
|---|---|---|
| `feat:` | `feat: add position sizing` | Features |
| `fix:` | `fix: correct order placement` | Bug Fixes |
| `perf:` | `perf: optimize backtest loop` | Performance |
| `docs:` | `docs: update API reference` | Documentation |
| `refactor:` | `refactor: simplify broker logic` | Refactoring |
| `test:` | `test: add backtest coverage` | Testing |

### Commit message format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Examples:**

```bash
# Feature (triggers minor bump on main)
git commit -m "feat(strategy): add trailing stop support"

# Bug fix (triggers patch bump on main)
git commit -m "fix(backtest): correct position size calculation"

# Breaking change (triggers major bump on main)
git commit -m "feat!: redesign IStrategy interface

BREAKING CHANGE: Strategy.on_event() signature changed"
```

## Release Pipeline

```
Push to main/dev
  │
  ├── Determine version (from commits)
  │
  ├── Build sdist ──────────────┐
  ├── Build wheels (6 matrix) ──┤  Phase 1: parallel builds
  ├── Build Docker (from src) ──┤
  │                             │
  ├── Test wheel install ◄──────┘  Phase 2: gate
  │
  ├── Create tag + changelog ──┐
  ├── Publish to PyPI ─────────┤  Phase 3: publish
  ├── Push Docker image ───────┤
  │                            │
  ├── GitHub Release (stable) ─┤  Phase 4: post-publish
  └── Deploy docs ─────────────┘
```

### What gets built

| Platform | Python | Wheel tag |
|---|---|---|
| Linux | 3.12, 3.13 | `manylinux_x86_64` |
| macOS Apple Silicon | 3.12, 3.13 | `macosx_arm64` |
| Windows | 3.12, 3.13 | `win_amd64` |

Docker image is built from source (multi-stage, no PyPI dependency).

## Manual Override

Normally you never need to manually create a release. But if needed:

```bash
# Preview what the next version would be
just next-version
just next-version dev
just next-version stable

# Show current version
just version

# Manual release (creates tag + pushes, triggers pipeline)
just release           # auto-detect channel from branch
just release stable    # force stable
just release dev       # force dev

# Preview changelog
just changelog
```

## CI/CD Pipelines

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | Push to main/dev, PRs | Lint + unit tests (fast feedback) |
| `release.yml` | Push to main/dev | Build, test, publish, deploy |

Both run on push to main/dev. CI gives fast lint+test feedback. Release does the full build+publish pipeline.

### Environments

The following GitHub environment is used with OIDC trusted publishing:

| Setting | Value |
|---|---|
| Environment | `pypi` |
| Owner | `xLydianSoftware` |
| Repository | `Qubx` |
| Workflow | `release.yml` |

## Changelog

The `CHANGELOG.md` is updated automatically by the release pipeline using [git-cliff](https://git-cliff.org/) based on conventional commits.

```bash
# Preview unreleased changes
just changelog

# Generate full changelog
just changelog-full
```

## Troubleshooting

### Tag already exists (pipeline skips release)

If a push to main/dev produces a version that already has a tag, the pipeline skips the release. This is normal for non-conventional commits (docs, chore, etc.).

### Version not showing correctly

The version is derived from git tags via `hatch-vcs`:

```bash
git fetch --tags
just version
```

### Triggering a release manually

Use the workflow dispatch trigger in GitHub Actions, or:

```bash
just release
```
