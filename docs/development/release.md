# Release Process

Guide to releasing new versions of Qubx using [python-semantic-release](https://python-semantic-release.readthedocs.io/).

## Conventional Commits

Qubx uses [Conventional Commits](https://www.conventionalcommits.org/) to automatically determine version bumps:

| Commit Type | Example | Version Bump |
|-------------|---------|--------------|
| `feat:` | `feat: add position sizing` | Minor (0.X.0) |
| `fix:` | `fix: correct order placement` | Patch (0.0.X) |
| `perf:` | `perf: optimize backtest loop` | Patch (0.0.X) |
| `BREAKING CHANGE:` | `feat!: redesign API` | Major (X.0.0) |

Other commit types (`docs`, `style`, `refactor`, `test`, `build`, `ci`, `chore`) do not trigger version bumps.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Examples:**

```bash
# Feature (minor bump)
git commit -m "feat(strategy): add trailing stop support"

# Bug fix (patch bump)
git commit -m "fix(backtest): correct position size calculation"

# Breaking change (major bump)
git commit -m "feat!: redesign IStrategy interface

BREAKING CHANGE: Strategy.on_event() signature changed"
```

## Version Commands

Available commands via `just`:

```bash
# Show current version (works on any branch)
just version

# Preview next version (dry run, no changes)
just next-version

# Create release commit + tag (standard workflow)
just bump

# Force version bump from any branch (no auto-push)
just bump-force patch   # or minor, major

# Full release: version + changelog + push
just release

# Custom prerelease from any branch (e.g., 0.7.38-mytoken.1)
just release-custom mytoken        # patch bump with custom token
just release-custom mytoken minor  # minor bump with custom token

# Generate/update changelog only
just changelog
```

## Release Workflow

### Standard Release (from main or dev)

```bash
# 1. Ensure clean working tree on main/dev
git checkout main && git pull

# 2. Create release (version bump + changelog + tag + push)
just release
```

This will:

1. Analyze commits since the last release
2. Determine the next version based on commit types
3. Update version in `pyproject.toml`
4. Update `CHANGELOG.md`
5. Create a commit and tag
6. Push to remote (triggers CI deployment)

### Custom Release (from any branch)

For releases from any branch with custom prerelease tokens:

```bash
# Create and push prerelease with custom token (e.g., 0.7.38-uv.1)
just release-custom uv           # patch bump + push
just release-custom uv minor     # minor bump + push
```

This creates the version, commits, tags, and pushes in one command.

Alternatively, force a regular version bump without pushing:

```bash
just bump-force patch  # or minor, major
# Then manually: git push && git push --tags
```

### Preview Next Version

To see what version would be next without making any changes:

```bash
just next-version
```

## CI/CD Pipeline

The CI workflow is triggered by:

- **Push to main/dev**: Runs build and tests
- **Pull requests**: Runs build and tests
- **Tag push (v*)**: Triggers deployment

### Deployment Jobs

When a `v*` tag is pushed, CI automatically:

1. **deploy-test-pypi**: Publishes to TestPyPI
2. **deploy-pypi**: Publishes to PyPI
3. **deploy-docs**: Deploys documentation via mike

### Manual Workflow Dispatch

You can manually trigger the build workflow from the GitHub Actions UI for testing purposes.

## Changelog

The changelog is automatically generated from commit messages. It's stored in `CHANGELOG.md` at the project root.

### Excluded Patterns

The following commit patterns are excluded from the changelog:

- `chore(release):` - Release commits
- `Merge` - Merge commits

### Manual Changelog Update

To regenerate the changelog without creating a release:

```bash
just changelog
```

## Branch Configuration

| Branch | Prerelease | Token | Example Version |
|--------|------------|-------|-----------------|
| `main` | No | - | `0.8.0` |
| `dev` | Yes | `dev` | `0.8.0-dev.1` |
| Any branch | Yes | Custom | `0.8.0-mytoken.1` |

Use `just release-custom <token>` from any branch to create a prerelease with your chosen token.

## Troubleshooting

### "No release will be made"

This happens when:

- No commits match release patterns (`feat:`, `fix:`, `perf:`)
- All commits are excluded types (`docs`, `chore`, etc.)

**Solution:** Ensure at least one commit uses a release-triggering type.

### Version Not Updating

Check that:

1. You're on the correct branch (`main` or `dev`)
2. The commit follows conventional commit format
3. You have the latest commits: `git fetch --all`

### Tag Already Exists

If a tag already exists for the version:

```bash
# Delete local tag
git tag -d v0.8.0

# Delete remote tag (use with caution)
git push origin :refs/tags/v0.8.0
```

### Force a Specific Version

Use `bump-force` to manually control the version increment:

```bash
# Force a patch release
just bump-force patch

# Force a minor release
just bump-force minor

# Force a major release
just bump-force major
```
