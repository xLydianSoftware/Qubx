# Task 018: Add Textual Dev Mode, Web Mode, and Port Options

## Goal
Add `--textual-dev`, `--textual-web`, and `--textual-port` options to `qubx run --textual` command to:
1. Enable Textual devtools console for debugging
2. Serve the Textual app in a web browser
3. Configure the port for both modes

## Background

### Textual Dev Mode
Textual framework provides dev mode for debugging TUI applications via a separate console. This is controlled by environment variables:
- `TEXTUAL=devtools` - Enables dev mode
- `TEXTUAL_DEVTOOLS_PORT=<port>` - Sets the devtools console port (default: 8081)

### Textual Web Mode
The `textual-serve` package allows serving Textual apps in a web browser:
- Runs the app as a subprocess
- Communicates via websocket
- Multiple browser tabs can connect to the same instance
- Default web server port: 8000

## Changes Made

### 1. Dependencies (`pyproject.toml`)
- Added `textual-serve = "^1.0.0"` to dependencies

### 2. CLI Options (`src/qubx/cli/commands.py`)
- Added `--textual-dev` flag to enable Textual dev mode (debug console)
- Added `--textual-web` flag to serve the app in a web browser
- Added `--textual-port` option to specify port (web server or devtools console)
  - Default: 8000 for web mode, 8081 for dev mode
- Added `--textual-host` option to specify the host for web server
  - Default: 0.0.0.0 (binds to all network interfaces, allows Tailscale/remote access)
- Updated `run()` function signature to accept these parameters
- Pass parameters to `run_strategy_yaml_in_textual()`

### 3. Textual Runner (`src/qubx/utils/runner/textual/__init__.py`)
- Added `dev_mode`, `web_mode`, `port`, and `host` parameters to `run_strategy_yaml_in_textual()`
- **Web Mode Logic:**
  - Import `textual_serve.server.Server`
  - Build qubx run command with all options
  - Start Server instance on specified host and port
  - Serve the app via websocket to browser
  - Binds to 0.0.0.0 by default (allows remote/Tailscale access)
- **Terminal Mode Logic (existing):**
  - Set `TEXTUAL=devtools` environment variable when dev_mode is True
  - Set `TEXTUAL_DEVTOOLS_PORT=<port>` environment variable
  - Run app normally in terminal

## Usage

### 1. Basic Textual Mode (terminal only)
```bash
poetry run qubx run config.yml --paper --textual
```

### 2. Web Mode (serve in browser)
```bash
# Default: host 0.0.0.0, port 8000 (accessible via Tailscale/remote)
poetry run qubx run config.yml --paper --textual --textual-web

# Custom port
poetry run qubx run config.yml --paper --textual --textual-web --textual-port 9000

# Custom host (localhost only)
poetry run qubx run config.yml --paper --textual --textual-web --textual-host localhost
```
Then open http://0.0.0.0:8000 (or your custom host/port) in a browser.
Multiple browser tabs can connect to the same running strategy instance.
With host 0.0.0.0, the server is accessible from other machines via Tailscale or local network.

### 3. Dev Mode (terminal with debug console)
```bash
# Terminal 1: Start the debug console (default port 8081)
textual console

# Terminal 2: Run qubx with dev mode
poetry run qubx run config.yml --paper --textual --textual-dev

# With custom port
textual console --port 9000
poetry run qubx run config.yml --paper --textual --textual-dev --textual-port 9000
```

### 4. Web Mode + Dev Mode (browser with debug console)
```bash
# Terminal 1: Start debug console
textual console --port 9001

# Terminal 2: Run qubx in web mode with dev console
poetry run qubx run config.yml --paper --textual --textual-web --textual-dev --textual-port 8000
```
Note: When using both modes, `--textual-port` controls the web server port.

## Testing
- ✓ Test basic terminal mode (no flags)
- ✓ Test web mode with default port
- ✓ Test web mode with custom port
- ✓ Test dev mode in terminal
- ✓ Test web mode + dev mode combo
- ✓ Verify multiple browser tabs connect to same instance

## Related Files
- `pyproject.toml:79` - Added textual-serve dependency
- `src/qubx/cli/commands.py:75-91` - CLI option definitions (including --textual-host)
- `src/qubx/utils/runner/textual/__init__.py:21-122` - Textual runner implementation with host support
