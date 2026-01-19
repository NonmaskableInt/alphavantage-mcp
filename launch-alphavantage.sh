#!/usr/bin/env sh
# Launch script for AlphaVantage MCP Server
# Works on macOS, Linux, and other POSIX-compatible systems

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Common uv installation paths (checked on all platforms)
UV_PATHS="
$HOME/.local/bin
$HOME/.cargo/bin
/opt/homebrew/bin
/usr/local/bin
$HOME/bin
"

# Add all existing paths
for p in $UV_PATHS; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done

# Also try to find uv in common locations if still not found
find_uv() {
    # Check if uv is already in PATH
    command -v uv 2>/dev/null && return 0

    # Try common absolute paths
    for uv_path in \
        "$HOME/.local/bin/uv" \
        "$HOME/.cargo/bin/uv" \
        "/opt/homebrew/bin/uv" \
        "/usr/local/bin/uv" \
        "$HOME/bin/uv"
    do
        if [ -x "$uv_path" ]; then
            echo "$uv_path"
            return 0
        fi
    done

    return 1
}

UV_BIN=$(find_uv)

if [ -z "$UV_BIN" ]; then
    echo "Error: 'uv' is not installed or not in PATH" >&2
    echo "Searched paths:" >&2
    for p in $UV_PATHS; do
        echo "  - $p" >&2
    done
    echo "" >&2
    echo "Install uv from: https://docs.astral.sh/uv/getting-started/installation/" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

exec "$UV_BIN" run alphavantage-mcp-server
