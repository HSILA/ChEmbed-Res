#!/bin/bash
set -e

# 1. Check and install uv if missing
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Reload environment for the current script
    # UV installer typically installs to ~/.local/bin or ~/.cargo/bin
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    echo "uv installed and environment reloaded for this script."
else
    echo "uv is already installed."
fi

# 2. Check and create venv if missing
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
else
    echo "Virtual environment already exists."
fi

# 3. Activate venv
# We activate it so subsequent 'uv pip install' commands target this environment
# shellcheck source=/dev/null
source .venv/bin/activate

# 4. Install build dependencies first
echo "Installing build dependencies..."
uv pip install torch==2.9.0 setuptools wheel ninja

# 5. Calculate MAX_JOBS
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
CORES=$(nproc)
# Handle potential zero or empty values safely
RAM_GB=${RAM_GB:-1}
CORES=${CORES:-1}

JOBS_RAM=$((RAM_GB / 2))
JOBS_CORES=$((CORES / 2))

if [ "$JOBS_RAM" -lt "$JOBS_CORES" ]; then
    MAX_JOBS="$JOBS_RAM"
else
    MAX_JOBS="$JOBS_CORES"
fi

# Ensure MAX_JOBS is at least 1
if [ "$MAX_JOBS" -lt 1 ]; then
    MAX_JOBS=1
fi

echo "Auto-configuring build: Detected ${RAM_GB}GB RAM, ${CORES} Cores -> MAX_JOBS=${MAX_JOBS}"

# 6. Install flash attention with limits
echo "Installing mteb[flash_attention]..."
MAX_JOBS=$MAX_JOBS uv pip install "mteb[flash_attention]==2.7.12" --no-build-isolation

# 7. Install additional MTEB extras
echo "Installing mteb extras (gritlm, openai)..."
uv pip install "mteb[gritlm]" "mteb[openai]"

# 7. Install the project and other dependencies
echo "Installing project dependencies..."
uv pip install .

echo "Installation complete!"
echo "Run 'source .venv/bin/activate' to start using the environment."
