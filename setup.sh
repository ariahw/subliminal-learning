# Setup file

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a  # Automatically export all variables
    source .env
    set +a  # Turn off automatic export
    echo "Loaded environment variables from .env file"
else
    echo ".env file not found, skipping environment variable loading"
fi

# Recommended settings
export WANDB_LOG_MODEL=false # Prevent sending model to weights and biases, prefer local storage
export WANDB_START_METHOD=thread # Use thread instead of process to avoid issues with wandb
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Recommended settings for flash attention install
# Change these settings if not for Ampere RTX A6000 is CC 8.6
export _GLIBCXX_USE_CXX11_ABI=0
export USE_CXX11_ABI=0
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=8


# Install uv
pip install uv

# Sync uv
uv sync