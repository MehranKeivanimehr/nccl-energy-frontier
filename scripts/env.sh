# Shared settings. Override these in the environment when needed.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PYTHON:=python3}"
: "${CUDA_HOME:=/usr/local/cuda-12.4}"
: "${NVCC_GENCODE:=-gencode=arch=compute_86,code=sm_86}"
NCCL_TESTS_COMMIT=afd59abab774a6a4b492eb344bd0c845a0b7bb4d
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
