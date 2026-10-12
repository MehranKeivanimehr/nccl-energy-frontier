# Shared settings for all scripts. Override any of these in the environment.
# PYTHON must have numpy, pandas, matplotlib, PyYAML, nvidia-ml-py and the
# nvidia-nccl-cu12 wheel (the NCCL library that nccl-tests is built against).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PYTHON:=python3}"
: "${CUDA_HOME:=/usr/local/cuda-12.4}"
: "${NVCC_GENCODE:=-gencode=arch=compute_86,code=sm_86}"   # RTX A6000 = sm_86
NCCL_TESTS_COMMIT=afd59abab774a6a4b492eb344bd0c845a0b7bb4d     # nccl-tests v2.21.1
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
