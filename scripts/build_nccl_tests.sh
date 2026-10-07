#!/usr/bin/env bash
# Clone nccl-tests at a pinned commit and build it against the NCCL shipped in
# the Python environment's nvidia-nccl-cu12 wheel (the NCCL PyTorch uses).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT/third_party"

NCCL_PKG="$("$PYTHON" -c 'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__) if nvidia.nccl.__file__ else list(nvidia.nccl.__path__)[0])')"
mkdir -p nccl_home/lib
ln -sfn "$NCCL_PKG/include" nccl_home/include
ln -sf "$NCCL_PKG/lib/libnccl.so.2" nccl_home/lib/libnccl.so.2
ln -sf libnccl.so.2 nccl_home/lib/libnccl.so

if [ ! -d nccl-tests/.git ]; then
  git clone https://github.com/NVIDIA/nccl-tests.git
fi
git -C nccl-tests fetch -q origin
git -C nccl-tests checkout -q "$NCCL_TESTS_COMMIT"
make -C nccl-tests -j16 CUDA_HOME="$CUDA_HOME" NCCL_HOME="$PWD/nccl_home" \
     NVCC_GENCODE="$NVCC_GENCODE" BUILDDIR="$PWD/nccl-tests/build" src.build
ls -1 nccl-tests/build/*_perf
