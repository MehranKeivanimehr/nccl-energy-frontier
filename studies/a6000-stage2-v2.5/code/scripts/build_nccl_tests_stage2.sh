#!/usr/bin/env bash
# Stage 2 binary: nccl-tests afd59ab + the unchanged v2.1 region patch + the Stage 2 patch, in its own directory.
#   STAGE2_PATCH_VERSION  v2.5 (default; corrective gate cg1) -> third_party/nccl-tests-stage2-v25
#                         v2.4 (superseded, kept for provenance) -> third_party/nccl-tests-stage2
# CPU-only compile; nothing here touches a GPU or NVML.
#   NCCL_TESTS_SRC  local nccl-tests git checkout containing commit afd59ab (cloned with --no-hardlinks, never modified)
#   NCCL_PKG_DIR    directory with include/nccl.h and lib/libnccl.so.2 (default: the Python env's nvidia-nccl-cu12 wheel)
# Refuses to overwrite an existing build directory.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT/third_party"
: "${NCCL_TESTS_SRC:?set NCCL_TESTS_SRC to a local nccl-tests checkout containing $NCCL_TESTS_COMMIT}"
V="${STAGE2_PATCH_VERSION:-v2.5}"
case "$V" in
  v2.5) DEST=nccl-tests-stage2-v25 ;;
  v2.4) DEST=nccl-tests-stage2 ;;
  *) echo "unknown STAGE2_PATCH_VERSION $V" >&2; exit 2 ;;
esac
P21="$REPO_ROOT/patches/nccl-tests-afd59ab-v2.1-region.patch"
P24="$REPO_ROOT/patches/nccl-tests-afd59ab-$V-stage2.patch"
[ -e "$DEST" ] && { echo "refusing: third_party/$DEST exists" >&2; exit 2; }

if [ -z "${NCCL_PKG_DIR:-}" ]; then
  NCCL_PKG_DIR="$("$PYTHON" -c 'import nvidia.nccl, os; print(os.path.dirname(nvidia.nccl.__file__) if nvidia.nccl.__file__ else list(nvidia.nccl.__path__)[0])')"
fi
mkdir -p nccl_home/lib
ln -sfn "$NCCL_PKG_DIR/include" nccl_home/include
ln -sf "$NCCL_PKG_DIR/lib/libnccl.so.2" nccl_home/lib/libnccl.so.2
ln -sf libnccl.so.2 nccl_home/lib/libnccl.so

git clone -q --no-hardlinks "$NCCL_TESTS_SRC" "$DEST"
git -C "$DEST" checkout -q "$NCCL_TESTS_COMMIT"
[ "$(git -C "$DEST" rev-parse HEAD)" = "$NCCL_TESTS_COMMIT" ] || { echo "wrong nccl-tests commit" >&2; exit 3; }
git -C "$DEST" remote set-url --push origin DISABLED_NO_PUSH
git -C "$DEST" apply --check "$P21" && git -C "$DEST" apply "$P21"
git -C "$DEST" apply --check "$P24" && git -C "$DEST" apply "$P24"
make -C "$DEST" -j"${BUILD_JOBS:-2}" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$PWD/nccl_home" \
     NVCC_GENCODE="$NVCC_GENCODE" BUILDDIR="$PWD/$DEST/build" src.build
{
  echo "utc $(date -u +%FT%TZ)"
  echo "nccl_tests_commit $(git -C "$DEST" rev-parse HEAD)"
  echo "patch_v2.1_sha256 $(sha256sum "$P21" | cut -d' ' -f1)"
  echo "patch_stage2_version $V"
  echo "patch_stage2_sha256 $(sha256sum "$P24" | cut -d' ' -f1)"
  echo "tree_diff_sha256 $(git -C "$DEST" diff | sha256sum | cut -d' ' -f1)"
  echo "common_cu_sha256 $(sha256sum "$DEST/src/common.cu" | cut -d' ' -f1)"
  echo "nvcc $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
  echo "gencode $NVCC_GENCODE"
  echo "libnccl_sha256 $(sha256sum "$(readlink -f nccl_home/lib/libnccl.so.2)" | cut -d' ' -f1)"
  for b in "$DEST"/build/*_perf; do echo "binary_sha256 $(basename "$b") $(sha256sum "$b" | cut -d' ' -f1)"; done
} > "$DEST/BUILD_RECORD.txt"
cat "$DEST/BUILD_RECORD.txt"
