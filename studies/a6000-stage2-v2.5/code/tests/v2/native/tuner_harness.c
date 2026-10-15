/* Loads a tuner plugin the way NCCL does and reports what one getCollInfo call changes (CPU only).
 * usage: tuner_harness <plugin.so> <nRanks> <nNodes> <coll 0..4> <nBytes>
 * Exports ncclGetVersion (value from MOCK_NCCL_VERSION, unset = symbol reports failure). */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "tuner.h"
__attribute__((visibility("default"))) ncclResult_t ncclGetVersion(int* v) {
  const char* e = getenv("MOCK_NCCL_VERSION");
  if (!e) return ncclInternalError;
  *v = atoi(e);
  return ncclSuccess;
}
static void logger(ncclDebugLogLevel level, unsigned long flags, const char* file, int line, const char* fmt, ...) {
  va_list ap; va_start(ap, fmt); fprintf(stderr, "LOG: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n"); va_end(ap);
}
int main(int argc, char** argv) {
  if (argc < 6) return 2;
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
  ncclTuner_v4_t* t = (ncclTuner_v4_t*)dlsym(h, "ncclTunerPlugin_v4");
  if (!t) return 4;
  void* ctx = NULL;
  if (t->init((size_t)atol(argv[2]), (size_t)atol(argv[3]), logger, &ctx) != ncclSuccess) { printf("INIT_ERROR\n"); return 5; }
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) table[a][p] = 1.0f;
  int nCh = -7;
  t->getCollInfo(ctx, (ncclFunc_t)atoi(argv[4]), (size_t)atoll(argv[5]), 1, (float**)table, NCCL_NUM_ALGORITHMS,
                 NCCL_NUM_PROTOCOLS, 0, &nCh);
  int hit = 0;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
    if (table[a][p] == 0.0f) { printf("RULE algo=%d proto=%d nChannels=%d\n", a, p, nCh); hit = 1; }
  if (!hit) printf("NO_RULE nChannels=%d\n", nCh);
  t->destroy(ctx);
  return 0;
}
