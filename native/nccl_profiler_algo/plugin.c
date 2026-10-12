/* nccl-energy-frontier: minimal NCCL profiler plugin (profiler API v4, NCCL 2.27.x).
 *
 * Records, for every collective a rank issues, the configuration NCCL actually
 * selected: (func, algo, proto, nChannels, nWarps, count, datatype), as a
 * histogram with first/last sequence numbers. Nothing is timed and no child
 * events are requested, so the per-collective cost is one small locked update.
 * At communicator finalize one JSON file per rank is written to
 * $NCCLE_PROFILER_DIR/prof_<pid>_r<rank>_<commHash>.json.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "profiler.h"

#define MAXK 128
typedef struct {
  char func[32], algo[32], proto[32], dtype[32];
  int nChannels, nWarps;
  size_t count;
  unsigned long long n, seqFirst, seqLast;
} Key;

typedef struct {
  pthread_mutex_t mu;
  Key k[MAXK];
  int nk;
  unsigned long long overflow, groups;
  int rank, nranks, nNodes;
  uint64_t commHash;
  char commName[128];
} Ctx;

static int groupDummy;

static void cp(char* dst, const char* src) { snprintf(dst, 32, "%s", src ? src : "null"); }

static ncclResult_t pInit(void** context, int* eActivationMask, const char* commName, uint64_t commHash,
                          int nNodes, int nranks, int rank, ncclDebugLogger_t logfn) {
  (void)logfn;
  Ctx* c = (Ctx*)calloc(1, sizeof(Ctx));
  if (!c) return ncclSystemError;
  pthread_mutex_init(&c->mu, NULL);
  c->rank = rank; c->nranks = nranks; c->nNodes = nNodes; c->commHash = commHash;
  snprintf(c->commName, sizeof(c->commName), "%s", commName ? commName : "");
  *eActivationMask = ncclProfileColl;  /* NCCL starts the parent group event itself */
  *context = c;
  return ncclSuccess;
}

static ncclResult_t pStart(void* context, void** eHandle, ncclProfilerEventDescr_v4_t* d) {
  Ctx* c = (Ctx*)context;
  *eHandle = NULL;
  if (d->type == ncclProfileGroup) {          /* must be non-NULL or NCCL emits no coll events */
    *eHandle = &groupDummy;
    __atomic_fetch_add(&c->groups, 1, __ATOMIC_RELAXED);
    return ncclSuccess;
  }
  if (d->type != ncclProfileColl) return ncclSuccess;
  pthread_mutex_lock(&c->mu);
  int i;
  for (i = 0; i < c->nk; i++) {
    Key* k = &c->k[i];
    if (k->count == d->coll.count && k->nChannels == d->coll.nChannels && k->nWarps == d->coll.nWarps &&
        !strcmp(k->func, d->coll.func) && !strcmp(k->algo, d->coll.algo) && !strcmp(k->proto, d->coll.proto) &&
        !strcmp(k->dtype, d->coll.datatype))
      break;
  }
  if (i == c->nk) {
    if (c->nk == MAXK) { c->overflow++; pthread_mutex_unlock(&c->mu); return ncclSuccess; }
    Key* k = &c->k[c->nk++];
    cp(k->func, d->coll.func); cp(k->algo, d->coll.algo); cp(k->proto, d->coll.proto); cp(k->dtype, d->coll.datatype);
    k->nChannels = d->coll.nChannels; k->nWarps = d->coll.nWarps; k->count = d->coll.count;
    k->seqFirst = d->coll.seqNumber;
  }
  c->k[i].n++;
  c->k[i].seqLast = d->coll.seqNumber;
  pthread_mutex_unlock(&c->mu);
  return ncclSuccess;
}

static ncclResult_t pStop(void* eHandle) { (void)eHandle; return ncclSuccess; }

static ncclResult_t pRecord(void* eHandle, ncclProfilerEventState_v4_t s, ncclProfilerEventStateArgs_v4_t* a) {
  (void)eHandle; (void)s; (void)a; return ncclSuccess;
}

static ncclResult_t pFinalize(void* context) {
  Ctx* c = (Ctx*)context;
  const char* dir = getenv("NCCLE_PROFILER_DIR");
  if (dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/prof_%d_r%d_%016llx.json", dir, (int)getpid(), c->rank,
             (unsigned long long)c->commHash);
    FILE* f = fopen(path, "w");
    if (f) {
      fprintf(f, "{\"schema\":\"nccle-profiler-v1\",\"plugin\":\"nccle-algo-v1\",\"pid\":%d,\"rank\":%d,"
                 "\"nranks\":%d,\"nNodes\":%d,\"commHash\":\"%016llx\",\"groups\":%llu,\"overflow\":%llu,\"colls\":[",
              (int)getpid(), c->rank, c->nranks, c->nNodes, (unsigned long long)c->commHash, c->groups, c->overflow);
      for (int i = 0; i < c->nk; i++) {
        Key* k = &c->k[i];
        fprintf(f, "%s{\"func\":\"%s\",\"algo\":\"%s\",\"proto\":\"%s\",\"nChannels\":%d,\"nWarps\":%d,"
                   "\"count\":%zu,\"datatype\":\"%s\",\"n\":%llu,\"seq_first\":%llu,\"seq_last\":%llu}",
                i ? "," : "", k->func, k->algo, k->proto, k->nChannels, k->nWarps, k->count, k->dtype,
                k->n, k->seqFirst, k->seqLast);
      }
      fprintf(f, "]}\n");
      fclose(f);
    }
  }
  pthread_mutex_destroy(&c->mu);
  free(c);
  return ncclSuccess;
}

ncclProfiler_v4_t ncclProfiler_v4 = { "nccle-algo-v1", pInit, pStart, pStop, pRecord, pFinalize };
