/*************************************************************************
 * nccl-energy-frontier v2.2: startup scope guard for the NCCL tuner plugin.
 *
 * NCCL's v4 tuner interface passes the plugin only (nRanks, nNodes) at init and
 * (collective, nBytes, pipeline ops, regBuff) per call. It does not pass the
 * datatype, reduction op, GPU identities or topology. This guard therefore checks
 * at communicator init what the process can observe, and disables every rule
 * (NCCL defaults apply) if anything is missing, unknown or different:
 *   - scope header present in the tuner file (#@scope key=value lines)
 *   - nRanks / nNodes equal the profile; nRanks equals the number of visible GPUs
 *   - CUDA_VISIBLE_DEVICES lists GPU UUIDs (index lists cannot be verified)
 *   - NVML (read-only, never the energy counter): driver version, GPU model and the
 *     PCI bus id set of the visible GPUs equal the profile
 *   - ncclGetVersion() of the loaded NCCL equals the profile's version code
 *   - placement NCCL environment equals the profile; NCCL_ALGO / NCCL_PROTO unset
 *   - NCCLE_TUNER_DECLARED_SCOPE equals the profile's declared dtype/op/in-place
 *     scope. This is a DECLARATION by the controlled harness, an assumption that
 *     the plugin cannot verify per call; outside that harness it is not a safety
 *     guarantee.
 ************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include "guard.h"

#include <ctype.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int nvmlReturn_t;
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct {
  char busIdLegacy[16];
  unsigned int domain, bus, device, pciDeviceId, pciSubSystemId;
  char busId[32];
} nvmlPciInfo_t;

#define MAXKV 40
typedef struct { char key[64]; char val[512]; } kv_t;

static void trim(char* s) {
  size_t n = strlen(s);
  while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = 0;
}

static int readScope(const char* file, kv_t* kv, int max) {
  FILE* f = fopen(file, "r");
  if (!f) return -1;
  char line[640];
  int n = 0;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "#@scope ", 8) != 0) continue;
    if (n == max || (!strchr(line, '\n') && !feof(f))) { fclose(f); return -2; }
    trim(line);
    char* eq = strchr(line + 8, '=');
    if (!eq) { fclose(f); return -3; }
    *eq = 0;
    for (int i = 0; i < n; i++) {
      if (strcmp(kv[i].key, line + 8) == 0) { fclose(f); return -3; }
    }
    if (strlen(line + 8) >= sizeof(kv[n].key) || strlen(eq + 1) >= sizeof(kv[n].val)) {
      fclose(f);
      return -2;                     /* over-long scope line: never truncate, fail closed */
    }
    memcpy(kv[n].key, line + 8, strlen(line + 8) + 1);
    memcpy(kv[n].val, eq + 1, strlen(eq + 1) + 1);
    n++;
  }
  fclose(f);
  return n;
}

static const char* get(kv_t* kv, int n, const char* key) {
  for (int i = 0; i < n; i++) if (strcmp(kv[i].key, key) == 0) return kv[i].val;
  return NULL;
}

static void lower(char* s) { for (; *s; s++) *s = (char)tolower((unsigned char)*s); }

/* comma list membership with case-insensitive comparison */
static int sameSet(const char* declared, char found[][32], int nFound) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", declared);
  lower(buf);
  int nDecl = 0;
  int matched[16] = {0};
  if (nFound <= 0 || nFound > 16 || strlen(declared) >= sizeof(buf)) return 0;
  char* t = buf;
  for (;;) {
    char* comma = strchr(t, ',');
    if (comma) *comma = '\0';
    if (!*t) return 0;
    int hit = -1;
    for (int i = 0; i < nFound; i++) {
      char f[32];
      snprintf(f, sizeof(f), "%s", found[i]);
      lower(f);
      if (strcmp(f, t) == 0) {
        if (hit != -1 || matched[i]) return 0;
        hit = i;
      }
    }
    if (hit == -1) return 0;
    matched[hit] = 1;
    nDecl++;
    if (!comma) break;
    t = comma + 1;
  }
  return nDecl == nFound;
}

/* Variables that change NCCL's transport/topology choice: each must match the profile's placement_env, unset
 * when the profile does not set it. */
static const char* PLACEMENT_VARS[] = {"NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_P2P_LEVEL", "NCCL_NET",
                                       "NCCL_IB_DISABLE", "NCCL_NVLS_ENABLE", "NCCL_MIN_NCHANNELS",
                                       "NCCL_MAX_NCHANNELS", "NCCL_PROTO", "NCCL_ALGO", NULL};

static int checkEnv(const char* placementEnv, char* why, size_t whylen) {
  char decl[512];
  snprintf(decl, sizeof(decl), "%s", strcmp(placementEnv, "none") == 0 ? "" : placementEnv);
  for (int i = 0; PLACEMENT_VARS[i]; i++) {
    const char* var = PLACEMENT_VARS[i];
    const char* have = getenv(var);
    const char* want = NULL;
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", decl);
    char* save = NULL;
    for (char* t = strtok_r(tmp, ";", &save); t; t = strtok_r(NULL, ";", &save)) {
      char* eq = strchr(t, '=');
      if (eq && (size_t)(eq - t) == strlen(var) && strncmp(t, var, strlen(var)) == 0) want = eq + 1;
    }
    if (want == NULL && have != NULL) { snprintf(why, whylen, "ENV_SET_NOT_IN_PROFILE:%s", var); return 0; }
    if (want != NULL && (have == NULL || strcmp(have, want) != 0)) {
      snprintf(why, whylen, "ENV_MISMATCH:%s", var);
      return 0;
    }
  }
  return 1;
}

static const char* REQUIRED[] = {"method", "gpu_model", "driver_version", "nccl_version_code", "nranks", "nnodes",
                                 "pci_bus_ids", "placement_env", "declared_scope", "allow_ranges", NULL};

int nccleGuardCheck(const char* file, size_t nRanks, size_t nNodes, int* allowRanges, char* why, size_t whylen) {
  kv_t kv[MAXKV];
  *allowRanges = 0;
  int n = readScope(file, kv, MAXKV);
  if (n == -2) { snprintf(why, whylen, "SCOPE_LINE_TOO_LONG"); return 0; }
  if (n == -3) { snprintf(why, whylen, "SCOPE_MALFORMED_OR_DUPLICATE"); return 0; }
  if (n < 0) { snprintf(why, whylen, "CONFIG_UNREADABLE"); return 0; }
  for (int i = 0; REQUIRED[i]; i++) {
    const char* v = get(kv, n, REQUIRED[i]);
    if (!v || !*v || strcmp(v, "None") == 0) { snprintf(why, whylen, "SCOPE_MISSING:%s", REQUIRED[i]); return 0; }
  }
  if (strcmp(get(kv, n, "method"), "v2.2") != 0) { snprintf(why, whylen, "SCOPE_METHOD_UNKNOWN"); return 0; }
  long wantRanks = atol(get(kv, n, "nranks")), wantNodes = atol(get(kv, n, "nnodes"));
  if ((long)nRanks != wantRanks || (long)nNodes != wantNodes) {
    snprintf(why, whylen, "RANKS_OR_NODES:%zu/%zu!=%ld/%ld", nRanks, nNodes, wantRanks, wantNodes);
    return 0;
  }
  if (!checkEnv(get(kv, n, "placement_env"), why, whylen)) return 0;
  const char* ds = getenv("NCCLE_TUNER_DECLARED_SCOPE");
  if (!ds || strcmp(ds, get(kv, n, "declared_scope")) != 0) {
    snprintf(why, whylen, "DECLARED_SCOPE_MISSING_OR_DIFFERENT");
    return 0;
  }
  /* NCCL library version of this process */
  int (*getVersion)(int*) = (int (*)(int*))dlsym(RTLD_DEFAULT, "ncclGetVersion");
  int ver = -1;
  if (!getVersion || getVersion(&ver) != 0) { snprintf(why, whylen, "NCCL_VERSION_UNKNOWN"); return 0; }
  if (ver != atoi(get(kv, n, "nccl_version_code"))) { snprintf(why, whylen, "NCCL_VERSION:%d", ver); return 0; }
  /* visible GPUs must be named by UUID */
  const char* cvd = getenv("CUDA_VISIBLE_DEVICES");
  if (!cvd || !*cvd) { snprintf(why, whylen, "CUDA_VISIBLE_DEVICES_UNSET"); return 0; }
  char uuids[16][96];
  int nU = 0;
  char cbuf[1600];
  if (strlen(cvd) >= sizeof(cbuf)) { snprintf(why, whylen, "CUDA_VISIBLE_DEVICES_NOT_UUIDS"); return 0; }
  snprintf(cbuf, sizeof(cbuf), "%s", cvd);
  char* t = cbuf;
  for (;;) {
    char* comma = strchr(t, ',');
    if (comma) *comma = '\0';
    if (nU >= 16 || strncmp(t, "GPU-", 4) != 0 || strlen(t) <= 4 || strlen(t) >= sizeof(uuids[0])) {
      snprintf(why, whylen, "CUDA_VISIBLE_DEVICES_NOT_UUIDS"); return 0;
    }
    for (int i = 0; i < nU; i++) {
      if (strcmp(t, uuids[i]) == 0) { snprintf(why, whylen, "CUDA_VISIBLE_DEVICES_NOT_UUIDS"); return 0; }
    }
    snprintf(uuids[nU++], sizeof(uuids[0]), "%s", t);
    if (!comma) break;
    t = comma + 1;
  }
  if (wantNodes != 1) { snprintf(why, whylen, "MULTI_NODE_PROFILE_UNSUPPORTED"); return 0; }
  if (nU != wantRanks) {                 /* single node: the communicator must span exactly the visible GPUs */
    snprintf(why, whylen, "VISIBLE_GPUS:%d!=nranks", nU);
    return 0;
  }
  /* NVML, read-only queries only */
  const char* lib = getenv("NCCLE_NVML_LIBRARY");
  void* h = dlopen(lib ? lib : "libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) { snprintf(why, whylen, "NVML_UNAVAILABLE"); return 0; }
  nvmlReturn_t (*init)(void) = (nvmlReturn_t (*)(void))dlsym(h, "nvmlInit_v2");
  nvmlReturn_t (*shut)(void) = (nvmlReturn_t (*)(void))dlsym(h, "nvmlShutdown");
  nvmlReturn_t (*drv)(char*, unsigned) = (nvmlReturn_t (*)(char*, unsigned))dlsym(h, "nvmlSystemGetDriverVersion");
  nvmlReturn_t (*byUuid)(const char*, nvmlDevice_t*) =
      (nvmlReturn_t (*)(const char*, nvmlDevice_t*))dlsym(h, "nvmlDeviceGetHandleByUUID");
  nvmlReturn_t (*name)(nvmlDevice_t, char*, unsigned) =
      (nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned))dlsym(h, "nvmlDeviceGetName");
  nvmlReturn_t (*pci)(nvmlDevice_t, nvmlPciInfo_t*) =
      (nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t*))dlsym(h, "nvmlDeviceGetPciInfo_v3");
  int ok = 0;
  if (!init || !shut || !drv || !byUuid || !name || !pci || init() != 0) {
    snprintf(why, whylen, "NVML_INIT");
    dlclose(h);
    return 0;
  }
  char d[96] = {0};
  char found[16][32];
  if (drv(d, sizeof(d)) != 0 || strcmp(d, get(kv, n, "driver_version")) != 0) {
    snprintf(why, whylen, "DRIVER_VERSION:%s", d);
    goto done;
  }
  for (int i = 0; i < nU; i++) {
    nvmlDevice_t dev;
    char nm[96] = {0};
    nvmlPciInfo_t pi;
    memset(&pi, 0, sizeof(pi));
    if (byUuid(uuids[i], &dev) != 0 || name(dev, nm, sizeof(nm)) != 0 || pci(dev, &pi) != 0) {
      snprintf(why, whylen, "NVML_DEVICE_QUERY");
      goto done;
    }
    if (strcmp(nm, get(kv, n, "gpu_model")) != 0) { snprintf(why, whylen, "GPU_MODEL:%s", nm); goto done; }
    snprintf(found[i], sizeof(found[i]), "%s", pi.busId);
  }
  if (!sameSet(get(kv, n, "pci_bus_ids"), found, nU)) { snprintf(why, whylen, "PCI_BUS_IDS"); goto done; }
  *allowRanges = strcmp(get(kv, n, "allow_ranges"), "1") == 0 && getenv("NCCLE_TUNER_ALLOW_RANGES") &&
                 strcmp(getenv("NCCLE_TUNER_ALLOW_RANGES"), "1") == 0;
  snprintf(why, whylen, "OK");
  ok = 1;
done:
  shut();
  dlclose(h);
  return ok;
}
