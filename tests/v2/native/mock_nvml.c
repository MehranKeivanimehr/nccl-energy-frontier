/* Test double for libnvidia-ml (CPU only). Driven by environment variables:
 *   MOCK_NVML_FAIL_INIT=1        nvmlInit_v2 fails
 *   MOCK_NVML_DRIVER=580.178.04  driver version string
 *   MOCK_NVML_GPUS="GPU-a=NVIDIA RTX A6000@00000000:81:00.0;GPU-b=...@..."
 * It implements only the read-only queries the guarded tuner uses; there is no energy counter. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int nvmlReturn_t;
typedef struct { char busIdLegacy[16]; unsigned int domain, bus, device, pciDeviceId, pciSubSystemId; char busId[32]; } nvmlPciInfo_t;
typedef struct { char uuid[96]; char name[96]; char bus[32]; } dev_t_;
static dev_t_ devs[16];
static int ndev = -1;
static void load(void) {
  if (ndev >= 0) return;
  ndev = 0;
  const char* g = getenv("MOCK_NVML_GPUS");
  if (!g) return;
  char buf[2048]; snprintf(buf, sizeof(buf), "%s", g);
  for (char* t = strtok(buf, ";"); t && ndev < 16; t = strtok(NULL, ";")) {
    char* eq = strchr(t, '='); char* at = eq ? strchr(eq, '@') : NULL;
    if (!eq || !at) continue;
    *eq = 0; *at = 0;
    snprintf(devs[ndev].uuid, 96, "%s", t); snprintf(devs[ndev].name, 96, "%s", eq + 1); snprintf(devs[ndev].bus, 32, "%s", at + 1);
    ndev++;
  }
}
nvmlReturn_t nvmlInit_v2(void) { const char* f = getenv("MOCK_NVML_FAIL_INIT"); load(); return f && *f == '1' ? 999 : 0; }
nvmlReturn_t nvmlShutdown(void) { return 0; }
nvmlReturn_t nvmlSystemGetDriverVersion(char* v, unsigned int n) { const char* d = getenv("MOCK_NVML_DRIVER"); if (!d) return 3; snprintf(v, n, "%s", d); return 0; }
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, void** dev) {
  load(); for (int i = 0; i < ndev; i++) if (strcmp(devs[i].uuid, uuid) == 0) { *dev = &devs[i]; return 0; } return 13; }
nvmlReturn_t nvmlDeviceGetName(void* dev, char* name, unsigned int n) { snprintf(name, n, "%s", ((dev_t_*)dev)->name); return 0; }
nvmlReturn_t nvmlDeviceGetPciInfo_v3(void* dev, nvmlPciInfo_t* p) { memset(p, 0, sizeof(*p)); snprintf(p->busId, 32, "%s", ((dev_t_*)dev)->bus); return 0; }
/* the guarded tuner must never touch the energy counter */
nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(void* dev, unsigned long long* e) { fprintf(stderr, "MOCK: COUNTER CALLED\n"); abort(); }
