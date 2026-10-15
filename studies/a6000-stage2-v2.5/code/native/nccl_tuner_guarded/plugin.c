/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* nccl-energy-frontier v2.2 guarded tuner: NCCL 2.27.5 ext-tuner/example + the two v2.1 fixes
 * (patches/nccl-2.27.5-example-tuner-fixes.patch) + a startup scope guard (guard.c) + exact-size rules only
 * unless the experimental range opt-in is given both in the file and in the environment. Any guard failure
 * disables every rule: NCCL's own selection applies. */
#include "tuner.h"
#include "guard.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>

#define __hidden __attribute__ ((visibility("hidden")))
#define MAX_LINE_LENGTH 256

// CSV field indices for configuration parsing
// Format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
#define CONFIG_FIELD_COLLTYPE     0
#define CONFIG_FIELD_MINBYTES     1
#define CONFIG_FIELD_MAXBYTES     2
#define CONFIG_FIELD_ALGORITHM    3
#define CONFIG_FIELD_PROTOCOL     4
#define CONFIG_FIELD_CHANNELS     5
#define CONFIG_FIELD_NNODES       6
#define CONFIG_FIELD_NRANKS       7
#define CONFIG_FIELD_PIPEOPS      8  // Optional field
#define CONFIG_FIELD_REGBUFF      9  // Optional field

// Field count constants
#define CONFIG_FIELDS_REQUIRED    8   // Minimum required fields (up to nRanks)
#define CONFIG_FIELDS_WITH_PIPEOPS 9  // Fields including numPipeOps
#define CONFIG_FIELDS_WITH_REGBUFF 10 // Fields including both numPipeOps and regBuff
#define CONFIG_FIELDS_MAX         10  // Maximum number of fields supported

typedef struct {
  ncclFunc_t collType;
  size_t minBytes;
  size_t maxBytes;
  int algorithm;
  int protocol;
  int nChannels;
  int nNodes;
  int nRanks;
  int numPipeOps;
  int regBuff;
} TuningConfig;

typedef struct {
  TuningConfig* configs;  // Changed from static array to dynamic pointer
  int numConfigs;
  int maxConfigs;         // Added to track allocated size
  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
  int allowRanges;        /* experimental range rules (min_bytes != max_bytes) permitted */
  int skippedRanges;
  int invalidLine;        /* v2.2 release audit: first malformed rule line (0 = none); the file is then rejected */
  char invalidWhy[160];
} TunerContext;

// Parse collective type from string
static ncclFunc_t parseCollType(const char* str) {
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  return (ncclFunc_t)-1; // v2.2 release audit: unknown -> invalid (was: silently allreduce)
}

// Convert collective type to string
static const char* collTypeToString(ncclFunc_t collType) {
  switch (collType) {
    case ncclFuncBroadcast: return "broadcast";
    case ncclFuncReduce: return "reduce";
    case ncclFuncAllGather: return "allgather";
    case ncclFuncReduceScatter: return "reducescatter";
    case ncclFuncAllReduce: return "allreduce";
    default: return "unknown";
  }
}

// Parse algorithm from string
static int parseAlgorithm(const char* str) {
  if (strcmp(str, "tree") == 0) return NCCL_ALGO_TREE;
  if (strcmp(str, "ring") == 0) return NCCL_ALGO_RING;
  if (strcmp(str, "collnet_direct") == 0) return NCCL_ALGO_COLLNET_DIRECT;
  if (strcmp(str, "collnet_chain") == 0) return NCCL_ALGO_COLLNET_CHAIN;
  if (strcmp(str, "nvls") == 0) return NCCL_ALGO_NVLS;
  if (strcmp(str, "nvls_tree") == 0) return NCCL_ALGO_NVLS_TREE;
  if (strcmp(str, "pat") == 0) return NCCL_ALGO_PAT;
  return -1; // v2.2 release audit: unknown -> invalid (was: silently ring)
}

// Convert algorithm to string
static const char* algorithmToString(int algorithm) {
  switch (algorithm) {
    case NCCL_ALGO_TREE: return "tree";
    case NCCL_ALGO_RING: return "ring";
    case NCCL_ALGO_COLLNET_DIRECT: return "collnet_direct";
    case NCCL_ALGO_COLLNET_CHAIN: return "collnet_chain";
    case NCCL_ALGO_NVLS: return "nvls";
    case NCCL_ALGO_NVLS_TREE: return "nvls_tree";
    case NCCL_ALGO_PAT: return "pat";
    default: return "unknown";
  }
}

// Parse protocol from string
static int parseProtocol(const char* str) {
  if (strcmp(str, "ll") == 0) return NCCL_PROTO_LL;
  if (strcmp(str, "ll128") == 0) return NCCL_PROTO_LL128;
  if (strcmp(str, "simple") == 0) return NCCL_PROTO_SIMPLE;
  return -1; // v2.2 release audit: unknown -> invalid (was: silently simple)
}

/* strict decimal parsing: the whole token must be a number within [lo, hi] */
static int parseU64(const char* s, unsigned long long* out) {
  char* end;
  if (!s || !*s || *s == '-' || *s == '+') return 0;
  errno = 0;
  unsigned long long v = strtoull(s, &end, 10);
  if (errno || *end != '\0') return 0;
  *out = v;
  return 1;
}

/* Preserve empty CSV columns and reject excess columns, including a trailing comma.
 * No process-global tokenizer state: communicator initialization can be concurrent. */
static int splitFields(char* line, char** tokens, int maxFields) {
  int count = 0;
  char* next = line;
  for (;;) {
    if (count == maxFields) return -1;
    char* field = next;
    char* comma = strchr(field, ',');
    if (comma) { *comma = '\0'; next = comma + 1; }
    while (*field == ' ' || *field == '\t') field++;
    char* end = field + strlen(field);
    while (end > field && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r')) *--end = '\0';
    if (!*field) return -1;
    tokens[count++] = field;
    if (!comma) return count;
  }
}

static int parseInt(const char* s, long lo, long hi, int* out) {
  char* end;
  if (!s || !*s) return 0;
  errno = 0;
  long v = strtol(s, &end, 10);
  if (errno || *end != '\0' || v < lo || v > hi) return 0;
  *out = (int)v;
  return 1;
}

// Convert protocol to string
static const char* protocolToString(int protocol) {
  switch (protocol) {
    case NCCL_PROTO_LL: return "ll";
    case NCCL_PROTO_LL128: return "ll128";
    case NCCL_PROTO_SIMPLE: return "simple";
    default: return "unknown";
  }
}

// Helper function to count valid configuration lines in file
static int countConfigLines(const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    return 0;
  }

  char line[MAX_LINE_LENGTH];
  int count = 0;

  while (fgets(line, sizeof(line), file)) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Check if line has content
    if (strlen(line) > 0) {
      count++;
    }
  }

  fclose(file);
  return count;
}

// Load configuration from file
static ncclResult_t loadConfig(TunerContext* ctx, const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Config file %s not found, using defaults", filename);
    }
    return ncclSuccess; // Not finding config file is not an error
  }

  // First pass: count valid configuration lines
  int configCount = countConfigLines(filename);
  if (configCount == 0) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: No valid configurations found in %s", filename);
    }
    fclose(file);
    return ncclSuccess;
  }

  // Allocate memory for configurations based on actual count
  ctx->configs = (TuningConfig*)malloc(configCount * sizeof(TuningConfig));
  if (!ctx->configs) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Failed to allocate memory for %d configurations", configCount);
    }
    fclose(file);
    return ncclSystemError;
  }

  ctx->maxConfigs = configCount;
  ctx->numConfigs = 0;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Allocated memory for %d configurations", configCount);
  }

  // Reset file pointer to beginning
  fseek(file, 0, SEEK_SET);

  char line[MAX_LINE_LENGTH];
  int lineNo = 0;

  while (fgets(line, sizeof(line), file) && ctx->numConfigs < ctx->maxConfigs) {
    lineNo++;
    if (!strchr(line, '\n') && !feof(file)) {
      ctx->invalidLine = lineNo;
      snprintf(ctx->invalidWhy, sizeof(ctx->invalidWhy), "line %d: too long", lineNo);
      break;
    }
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    // Remove trailing newline
    line[strcspn(line, "\n")] = 0;

    // Parse CSV format: colltype,minbytes,maxbytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
    char* tokens[CONFIG_FIELDS_MAX];

    // Make a copy of the line for tokenizing
    char lineCopy[MAX_LINE_LENGTH];
    strncpy(lineCopy, line, sizeof(lineCopy));
    lineCopy[sizeof(lineCopy) - 1] = '\0';

    int tokenCount = splitFields(lineCopy, tokens, CONFIG_FIELDS_MAX);

    // Validate field count: support required fields (8), with pipeOps (9), or with regBuff (10)
    // v2.2 release audit: a rule line that does not parse completely rejects the whole file (fail closed); the
    // upstream example silently skipped short lines and mapped unknown names/numbers to defaults
    if (tokenCount < CONFIG_FIELDS_REQUIRED || tokenCount > CONFIG_FIELDS_MAX) {
      ctx->invalidLine = lineNo;
      snprintf(ctx->invalidWhy, sizeof(ctx->invalidWhy), "line %d: %d fields", lineNo, tokenCount);
      break;
    }
    {
      int coll = (int)parseCollType(tokens[CONFIG_FIELD_COLLTYPE]);
      int algo = parseAlgorithm(tokens[CONFIG_FIELD_ALGORITHM]);
      int proto = parseProtocol(tokens[CONFIG_FIELD_PROTOCOL]);
      unsigned long long mn = 0, mx = 0;
      int ch = 0, nn = 0, nr = 0, po = -1, rb = -1;
      if (coll < 0 || algo < 0 || proto < 0 ||
          !parseU64(tokens[CONFIG_FIELD_MINBYTES], &mn) || !parseU64(tokens[CONFIG_FIELD_MAXBYTES], &mx) || mn > mx ||
          mn > SIZE_MAX || mx > SIZE_MAX ||
          !parseInt(tokens[CONFIG_FIELD_CHANNELS], -1, 64, &ch) || ch == 0 ||
          !parseInt(tokens[CONFIG_FIELD_NNODES], -1, 1 << 20, &nn) ||
          !parseInt(tokens[CONFIG_FIELD_NRANKS], -1, 1 << 20, &nr) ||
          (tokenCount >= CONFIG_FIELDS_WITH_PIPEOPS && !parseInt(tokens[CONFIG_FIELD_PIPEOPS], -1, 1 << 20, &po)) ||
          (tokenCount >= CONFIG_FIELDS_WITH_REGBUFF && !parseInt(tokens[CONFIG_FIELD_REGBUFF], -1, 1, &rb))) {
        ctx->invalidLine = lineNo;
        snprintf(ctx->invalidWhy, sizeof(ctx->invalidWhy), "line %d: invalid field", lineNo);
        break;
      }
      if (!ctx->allowRanges && mn != mx) {
        ctx->skippedRanges++;   /* unmeasured sizes inside a range: not a validated profile entry */
        continue;
      }
      TuningConfig* config = &ctx->configs[ctx->numConfigs];
      config->collType = (ncclFunc_t)coll;
      config->minBytes = (size_t)mn;
      config->maxBytes = (size_t)mx;
      config->algorithm = algo;
      config->protocol = proto;
      config->nChannels = ch;
      config->nNodes = nn;
      config->nRanks = nr;
      config->numPipeOps = po;   // -1 (absent) means match any numPipeOps
      config->regBuff = rb;      // -1 (absent) means match any regBuff value

      ctx->numConfigs++;

      if (ctx->logFunction) {
        if (config->numPipeOps == -1 && config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks);
        } else if (config->regBuff == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=any",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps);
        } else if (config->numPipeOps == -1) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=any regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->regBuff);
        } else {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Loaded config: %s [%zu-%zu] %s/%s channels=%d nodes=%d ranks=%d pipeOps=%d regBuff=%d",
                           tokens[CONFIG_FIELD_COLLTYPE], config->minBytes, config->maxBytes,
                           tokens[CONFIG_FIELD_ALGORITHM], tokens[CONFIG_FIELD_PROTOCOL],
                           config->nChannels, config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
        }
      }
    }
  }

  fclose(file);
  if (ctx->invalidLine) {
    ctx->numConfigs = 0;     /* no partial rule set */
    return ncclSuccess;
  }
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: Loaded %d tuning configurations from %s", ctx->numConfigs, filename);
  }
  return ncclSuccess;
}

__hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) {
  TunerContext* ctx = (TunerContext*)malloc(sizeof(TunerContext));
  if (!ctx) return ncclSystemError;

  ctx->configs = NULL;     // Initialize to NULL
  ctx->numConfigs = 0;
  ctx->maxConfigs = 0;     // Initialize to 0
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;
  ctx->allowRanges = 0;
  ctx->skippedRanges = 0;
  ctx->invalidLine = 0;
  ctx->invalidWhy[0] = '\0';

  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/ExamplePlugin: Initializing tuner for %zu nodes, %zu ranks", nNodes, nRanks);
  }

  // Try to load config file from environment variable or default location
  const char* configFile = getenv("NCCL_TUNER_CONFIG_FILE");
  if (!configFile) {
    configFile = "nccl_tuner.conf"; // default config file name
  }

  char why[256];
  if (!nccleGuardCheck(configFile, nRanks, nNodes, &ctx->allowRanges, why, sizeof(why))) {
    if (logFunction) {
      logFunction(NCCL_LOG_INFO, NCCL_INIT | NCCL_TUNING, __FILE__, __LINE__,
                  "TUNER/NcclEnergyGuarded: DISABLED (%s); no rules loaded, NCCL defaults apply", why);
    }
    *context = ctx;
    return ncclSuccess;
  }

  ncclResult_t result = loadConfig(ctx, configFile);
  if (logFunction && ctx->invalidLine) {
    logFunction(NCCL_LOG_INFO, NCCL_INIT | NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/NcclEnergyGuarded: DISABLED (invalid rule, %s); no rules loaded, NCCL defaults apply",
                ctx->invalidWhy);
  } else if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_INIT | NCCL_TUNING, __FILE__, __LINE__,
                "TUNER/NcclEnergyGuarded: ENABLED scope OK; %d rules, %d range rules skipped, ranges=%d",
                ctx->numConfigs, ctx->skippedRanges, ctx->allowRanges);
  }
  if (result != ncclSuccess) {
    if (ctx->configs) {
      free(ctx->configs);  // Clean up allocated memory on error
    }
    free(ctx);
    return result;
  }

  *context = ctx;
  return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  TunerContext* ctx = (TunerContext*)context;
  if (!ctx) return ncclInternalError;
  if (ctx->numConfigs == 0) { *nChannels = 0; return ncclSuccess; }   /* disabled or empty: NCCL defaults */

  // nccl-energy-frontier fix 1: the upstream example sets *nChannels = 1 on every call, which forces one channel on
  // every collective (matched or not). NCCL treats 0 as "keep NCCL's own channel count" (src/enqueue.cc getAlgoInfo).
  *nChannels = 0;
  // nccl-energy-frontier fix 2: NCCL passes a contiguous float[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] array cast to
  // float** (src/enqueue.cc). Indexing it as float** dereferences cost values as pointers; NCCL itself casts it back.
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: pluginGetCollInfo called - collType=%s, nBytes=%zu, numPipeOps=%d, regBuff=%d, numConfigs=%d",
                     collTypeToString(collType), nBytes, numPipeOps, regBuff, ctx->numConfigs);
  }

  // Look for matching configuration
  for (int i = 0; i < ctx->numConfigs; i++) {
    TuningConfig* config = &ctx->configs[i];

    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER/ExamplePlugin: Checking config %d - collType=%s, minBytes=%zu, maxBytes=%zu, algo=%s, proto=%s, nNodes=%d, nRanks=%d, numPipeOps=%d, regBuff=%d",
                       i, collTypeToString(config->collType), config->minBytes, config->maxBytes, algorithmToString(config->algorithm), protocolToString(config->protocol),
                       config->nNodes, config->nRanks, config->numPipeOps, config->regBuff);
    }

    // Check if this config matches the current collective, size range, topology, pipeline ops, and regBuff
    if (config->collType == collType &&
        nBytes >= config->minBytes &&
        nBytes <= config->maxBytes &&
        (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes) &&
        (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks) &&
        (config->numPipeOps == -1 || config->numPipeOps == numPipeOps) &&
        (config->regBuff == -1 || config->regBuff == regBuff)) {

      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config matches. Applying algo=%s, proto=%s, channels=%d",
                         algorithmToString(config->algorithm), protocolToString(config->protocol), config->nChannels);
      }

      // Check bounds
      if (config->algorithm < numAlgo && config->protocol < numProto) {
        if (table[config->algorithm][config->protocol] != NCCL_ALGO_PROTO_IGNORE) {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Setting cost table[%s][%s] (%p) = 0.0 (was %.1f)",
                             algorithmToString(config->algorithm), protocolToString(config->protocol),
                             &table[config->algorithm][config->protocol], table[config->algorithm][config->protocol]);
          }
          table[config->algorithm][config->protocol] = 0.0; // Set low cost to prefer this configuration

          // Only override channels if not set to -1 (keep default)
          if (config->nChannels != -1) {
            *nChannels = config->nChannels;
          }

          if (ctx->logFunction) {
            if (config->nChannels == -1) {
              ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=default (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nNodes, config->nRanks);
            } else {
              ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                               "TUNER/ExamplePlugin: Applied config for collType=%s, bytes=%zu, pipeOps=%d, regBuff=%d: algo=%s, proto=%s, channels=%d (nodes=%d, ranks=%d)",
                               collTypeToString(config->collType), nBytes, numPipeOps, regBuff, algorithmToString(config->algorithm), protocolToString(config->protocol),
                               config->nChannels, config->nNodes, config->nRanks);
            }
          }
          return ncclSuccess;
        } else {
          if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "TUNER/ExamplePlugin: Algorithm/protocol combination [%s][%s] is marked as IGNORE",
                             algorithmToString(config->algorithm), protocolToString(config->protocol));
          }
        }
      } else {
        if (ctx->logFunction) {
          ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER/ExamplePlugin: Algorithm/protocol out of bounds - algo=%s (max %d), proto=%s (max %d)",
                           algorithmToString(config->algorithm), numAlgo, protocolToString(config->protocol), numProto);
        }
      }
    } else {
      if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "TUNER/ExamplePlugin: Config does not match - collType match=%d, size match=%d, nodes match=%d, ranks match=%d, pipeOps match=%d, regBuff match=%d",
                         config->collType == collType,
                         (nBytes >= config->minBytes && nBytes <= config->maxBytes),
                         (config->nNodes == -1 || config->nNodes == (int)ctx->nNodes),
                         (config->nRanks == -1 || config->nRanks == (int)ctx->nRanks),
                         (config->numPipeOps == -1 || config->numPipeOps == numPipeOps),
                         (config->regBuff == -1 || config->regBuff == regBuff));
      }
    }
  }

  // If no specific config found, apply default behavior
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "TUNER/ExamplePlugin: No matching config found");
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginDestroy(void* context) {
  if (context) {
    TunerContext* ctx = (TunerContext*)context;
    if (ctx->configs) {
      free(ctx->configs);  // Free dynamically allocated configs array
    }
    free(context);
  }
  return ncclSuccess;
}

#define PLUGIN_NAME "NcclEnergyGuarded"

const ncclTuner_v4_t ncclTunerPlugin_v4 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};
