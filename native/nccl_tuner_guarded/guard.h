#ifndef NCCLE_GUARD_H_
#define NCCLE_GUARD_H_
#include <stddef.h>
/* Returns 1 if the tuner file's #@scope header matches this process (rules may be used), else 0 with a reason. */
__attribute__((visibility("hidden"))) int nccleGuardCheck(const char* file, size_t nRanks, size_t nNodes,
                                                          int* allowRanges, char* why, size_t whylen);
#endif
