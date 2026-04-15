#ifndef CVH_OPENMP_UTILS_H
#define CVH_OPENMP_UTILS_H

// Backward-compatibility shim:
// Existing kernels include this file for should_parallelize_1d_loop().
// The implementation now lives in parallel_runtime.h and is backend-agnostic.

#include "parallel_runtime.h"

#endif  // CVH_OPENMP_UTILS_H
