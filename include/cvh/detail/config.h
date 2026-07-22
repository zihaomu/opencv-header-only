#ifndef CVH_DETAIL_CONFIG_H
#define CVH_DETAIL_CONFIG_H

// Mode contract:
// - CVH_LITE: header-only fallback mode
// - CVH_NATIVE: linked native backend-enhanced mode
//
// If neither is provided by build flags, default to LITE so plain header users
// get a runnable baseline contract by default.
#if defined(CVH_FULL) && !defined(CVH_NATIVE)
#define CVH_NATIVE 1
#endif

#if defined(CVH_LITE) && defined(CVH_NATIVE)
#error "CVH_LITE and CVH_NATIVE cannot be enabled at the same time"
#endif

#if !defined(CVH_LITE) && !defined(CVH_NATIVE)
#define CVH_LITE 1
#endif

// Optional capability toggles. These do not switch mode; they only enable
// enhancements for codepaths that support them.
#ifndef CVH_ENABLE_XSIMD
#define CVH_ENABLE_XSIMD 0
#endif

#ifndef CVH_ENABLE_THREADS
#define CVH_ENABLE_THREADS 0
#endif

#ifndef CVH_ENABLE_FAST_MATH
#define CVH_ENABLE_FAST_MATH 0
#endif

#endif  // CVH_DETAIL_CONFIG_H
