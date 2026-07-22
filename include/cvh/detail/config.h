#ifndef CVH_DETAIL_CONFIG_H
#define CVH_DETAIL_CONFIG_H

// Public package contract:
// - cvh::headers: pure header-only baseline
// - cvh::headers_fast: pure header-only baseline plus accepted SIMD fast paths
//
// CVH_LITE remains the default compatibility macro for plain header users.
// CVH_NATIVE/CVH_FULL are legacy internal switches for development-only .cpp
// experiments and are not part of the installed package surface.
//
// If no legacy mode macro is provided, default to CVH_LITE so plain header users
// get a runnable baseline by default.
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

// xsimd is quarantined as a legacy/experimental adapter while P5 removes it
// from the public fast profile. Internal tests must opt in explicitly.
#ifndef CVH_ENABLE_LEGACY_XSIMD
#define CVH_ENABLE_LEGACY_XSIMD 0
#endif

#ifndef CVH_ENABLE_THREADS
#define CVH_ENABLE_THREADS 0
#endif

#ifndef CVH_ENABLE_FAST_MATH
#define CVH_ENABLE_FAST_MATH 0
#endif

#ifndef CVH_ENABLE_OPENCV_INTRIN
#define CVH_ENABLE_OPENCV_INTRIN 0
#endif

#ifndef CVH_ENABLE_PLATFORM_INTRINSICS
#define CVH_ENABLE_PLATFORM_INTRINSICS 0
#endif

#endif  // CVH_DETAIL_CONFIG_H
