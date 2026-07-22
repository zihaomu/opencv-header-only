#include "cvh/cvh.h"

#if defined(CVH_EXPECT_NATIVE)
#ifndef CVH_NATIVE
#error "Expected CVH_NATIVE to be enabled for native mode smoke target"
#endif
#ifdef CVH_LITE
#error "CVH_LITE must not be enabled together with CVH_NATIVE"
#endif
#else
#ifndef CVH_LITE
#error "Expected CVH_LITE default mode for header-only smoke target"
#endif
#ifdef CVH_NATIVE
#error "CVH_NATIVE must not be enabled for lite smoke target"
#endif
#endif

int main()
{
    return 0;
}
