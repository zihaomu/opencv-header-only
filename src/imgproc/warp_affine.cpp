#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

void warp_affine_backend_impl(const Mat& src,
                              Mat& dst,
                              const Mat& M,
                              Size dsize,
                              int flags,
                              int borderMode,
                              const Scalar& borderValue)
{
    warpAffine_fallback(src, dst, M, dsize, flags, borderMode, borderValue);
}

} // namespace detail
} // namespace cvh
