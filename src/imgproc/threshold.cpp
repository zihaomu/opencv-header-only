#include "fastpath_common.h"

namespace cvh
{
namespace detail
{

namespace
{
bool try_threshold_fastpath_f32(const Mat& src, Mat& dst, double thresh, double maxval, int type, double* out_ret)
{
    if (out_ret == nullptr)
    {
        return false;
    }

    if (src.empty() || src.depth() != CV_32F)
    {
        return false;
    }

    const bool is_dryrun = (type & THRESH_DRYRUN) != 0;
    type &= ~THRESH_DRYRUN;

    const int automatic_thresh = type & (~THRESH_MASK);
    const int thresh_type = type & THRESH_MASK;

    if (automatic_thresh == THRESH_OTSU || automatic_thresh == THRESH_TRIANGLE)
    {
        CV_Error(Error::StsBadArg, "threshold: OTSU/TRIANGLE requires CV_8UC1 source");
    }

    if (automatic_thresh != 0)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported automatic threshold flag=%d", automatic_thresh));
    }

    if (thresh_type != THRESH_BINARY &&
        thresh_type != THRESH_BINARY_INV &&
        thresh_type != THRESH_TRUNC &&
        thresh_type != THRESH_TOZERO &&
        thresh_type != THRESH_TOZERO_INV)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
    }

    const float thresh_f = static_cast<float>(thresh);
    const float max_f = static_cast<float>(maxval);
    *out_ret = thresh;

    if (is_dryrun)
    {
        return true;
    }

    dst.create(src.dims, src.size.p, src.type());

    const std::size_t scalar_count = src.total() * static_cast<std::size_t>(src.channels());
    const float* src_ptr = reinterpret_cast<const float*>(src.data);
    float* dst_ptr = reinterpret_cast<float*>(dst.data);

    if (src.isContinuous() && dst.isContinuous())
    {
        const bool do_parallel = should_parallelize_threshold_contiguous(scalar_count);
        parallel_for_count_if(do_parallel, scalar_count, [&](std::size_t i) {
            const float s = src_ptr[i];
            const bool cond = s > thresh_f;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_ptr[i] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                dst_ptr[i] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                dst_ptr[i] = cond ? thresh_f : s;
                break;
            case THRESH_TOZERO:
                dst_ptr[i] = cond ? s : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                dst_ptr[i] = cond ? 0.0f : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        });
        return true;
    }

    CV_Assert(src.dims == 2 && "threshold: non-contiguous path supports 2D Mat only");
    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const std::size_t src_step = src.step(0);
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel = should_parallelize_threshold_rows(rows, cols_scalar);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src.data + static_cast<std::size_t>(y) * src_step);
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        for (int x = 0; x < cols_scalar; ++x)
        {
            const float s = src_row[x];
            const bool cond = s > thresh_f;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_row[x] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                dst_row[x] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                dst_row[x] = cond ? thresh_f : s;
                break;
            case THRESH_TOZERO:
                dst_row[x] = cond ? s : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                dst_row[x] = cond ? 0.0f : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        }
    });

    return true;
}


} // namespace

double threshold_backend_impl(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    double ret_value = 0.0;
    if (try_threshold_fastpath_f32(src, dst, thresh, maxval, type, &ret_value))
    {
        return ret_value;
    }

    return threshold_fallback(src, dst, thresh, maxval, type);
}

} // namespace detail
} // namespace cvh
