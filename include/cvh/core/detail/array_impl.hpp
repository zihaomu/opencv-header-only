#ifndef CVH_CORE_DETAIL_ARRAY_IMPL_HPP
#define CVH_CORE_DETAIL_ARRAY_IMPL_HPP

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

namespace cvh
{
namespace array_detail
{

inline std::vector<int> coordinates_from_linear(const Mat& mat, size_t linear)
{
    std::vector<int> coordinates(static_cast<size_t>(mat.dims), 0);
    for (int dim = mat.dims - 1; dim >= 0; --dim)
    {
        const size_t extent = static_cast<size_t>(mat.size.p[dim]);
        coordinates[static_cast<size_t>(dim)] =
            static_cast<int>(linear % extent);
        linear /= extent;
    }
    return coordinates;
}

inline size_t pixel_offset(const Mat& mat, const std::vector<int>& coordinates)
{
    size_t offset = 0;
    for (int dim = 0; dim < mat.dims; ++dim)
    {
        offset += static_cast<size_t>(coordinates[static_cast<size_t>(dim)]) *
                  mat.step(dim);
    }
    return offset;
}

inline size_t pixel_offset_from_linear(const Mat& mat, size_t linear)
{
    size_t offset = 0;
    for (int dim = mat.dims - 1; dim >= 0; --dim)
    {
        const size_t extent = static_cast<size_t>(mat.size.p[dim]);
        const size_t coordinate = linear % extent;
        linear /= extent;
        offset += coordinate * mat.step(dim);
    }
    return offset;
}

inline const uchar* pixel_at(const Mat& mat, size_t linear)
{
    return mat.data + pixel_offset_from_linear(mat, linear);
}

inline uchar* pixel_at(Mat& mat, size_t linear)
{
    return mat.data + pixel_offset_from_linear(mat, linear);
}

inline bool shares_storage(const Mat& a, const Mat& b)
{
    return !a.empty() && !b.empty() && a.u != nullptr && a.u == b.u;
}

inline Mat alias_safe_source(const Mat& src, const Mat& dst)
{
    return shares_storage(src, dst) && src.data != dst.data ? src.clone() : src;
}

inline void validate_copy_mask(const Mat& src, const Mat& mask)
{
    if (!mask.empty() &&
        (mask.type() != CV_8UC1 || mask.shape() != src.shape()))
    {
        CV_Error(
            Error::StsBadArg,
            "copyTo mask must be CV_8UC1 with the same shape as src");
    }
}

inline int total_channels(const Mat* mats, size_t count)
{
    int total = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (mats[i].channels() > std::numeric_limits<int>::max() - total)
        {
            CV_Error(Error::StsOutOfRange, "mixChannels channel count overflow");
        }
        total += mats[i].channels();
    }
    return total;
}

inline std::pair<size_t, int> resolve_channel(const Mat* mats,
                                              size_t count,
                                              int channel)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (channel < mats[i].channels())
        {
            return {i, channel};
        }
        channel -= mats[i].channels();
    }
    CV_Error(Error::StsOutOfRange, "mixChannels channel index is out of range");
    return {0, 0};
}

inline void validate_same_geometry_and_depth(const Mat& reference,
                                             const Mat& candidate,
                                             const char* fn_name)
{
    if (candidate.empty() || candidate.shape() != reference.shape() ||
        candidate.depth() != reference.depth())
    {
        CV_Error_(Error::StsBadArg,
                  ("%s inputs and outputs must have the same shape and depth",
                   fn_name));
    }
}

inline std::vector<Mat> snapshot_sources(const Mat* src, size_t count)
{
    if (src == nullptr || count == 0)
    {
        CV_Error(Error::StsBadArg, "concat expects at least one input");
    }
    return std::vector<Mat>(src, src + count);
}

}  // namespace array_detail

inline int borderInterpolate(int p, int len, int borderType)
{
    if (len <= 0)
    {
        CV_Error(Error::StsBadSize, "borderInterpolate expects len > 0");
    }
    borderType &= ~BORDER_ISOLATED;
    if (static_cast<unsigned>(p) < static_cast<unsigned>(len))
    {
        return p;
    }
    if (borderType == BORDER_REPLICATE)
    {
        return p < 0 ? 0 : len - 1;
    }
    if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        if (len == 1)
        {
            return 0;
        }
        const int delta = borderType == BORDER_REFLECT_101 ? 1 : 0;
        do
        {
            if (p < 0)
            {
                p = -p - 1 + delta;
            }
            else
            {
                p = len - 1 - (p - len) - delta;
            }
        }
        while (static_cast<unsigned>(p) >= static_cast<unsigned>(len));
        return p;
    }
    if (borderType == BORDER_WRAP)
    {
        if (p < 0)
        {
            p -= ((p - len + 1) / len) * len;
        }
        if (p >= len)
        {
            p %= len;
        }
        return p;
    }
    if (borderType == BORDER_CONSTANT)
    {
        return -1;
    }
    CV_Error_(Error::StsBadArg,
              ("borderInterpolate unsupported borderType=%d", borderType));
    return -1;
}

inline void copyTo(const Mat& src, Mat& dst, const Mat& mask)
{
    if (src.empty())
    {
        dst.release();
        return;
    }
    array_detail::validate_copy_mask(src, mask);
    if (mask.empty())
    {
        if (src.data == dst.data)
        {
            return;
        }
        const Mat source = array_detail::alias_safe_source(src, dst);
        source.copyTo(dst);
        return;
    }
    if (src.data == dst.data)
    {
        return;
    }

    const Mat source = array_detail::alias_safe_source(src, dst);
    const bool allocate =
        dst.empty() || dst.type() != source.type() ||
        dst.shape() != source.shape();
    if (allocate)
    {
        dst.create(source.dims, source.size.p, source.type());
        dst.setTo(Scalar::all(0.0));
    }
    for (size_t i = 0; i < source.total(); ++i)
    {
        if (*array_detail::pixel_at(mask, i) != 0)
        {
            std::memcpy(
                array_detail::pixel_at(dst, i),
                array_detail::pixel_at(source, i),
                source.elemSize());
        }
    }
}

inline void mixChannels(const Mat* src,
                        size_t nsrcs,
                        Mat* dst,
                        size_t ndsts,
                        const int* fromTo,
                        size_t npairs)
{
    if (src == nullptr || nsrcs == 0 || dst == nullptr || ndsts == 0)
    {
        CV_Error(Error::StsBadArg, "mixChannels expects non-empty arrays");
    }
    if (npairs != 0 && fromTo == nullptr)
    {
        CV_Error(Error::StsBadArg, "mixChannels fromTo must not be null");
    }
    const Mat& reference = src[0];
    if (reference.empty())
    {
        CV_Error(Error::StsBadArg, "mixChannels expects non-empty inputs");
    }
    for (size_t i = 0; i < nsrcs; ++i)
    {
        array_detail::validate_same_geometry_and_depth(
            reference, src[i], "mixChannels");
    }
    for (size_t i = 0; i < ndsts; ++i)
    {
        array_detail::validate_same_geometry_and_depth(
            reference, dst[i], "mixChannels");
    }

    const int source_channels = array_detail::total_channels(src, nsrcs);
    const int destination_channels = array_detail::total_channels(dst, ndsts);
    std::vector<Mat> source_snapshots(src, src + nsrcs);
    for (size_t i = 0; i < nsrcs; ++i)
    {
        for (size_t j = 0; j < ndsts; ++j)
        {
            if (array_detail::shares_storage(src[i], dst[j]))
            {
                source_snapshots[i] = src[i].clone();
                break;
            }
        }
    }

    const size_t scalar_bytes = reference.elemSize1();
    for (size_t pair = 0; pair < npairs; ++pair)
    {
        const int from = fromTo[2 * pair];
        const int to = fromTo[2 * pair + 1];
        if (to < 0 || to >= destination_channels || from >= source_channels)
        {
            CV_Error(Error::StsOutOfRange, "mixChannels route is out of range");
        }
        const auto destination =
            array_detail::resolve_channel(dst, ndsts, to);
        std::pair<size_t, int> source{0, 0};
        if (from >= 0)
        {
            source = array_detail::resolve_channel(
                source_snapshots.data(), nsrcs, from);
        }
        for (size_t pixel = 0; pixel < reference.total(); ++pixel)
        {
            uchar* destination_pixel =
                array_detail::pixel_at(dst[destination.first], pixel);
            uchar* destination_scalar =
                destination_pixel +
                static_cast<size_t>(destination.second) * scalar_bytes;
            if (from < 0)
            {
                std::memset(destination_scalar, 0, scalar_bytes);
            }
            else
            {
                const uchar* source_pixel =
                    array_detail::pixel_at(
                        source_snapshots[source.first], pixel);
                std::memcpy(
                    destination_scalar,
                    source_pixel +
                        static_cast<size_t>(source.second) * scalar_bytes,
                    scalar_bytes);
            }
        }
    }
}

inline void mixChannels(const std::vector<Mat>& src,
                        std::vector<Mat>& dst,
                        const std::vector<int>& fromTo)
{
    if ((fromTo.size() & 1u) != 0)
    {
        CV_Error(Error::StsBadArg, "mixChannels fromTo must contain pairs");
    }
    mixChannels(
        src.data(),
        src.size(),
        dst.data(),
        dst.size(),
        fromTo.data(),
        fromTo.size() / 2);
}

inline void extractChannel(const Mat& src, Mat& dst, int coi)
{
    if (src.empty() || coi < 0 || coi >= src.channels())
    {
        CV_Error(Error::StsOutOfRange, "extractChannel channel is out of range");
    }
    const Mat source = array_detail::alias_safe_source(src, dst);
    dst.create(
        source.dims,
        source.size.p,
        CV_MAKETYPE(source.depth(), 1));
    const int route[] = {coi, 0};
    mixChannels(&source, 1, &dst, 1, route, 1);
}

inline void insertChannel(const Mat& src, Mat& dst, int coi)
{
    if (src.empty() || src.channels() != 1 || dst.empty() ||
        src.shape() != dst.shape() || src.depth() != dst.depth() ||
        coi < 0 || coi >= dst.channels())
    {
        CV_Error(
            Error::StsBadArg,
            "insertChannel expects single-channel src and compatible dst/coi");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    const int route[] = {0, coi};
    mixChannels(&source, 1, &dst, 1, route, 1);
}

inline void flip(const Mat& src, Mat& dst, int flipCode)
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "flip expects non-empty 2D src");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    dst.create(source.dims, source.size.p, source.type());
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        const int source_y = flipCode == 0 || flipCode < 0
                                 ? source.size.p[0] - 1 - y
                                 : y;
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            const int source_x = flipCode > 0 || flipCode < 0
                                     ? source.size.p[1] - 1 - x
                                     : x;
            std::memcpy(
                dst.pixelPtr(y, x),
                source.pixelPtr(source_y, source_x),
                source.elemSize());
        }
    }
}

inline void flipND(const Mat& src, Mat& dst, int axis)
{
    if (src.empty())
    {
        CV_Error(Error::StsBadArg, "flipND expects non-empty src");
    }
    if (axis < -src.dims || axis >= src.dims)
    {
        CV_Error(Error::StsOutOfRange, "flipND axis is out of range");
    }
    if (axis < 0)
    {
        axis += src.dims;
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    dst.create(source.dims, source.size.p, source.type());
    for (size_t i = 0; i < source.total(); ++i)
    {
        std::vector<int> destination_coordinates =
            array_detail::coordinates_from_linear(dst, i);
        std::vector<int> source_coordinates = destination_coordinates;
        source_coordinates[static_cast<size_t>(axis)] =
            source.size.p[axis] - 1 -
            source_coordinates[static_cast<size_t>(axis)];
        std::memcpy(
            dst.data +
                array_detail::pixel_offset(dst, destination_coordinates),
            source.data +
                array_detail::pixel_offset(source, source_coordinates),
            source.elemSize());
    }
}

inline void rotate(const Mat& src, Mat& dst, int rotateCode)
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "rotate expects non-empty 2D src");
    }
    if (rotateCode != ROTATE_90_CLOCKWISE && rotateCode != ROTATE_180 &&
        rotateCode != ROTATE_90_COUNTERCLOCKWISE)
    {
        CV_Error(Error::StsBadArg, "rotate code is unsupported");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    const int source_rows = source.size.p[0];
    const int source_cols = source.size.p[1];
    const int destination_rows =
        rotateCode == ROTATE_180 ? source_rows : source_cols;
    const int destination_cols =
        rotateCode == ROTATE_180 ? source_cols : source_rows;
    dst.create({destination_rows, destination_cols}, source.type());
    for (int y = 0; y < destination_rows; ++y)
    {
        for (int x = 0; x < destination_cols; ++x)
        {
            int source_y = 0;
            int source_x = 0;
            if (rotateCode == ROTATE_90_CLOCKWISE)
            {
                source_y = source_rows - 1 - x;
                source_x = y;
            }
            else if (rotateCode == ROTATE_180)
            {
                source_y = source_rows - 1 - y;
                source_x = source_cols - 1 - x;
            }
            else
            {
                source_y = x;
                source_x = source_cols - 1 - y;
            }
            std::memcpy(
                dst.pixelPtr(y, x),
                source.pixelPtr(source_y, source_x),
                source.elemSize());
        }
    }
}

inline void repeat(const Mat& src, int ny, int nx, Mat& dst)
{
    if (src.empty() || src.dims != 2 || ny <= 0 || nx <= 0)
    {
        CV_Error(Error::StsBadArg, "repeat expects 2D src and positive factors");
    }
    if (src.size.p[0] > std::numeric_limits<int>::max() / ny ||
        src.size.p[1] > std::numeric_limits<int>::max() / nx)
    {
        CV_Error(Error::StsOutOfRange, "repeat output shape overflow");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    dst.create(
        {source.size.p[0] * ny, source.size.p[1] * nx}, source.type());
    for (int y = 0; y < dst.size.p[0]; ++y)
    {
        for (int x = 0; x < dst.size.p[1]; ++x)
        {
            std::memcpy(
                dst.pixelPtr(y, x),
                source.pixelPtr(
                    y % source.size.p[0], x % source.size.p[1]),
                source.elemSize());
        }
    }
}

inline void hconcat(const Mat* src, size_t nsrc, Mat& dst)
{
    const std::vector<Mat> sources =
        array_detail::snapshot_sources(src, nsrc);
    const Mat& first = sources[0];
    if (first.empty() || first.dims != 2)
    {
        CV_Error(Error::StsBadArg, "hconcat expects non-empty 2D inputs");
    }
    int output_cols = 0;
    for (const Mat& input : sources)
    {
        if (input.empty() || input.dims != 2 ||
            input.type() != first.type() ||
            input.size.p[0] != first.size.p[0] ||
            input.size.p[1] > std::numeric_limits<int>::max() - output_cols)
        {
            CV_Error(Error::StsBadArg, "hconcat input mismatch");
        }
        output_cols += input.size.p[1];
    }
    dst.create({first.size.p[0], output_cols}, first.type());
    for (int y = 0; y < first.size.p[0]; ++y)
    {
        int destination_x = 0;
        for (const Mat& input : sources)
        {
            const size_t row_bytes =
                static_cast<size_t>(input.size.p[1]) * input.elemSize();
            std::memcpy(
                dst.pixelPtr(y, destination_x),
                input.pixelPtr(y, 0),
                row_bytes);
            destination_x += input.size.p[1];
        }
    }
}

inline void hconcat(const Mat& src1, const Mat& src2, Mat& dst)
{
    const Mat sources[] = {src1, src2};
    hconcat(sources, 2, dst);
}

inline void hconcat(const std::vector<Mat>& src, Mat& dst)
{
    hconcat(src.data(), src.size(), dst);
}

inline void vconcat(const Mat* src, size_t nsrc, Mat& dst)
{
    const std::vector<Mat> sources =
        array_detail::snapshot_sources(src, nsrc);
    const Mat& first = sources[0];
    if (first.empty() || first.dims != 2)
    {
        CV_Error(Error::StsBadArg, "vconcat expects non-empty 2D inputs");
    }
    int output_rows = 0;
    for (const Mat& input : sources)
    {
        if (input.empty() || input.dims != 2 ||
            input.type() != first.type() ||
            input.size.p[1] != first.size.p[1] ||
            input.size.p[0] > std::numeric_limits<int>::max() - output_rows)
        {
            CV_Error(Error::StsBadArg, "vconcat input mismatch");
        }
        output_rows += input.size.p[0];
    }
    dst.create({output_rows, first.size.p[1]}, first.type());
    int destination_y = 0;
    const size_t row_bytes =
        static_cast<size_t>(first.size.p[1]) * first.elemSize();
    for (const Mat& input : sources)
    {
        for (int y = 0; y < input.size.p[0]; ++y)
        {
            std::memcpy(
                dst.pixelPtr(destination_y++, 0),
                input.pixelPtr(y, 0),
                row_bytes);
        }
    }
}

inline void vconcat(const Mat& src1, const Mat& src2, Mat& dst)
{
    const Mat sources[] = {src1, src2};
    vconcat(sources, 2, dst);
}

inline void vconcat(const std::vector<Mat>& src, Mat& dst)
{
    vconcat(src.data(), src.size(), dst);
}

inline void broadcast(const Mat& src,
                      const std::vector<int>& shape,
                      Mat& dst)
{
    if (src.empty() || shape.empty() ||
        shape.size() > static_cast<size_t>(MAT_MAX_DIM) ||
        src.dims > static_cast<int>(shape.size()))
    {
        CV_Error(Error::StsBadArg, "broadcast shape is incompatible");
    }
    for (int extent : shape)
    {
        if (extent <= 0)
        {
            CV_Error(Error::StsBadSize, "broadcast shape must be positive");
        }
    }
    const size_t leading = shape.size() - static_cast<size_t>(src.dims);
    for (int dim = 0; dim < src.dims; ++dim)
    {
        const int source_extent = src.size.p[dim];
        const int destination_extent =
            shape[leading + static_cast<size_t>(dim)];
        if (source_extent != 1 && source_extent != destination_extent)
        {
            CV_Error(Error::StsUnmatchedSizes, "broadcast extent mismatch");
        }
    }

    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    dst.create(shape, source.type());
    for (size_t i = 0; i < dst.total(); ++i)
    {
        const std::vector<int> destination_coordinates =
            array_detail::coordinates_from_linear(dst, i);
        std::vector<int> source_coordinates(
            static_cast<size_t>(source.dims), 0);
        for (int dim = 0; dim < source.dims; ++dim)
        {
            source_coordinates[static_cast<size_t>(dim)] =
                source.size.p[dim] == 1
                    ? 0
                    : destination_coordinates[
                          leading + static_cast<size_t>(dim)];
        }
        std::memcpy(
            array_detail::pixel_at(dst, i),
            source.data +
                array_detail::pixel_offset(source, source_coordinates),
            source.elemSize());
    }
}

inline void broadcast(const Mat& src, const Mat& shape, Mat& dst)
{
    if (shape.empty() || shape.type() != CV_32SC1 ||
        !shape.isContinuous())
    {
        CV_Error(
            Error::StsBadArg,
            "broadcast target shape must be a continuous CV_32SC1 Mat");
    }
    const int* values = reinterpret_cast<const int*>(shape.data);
    broadcast(
        src,
        std::vector<int>(values, values + shape.total()),
        dst);
}

inline void swap(Mat& a, Mat& b)
{
    if (&a == &b)
    {
        return;
    }
    Mat temporary = a;
    a = b;
    b = temporary;
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_ARRAY_IMPL_HPP
