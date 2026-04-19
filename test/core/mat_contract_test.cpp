#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "src/core/kernel/transpose_kernel.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace cvh;

namespace {

void fill_with_byte_pattern(Mat& mat)
{
    const size_t byte_count = mat.total() * static_cast<size_t>(CV_ELEM_SIZE(mat.type()));
    for (size_t i = 0; i < byte_count; ++i)
    {
        mat.data[i] = static_cast<uchar>((i * 131u + 17u) & 0xFFu);
    }
}

void expect_transpose2d_bytes_equal(const Mat& src, const Mat& dst)
{
    ASSERT_EQ(src.dims, 2);
    ASSERT_EQ(dst.dims, 2);
    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.size[0], src.size[1]);
    ASSERT_EQ(dst.size[1], src.size[0]);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t elem_bytes = static_cast<size_t>(CV_ELEM_SIZE(src.type()));
    const size_t src_row_bytes = static_cast<size_t>(cols) * elem_bytes;
    const size_t dst_row_bytes = static_cast<size_t>(rows) * elem_bytes;

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            const size_t src_offset = static_cast<size_t>(row) * src_row_bytes + static_cast<size_t>(col) * elem_bytes;
            const size_t dst_offset = static_cast<size_t>(col) * dst_row_bytes + static_cast<size_t>(row) * elem_bytes;
            for (size_t b = 0; b < elem_bytes; ++b)
            {
                ASSERT_EQ(dst.data[dst_offset + b], src.data[src_offset + b])
                    << "row=" << row << ", col=" << col << ", byte=" << b;
            }
        }
    }
}

void expect_transpose_last2_3d_bytes_equal(const Mat& src, const Mat& dst)
{
    ASSERT_EQ(src.dims, 3);
    ASSERT_EQ(dst.dims, 3);
    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.size[0], src.size[0]);
    ASSERT_EQ(dst.size[1], src.size[2]);
    ASSERT_EQ(dst.size[2], src.size[1]);

    const int batch = src.size[0];
    const int rows = src.size[1];
    const int cols = src.size[2];
    const size_t elem_bytes = static_cast<size_t>(CV_ELEM_SIZE(src.type()));
    const size_t src_plane_bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * elem_bytes;
    const size_t dst_plane_bytes = static_cast<size_t>(cols) * static_cast<size_t>(rows) * elem_bytes;

    for (int bidx = 0; bidx < batch; ++bidx)
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                const size_t src_offset = static_cast<size_t>(bidx) * src_plane_bytes +
                                          (static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)) * elem_bytes;
                const size_t dst_offset = static_cast<size_t>(bidx) * dst_plane_bytes +
                                          (static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)) * elem_bytes;
                for (size_t by = 0; by < elem_bytes; ++by)
                {
                    ASSERT_EQ(dst.data[dst_offset + by], src.data[src_offset + by])
                        << "batch=" << bidx << ", row=" << row << ", col=" << col << ", byte=" << by;
                }
            }
        }
    }
}

class DispatchModeGuard
{
public:
    explicit DispatchModeGuard(cpu::DispatchMode mode)
        : previous_(cpu::dispatch_mode())
    {
        cpu::set_dispatch_mode(mode);
    }

    ~DispatchModeGuard()
    {
        cpu::set_dispatch_mode(previous_);
    }

private:
    cpu::DispatchMode previous_;
};

void fill_with_byte_pattern(std::vector<uchar>& bytes)
{
    for (size_t i = 0; i < bytes.size(); ++i)
    {
        bytes[i] = static_cast<uchar>((i * 131u + 17u) & 0xFFu);
    }
}

void transpose2d_reference_bytes(const uchar* src,
                                 uchar* dst,
                                 int rows,
                                 int cols,
                                 size_t elem_size)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            std::memcpy(dst + (static_cast<size_t>(col) * rows + row) * elem_size,
                        src + (static_cast<size_t>(row) * cols + col) * elem_size,
                        elem_size);
        }
    }
}

void expect_transpose_kernel_bytes_equal(int rows,
                                         int cols,
                                         size_t elem_size,
                                         cpu::DispatchMode mode)
{
    const size_t byte_count = static_cast<size_t>(rows) * static_cast<size_t>(cols) * elem_size;

    std::vector<uchar> src(byte_count);
    std::vector<uchar> dst(byte_count);
    std::vector<uchar> ref(byte_count);
    fill_with_byte_pattern(src);

    transpose2d_reference_bytes(src.data(), ref.data(), rows, cols, elem_size);

    DispatchModeGuard guard(mode);
    cpu::reset_last_dispatch_tag();

    if (mode == cpu::DispatchMode::XSimdOnly)
    {
        try
        {
            cpu::transpose2d_kernel_blocked(src.data(), dst.data(), rows, cols, elem_size, 1);
        }
        catch (const Exception& e)
        {
            ASSERT_EQ(e.code, Error::StsNotImplemented)
                << "unexpected exception code in xsimd-only transpose2d kernel";
            return;
        }
        ASSERT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::XSimd);
    }
    else
    {
        cpu::transpose2d_kernel_blocked(src.data(), dst.data(), rows, cols, elem_size, 1);
        ASSERT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::Unknown);
    }

    ASSERT_EQ(dst.size(), ref.size());
    for (size_t i = 0; i < byte_count; ++i)
    {
        ASSERT_EQ(dst[i], ref[i]) << "byte=" << i;
    }
}

}  // namespace

TEST(MatContract_TEST, clone_is_deep_copy)
{
    Mat src({2, 3}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    for (int i = 0; i < 6; ++i)
    {
        src_data[i] = static_cast<float>(i + 1);
    }

    Mat cloned = src.clone();
    ASSERT_EQ(cloned.shape(), src.shape());
    ASSERT_EQ(cloned.type(), src.type());
    ASSERT_NE(cloned.data, src.data);

    float* cloned_data = reinterpret_cast<float*>(cloned.data);
    cloned_data[0] = -100.0f;
    EXPECT_FLOAT_EQ(src_data[0], 1.0f);
}

TEST(MatContract_TEST, copy_assignment_is_shallow_copy)
{
    Mat src({2, 2}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    src_data[0] = 10;
    src_data[1] = 20;
    src_data[2] = 30;
    src_data[3] = 40;

    Mat alias = src;
    ASSERT_EQ(alias.data, src.data);

    int* alias_data = reinterpret_cast<int*>(alias.data);
    alias_data[1] = -7;
    EXPECT_EQ(src_data[1], -7);
}

TEST(MatContract_TEST, reshape_success_shares_storage_and_preserves_total)
{
    Mat src({2, 3}, CV_32F);
    src = 2.0f;

    Mat reshaped = src.reshape({3, 2});
    ASSERT_EQ(reshaped.total(), src.total());
    ASSERT_EQ(reshaped.data, src.data);

    float* reshaped_data = reinterpret_cast<float*>(reshaped.data);
    reshaped_data[5] = 11.0f;
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(src.data)[5], 11.0f);
}

TEST(MatContract_TEST, reshape_total_mismatch_throws)
{
    Mat src({2, 3}, CV_32F);
    EXPECT_THROW((void)src.reshape({5}), Exception);
}

TEST(MatContract_TEST, convert_to_uint8_preserves_shape_and_saturates)
{
    Mat src({1, 5}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    src_data[0] = -5.1f;
    src_data[1] = 0.4f;
    src_data[2] = 12.6f;
    src_data[3] = 255.0f;
    src_data[4] = 300.0f;

    Mat dst;
    src.convertTo(dst, CV_8U);

    ASSERT_EQ(dst.shape(), src.shape());
    ASSERT_EQ(dst.type(), CV_8U);

    const uchar* out = reinterpret_cast<const uchar*>(dst.data);
    EXPECT_EQ(out[0], static_cast<uchar>(0));
    EXPECT_EQ(out[1], static_cast<uchar>(0));
    EXPECT_EQ(out[2], static_cast<uchar>(13));
    EXPECT_EQ(out[3], static_cast<uchar>(255));
    EXPECT_EQ(out[4], static_cast<uchar>(255));
}

TEST(MatContract_TEST, convert_to_unsupported_type_throws)
{
    Mat src({1, 4}, CV_32F);
    Mat dst;
    EXPECT_THROW(src.convertTo(dst, CV_64F), Exception);
}

TEST(MatContract_TEST, setto_covers_all_elements_for_odd_16bit_shape)
{
    Mat m({3}, CV_16U);
    m.setTo(9.0f);

    const ushort* out = reinterpret_cast<const ushort*>(m.data);
    EXPECT_EQ(out[0], static_cast<ushort>(9));
    EXPECT_EQ(out[1], static_cast<ushort>(9));
    EXPECT_EQ(out[2], static_cast<ushort>(9));
}

TEST(MatContract_TEST, setto_covers_all_elements_for_odd_16s_shape)
{
    Mat m({5}, CV_16S);
    m.setTo(-3.0f);

    const short* out = reinterpret_cast<const short*>(m.data);
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(out[i], static_cast<short>(-3));
    }
}

TEST(MatContract_TEST, setto_covers_all_elements_for_odd_16f_shape)
{
    Mat m({3}, CV_16F);
    m.setTo(1.75f);

    const hfloat* out = reinterpret_cast<const hfloat*>(m.data);
    EXPECT_NEAR(static_cast<float>(out[0]), 1.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(out[1]), 1.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(out[2]), 1.75f, 1e-3f);
}

TEST(MatContract_TEST, empty_mat_copyto_releases_destination)
{
    Mat src;
    Mat dst({2, 2}, CV_32F);
    dst = 3.0f;

    src.copyTo(dst);
    EXPECT_TRUE(dst.empty());
}

TEST(MatContract_TEST, copyto_type_mismatch_throws)
{
    Mat src({2, 2}, CV_32F);
    src = 1.0f;

    Mat dst({2, 2}, CV_32S);
    EXPECT_THROW(src.copyTo(dst), Exception);
}

TEST(MatContract_TEST, external_memory_is_not_owned_by_mat)
{
    auto* raw = static_cast<float*>(std::malloc(4 * sizeof(float)));
    ASSERT_NE(raw, nullptr);
    raw[0] = 1.0f;
    raw[1] = 2.0f;
    raw[2] = 3.0f;
    raw[3] = 4.0f;

    {
        Mat wrapped({2, 2}, CV_32F, raw);
        wrapped.setTo(6.0f);
    }

    raw[0] = 7.0f;
    EXPECT_FLOAT_EQ(raw[0], 7.0f);
    std::free(raw);
}

TEST(MatContract_TEST, unsupported_depth_is_rejected_in_create)
{
    Mat m;
    const int sizes[2] = {2, 2};
    EXPECT_THROW(m.create(2, sizes, CV_64F), Exception);
}

TEST(MatContract_TEST, at_i0_checks_upper_bound)
{
    Mat m({2, 2}, CV_8U);
    m.setTo(1.0f);

    EXPECT_NO_THROW((void)m.at<uchar>(0));
    EXPECT_NO_THROW((void)m.at<uchar>(3));
    EXPECT_THROW((void)m.at<uchar>(4), Exception);

    const Mat cm = m;
    EXPECT_NO_THROW((void)cm.at<uchar>(3));
    EXPECT_THROW((void)cm.at<uchar>(4), Exception);
}

TEST(MatContract_TEST, transpose2d_preserves_interleaved_bytes_for_multi_type_multi_channel)
{
    const int types[] = {
        CV_8UC1,
        CV_8UC3,
        CV_8UC4,
        CV_16SC1,
        CV_16SC3,
        CV_16FC4,
        CV_32SC2,
        CV_32FC3,
        CV_32FC4,
    };

    for (const int type : types)
    {
        Mat src({5, 7}, type);
        fill_with_byte_pattern(src);

        Mat dst = transpose(src);
        expect_transpose2d_bytes_equal(src, dst);
    }
}

TEST(MatContract_TEST, transpose3d_last_two_swap_preserves_interleaved_bytes_for_multichannel_types)
{
    const int types[] = {
        CV_8UC3,
        CV_16SC3,
        CV_32FC3,
        CV_32FC4,
    };

    for (const int type : types)
    {
        Mat src({2, 4, 6}, type);
        fill_with_byte_pattern(src);

        Mat dst = transpose(src);
        expect_transpose_last2_3d_bytes_equal(src, dst);
    }
}

TEST(MatContract_TEST, transpose2d_kernel_blocked_matches_reference_for_elem_sizes_and_shapes_in_auto_mode)
{
    const int shapes[][2] = {
        {11, 29},
        {5, 7},
        {13, 29},
        {64, 65},
    };

    const size_t elem_sizes[] = {1, 2, 4, 8};

    for (const auto& shape : shapes)
    {
        for (const size_t elem_size : elem_sizes)
        {
            SCOPED_TRACE(::testing::Message()
                         << "rows=" << shape[0]
                         << ", cols=" << shape[1]
                         << ", elem_size=" << elem_size);
            expect_transpose_kernel_bytes_equal(shape[0], shape[1], elem_size, cpu::DispatchMode::Auto);
        }
    }
}

TEST(MatContract_TEST, transpose2d_kernel_blocked_xsimd_only_mode_is_correct_or_reports_not_implemented)
{
    const int shapes[][2] = {
        {11, 29},
        {5, 7},
        {13, 29},
        {64, 65},
    };

    const size_t elem_sizes[] = {1, 2, 4, 8};

    for (const auto& shape : shapes)
    {
        for (const size_t elem_size : elem_sizes)
        {
            SCOPED_TRACE(::testing::Message()
                         << "rows=" << shape[0]
                         << ", cols=" << shape[1]
                         << ", elem_size=" << elem_size);
            expect_transpose_kernel_bytes_equal(shape[0], shape[1], elem_size, cpu::DispatchMode::XSimdOnly);
        }
    }
}
