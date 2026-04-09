#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdlib>
#include <vector>

using namespace cvh;

namespace
{

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 255;
    }
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const int diff = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 1e9f;
    }
    CV_Assert(a.depth() == CV_32F);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float diff = std::fabs(pa[i] - pb[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

Mat threshold_reference_u8(const Mat& src, double thresh, double maxval, int type)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(type == THRESH_BINARY || type == THRESH_BINARY_INV);

    Mat out;
    out.create(src.dims, src.size.p, src.type());
    const uchar max_u8 = saturate_cast<uchar>(maxval);

    if (src.isContinuous() && out.isContinuous())
    {
        const size_t scalar_count = src.total() * static_cast<size_t>(src.channels());
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const bool cond = static_cast<double>(src.data[i]) > thresh;
            out.data[i] = (type == THRESH_BINARY) ? (cond ? max_u8 : 0) : (cond ? 0 : max_u8);
        }
        return out;
    }

    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const size_t src_step = src.step(0);
    const size_t dst_step = out.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const uchar* srow = src.data + static_cast<size_t>(y) * src_step;
        uchar* drow = out.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols_scalar; ++x)
        {
            const bool cond = static_cast<double>(srow[x]) > thresh;
            drow[x] = (type == THRESH_BINARY) ? (cond ? max_u8 : 0) : (cond ? 0 : max_u8);
        }
    }
    return out;
}

Mat threshold_reference_f32(const Mat& src, double thresh, double maxval, int type)
{
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(type == THRESH_BINARY ||
              type == THRESH_BINARY_INV ||
              type == THRESH_TRUNC ||
              type == THRESH_TOZERO ||
              type == THRESH_TOZERO_INV);

    Mat out;
    out.create(src.dims, src.size.p, src.type());
    const float thresh_f = static_cast<float>(thresh);
    const float max_f = static_cast<float>(maxval);

    if (src.isContinuous() && out.isContinuous())
    {
        const size_t scalar_count = src.total() * static_cast<size_t>(src.channels());
        const float* s = reinterpret_cast<const float*>(src.data);
        float* d = reinterpret_cast<float*>(out.data);
        for (size_t i = 0; i < scalar_count; ++i)
        {
            const float v = s[i];
            const bool cond = v > thresh_f;
            switch (type)
            {
            case THRESH_BINARY:
                d[i] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                d[i] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                d[i] = cond ? thresh_f : v;
                break;
            case THRESH_TOZERO:
                d[i] = cond ? v : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                d[i] = cond ? 0.0f : v;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("unsupported threshold type=%d", type));
            }
        }
        return out;
    }

    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const size_t src_step = src.step(0);
    const size_t dst_step = out.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const float* srow = reinterpret_cast<const float*>(src.data + static_cast<size_t>(y) * src_step);
        float* drow = reinterpret_cast<float*>(out.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols_scalar; ++x)
        {
            const float v = srow[x];
            const bool cond = v > thresh_f;
            switch (type)
            {
            case THRESH_BINARY:
                drow[x] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                drow[x] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                drow[x] = cond ? thresh_f : v;
                break;
            case THRESH_TOZERO:
                drow[x] = cond ? v : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                drow[x] = cond ? 0.0f : v;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("unsupported threshold type=%d", type));
            }
        }
    }
    return out;
}

}  // namespace

TEST(ImgprocThreshold_TEST, binary_and_binary_inv_match_reference)
{
    Mat src({2, 4}, CV_8UC1);
    const uchar vals[8] = {0, 10, 80, 81, 120, 200, 250, 255};
    for (int i = 0; i < 8; ++i)
    {
        src.data[i] = vals[i];
    }

    Mat out_bin;
    const double ret_bin = threshold(src, out_bin, 80.0, 255.0, THRESH_BINARY);
    EXPECT_DOUBLE_EQ(ret_bin, 80.0);
    EXPECT_EQ(max_abs_diff_u8(out_bin, threshold_reference_u8(src, 80.0, 255.0, THRESH_BINARY)), 0);

    Mat out_inv;
    const double ret_inv = threshold(src, out_inv, 80.0, 255.0, THRESH_BINARY_INV);
    EXPECT_DOUBLE_EQ(ret_inv, 80.0);
    EXPECT_EQ(max_abs_diff_u8(out_inv, threshold_reference_u8(src, 80.0, 255.0, THRESH_BINARY_INV)), 0);
}

TEST(ImgprocThreshold_TEST, supports_multi_channel_u8)
{
    Mat src({2, 2}, CV_8UC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                src.at<uchar>(y, x, c) = static_cast<uchar>(y * 60 + x * 40 + c * 30);
            }
        }
    }

    Mat out;
    threshold(src, out, 90.0, 200.0, THRESH_BINARY);
    ASSERT_EQ(out.type(), src.type());
    EXPECT_EQ(max_abs_diff_u8(out, threshold_reference_u8(src, 90.0, 200.0, THRESH_BINARY)), 0);
}

TEST(ImgprocThreshold_TEST, non_contiguous_roi_path_matches_reference)
{
    Mat parent({6, 7}, CV_8UC1);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x) = static_cast<uchar>((y * 19 + x * 13) % 256);
        }
    }

    Mat roi = parent(Range(1, 5), Range(2, 6));
    ASSERT_FALSE(roi.isContinuous());

    Mat out;
    threshold(roi, out, 100.0, 255.0, THRESH_BINARY_INV);
    EXPECT_EQ(max_abs_diff_u8(out, threshold_reference_u8(roi, 100.0, 255.0, THRESH_BINARY_INV)), 0);
}

TEST(ImgprocThreshold_TEST, maxval_is_saturated_to_u8)
{
    Mat src({1, 4}, CV_8UC1);
    src.at<uchar>(0, 0) = 0;
    src.at<uchar>(0, 1) = 1;
    src.at<uchar>(0, 2) = 2;
    src.at<uchar>(0, 3) = 255;

    Mat out;
    threshold(src, out, 0.0, 300.0, THRESH_BINARY);
    EXPECT_EQ(out.at<uchar>(0, 0), 0);
    EXPECT_EQ(out.at<uchar>(0, 1), 255);
    EXPECT_EQ(out.at<uchar>(0, 2), 255);
    EXPECT_EQ(out.at<uchar>(0, 3), 255);
}

TEST(ImgprocThreshold_TEST, threshold_dryrun_from_upstream_keeps_input_unchanged)
{
    // Ported from OpenCV: modules/imgproc/test/test_thresh.cpp
    // TEST(Imgproc_Threshold, threshold_dryrun)
    Mat input_original({16, 16}, CV_8UC1);
    input_original = 2;
    Mat input = input_original.clone();

    const std::vector<int> thresh_types = {
        THRESH_BINARY,
        THRESH_BINARY_INV,
        THRESH_TRUNC,
        THRESH_TOZERO,
        THRESH_TOZERO_INV,
    };
    const std::vector<int> thresh_flags = {
        0,
        THRESH_OTSU,
        THRESH_TRIANGLE,
    };

    for (int thresh_type : thresh_types)
    {
        for (int thresh_flag : thresh_flags)
        {
            SCOPED_TRACE(thresh_type);
            SCOPED_TRACE(thresh_flag);
            const int composed = thresh_type | thresh_flag | THRESH_DRYRUN;
            const double ret = threshold(input, input, 2.0, 0.0, composed);
            (void)ret;
            EXPECT_EQ(max_abs_diff_u8(input, input_original), 0);
        }
    }
}

TEST(ImgprocThreshold_TEST, throws_on_unsupported_depth_or_type)
{
    Mat out;
    const Mat src_u16({4, 4}, CV_16UC1);
    EXPECT_THROW(threshold(src_u16, out, 80.0, 255.0, THRESH_BINARY), Exception);

    const Mat src_u8({4, 4}, CV_8UC1);
    EXPECT_THROW(threshold(src_u8, out, 80.0, 255.0, 12345), Exception);
    EXPECT_THROW(threshold(src_u8, out, 80.0, 255.0, THRESH_BINARY | THRESH_OTSU | THRESH_TRIANGLE), Exception);
}

TEST(ImgprocThreshold_TEST, supports_cv32f_fixed_threshold_types_for_c1_c3_c4)
{
    for (int cn : {1, 3, 4})
    {
        SCOPED_TRACE(cn);
        Mat src({5, 7}, CV_MAKETYPE(CV_32F, cn));
        for (int y = 0; y < src.size[0]; ++y)
        {
            for (int x = 0; x < src.size[1]; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    const float v = static_cast<float>((y - 2) * 0.9 + (x - 3) * 0.45 + c * 0.33);
                    src.at<float>(y, x, c) = v;
                }
            }
        }

        for (int t : {THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV})
        {
            SCOPED_TRACE(t);
            Mat out;
            const double ret = threshold(src, out, 0.25, 2.75, t);
            EXPECT_DOUBLE_EQ(ret, 0.25);
            ASSERT_EQ(out.type(), src.type());
            ASSERT_EQ(out.size[0], src.size[0]);
            ASSERT_EQ(out.size[1], src.size[1]);

            const Mat ref = threshold_reference_f32(src, 0.25, 2.75, t);
            EXPECT_LE(max_abs_diff_f32(out, ref), 1e-6f);
        }
    }
}

TEST(ImgprocThreshold_TEST, cv32f_non_contiguous_roi_and_dryrun_match_reference)
{
    Mat parent({7, 11}, CV_32FC3);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                parent.at<float>(y, x, c) = static_cast<float>((y * 0.7) - (x * 0.2) + c * 1.25);
            }
        }
    }

    Mat roi = parent(Range(1, 6), Range(2, 10));
    ASSERT_FALSE(roi.isContinuous());

    Mat out;
    const double ret = threshold(roi, out, -0.1, 3.5, THRESH_TOZERO);
    EXPECT_DOUBLE_EQ(ret, -0.1);
    const Mat ref = threshold_reference_f32(roi, -0.1, 3.5, THRESH_TOZERO);
    EXPECT_LE(max_abs_diff_f32(out, ref), 1e-6f);

    Mat dry_input = roi.clone();
    Mat dry_original = dry_input.clone();
    const double dry_ret = threshold(dry_input, dry_input, -0.1, 3.5, THRESH_BINARY | THRESH_DRYRUN);
    EXPECT_DOUBLE_EQ(dry_ret, -0.1);
    EXPECT_LE(max_abs_diff_f32(dry_input, dry_original), 0.0f);
}

TEST(ImgprocThreshold_TEST, cv32f_rejects_otsu_and_triangle_flags)
{
    Mat src({8, 8}, CV_32FC1);
    src = Scalar(0.5, 0.0, 0.0, 0.0);
    Mat out;

    EXPECT_THROW(threshold(src, out, 0.3, 1.0, THRESH_BINARY | THRESH_OTSU), Exception);
    EXPECT_THROW(threshold(src, out, 0.3, 1.0, THRESH_BINARY | THRESH_TRIANGLE), Exception);
}
