#include "cvh.h"
#include "gtest/gtest.h"

#include <vector>

using namespace cvh;

namespace
{

std::vector<float> interleaved_to_chw(const Mat& src)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const MatShape shape = src.shape();
    const int h = shape[0];
    const int w = shape[1];
    const int c = src.channels();

    std::vector<float> chw(static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c), 0.0f);
    for (int ch = 0; ch < c; ++ch)
    {
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const size_t chw_index =
                        (static_cast<size_t>(ch) * static_cast<size_t>(h) + static_cast<size_t>(y)) *
                        static_cast<size_t>(w) + static_cast<size_t>(x);
                chw[chw_index] = src.at<float>(y, x, ch);
            }
        }
    }
    return chw;
}

Mat chw_to_interleaved(const std::vector<float>& chw, int h, int w, int c)
{
    CV_Assert(static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c) == chw.size());
    Mat dst({h, w}, CV_MAKETYPE(CV_32F, c));
    for (int ch = 0; ch < c; ++ch)
    {
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                const size_t chw_index =
                        (static_cast<size_t>(ch) * static_cast<size_t>(h) + static_cast<size_t>(y)) *
                        static_cast<size_t>(w) + static_cast<size_t>(x);
                dst.at<float>(y, x, ch) = chw[chw_index];
            }
        }
    }
    return dst;
}

} // namespace

TEST(MatLayoutSemantics_TEST, interleaved_address_formula_matches_pixelptr_and_at)
{
    Mat m({2, 3}, CV_8UC3);
    uchar* raw = m.ptr();
    for (int i = 0; i < 18; ++i)
    {
        raw[i] = static_cast<uchar>(i);
    }

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            const uchar* pix = m.pixelPtr(y, x);
            const size_t pixel_offset = static_cast<size_t>(y) * m.step(0) + static_cast<size_t>(x) * m.elemSize();
            EXPECT_EQ(pix, raw + pixel_offset);

            for (int ch = 0; ch < 3; ++ch)
            {
                const size_t offset = pixel_offset + static_cast<size_t>(ch) * m.elemSize1();
                EXPECT_EQ(m.at<uchar>(y, x, ch), raw[offset]);
            }
        }
    }

    EXPECT_EQ(static_cast<int>(m.at<uchar>(1, 2, 0)), 15);
    EXPECT_EQ(static_cast<int>(m.at<uchar>(1, 2, 1)), 16);
    EXPECT_EQ(static_cast<int>(m.at<uchar>(1, 2, 2)), 17);

    EXPECT_THROW((void)m.pixelPtr(2, 0), Exception);
    EXPECT_THROW((void)m.at<uchar>(0, 0, 3), Exception);
    EXPECT_THROW((void)m.at<float>(0, 0, 0), Exception);
}

TEST(MatLayoutSemantics_TEST, roi_uses_parent_stride_and_preserves_address_formula)
{
    Mat src({4, 5}, CV_8UC3);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src.at<uchar>(y, x, ch) = static_cast<uchar>(y * 40 + x * 7 + ch);
            }
        }
    }

    Mat roi = src(Range(1, 3), Range(2, 5));
    ASSERT_EQ(roi.shape(), (MatShape{2, 3}));
    ASSERT_FALSE(roi.isContinuous());
    EXPECT_EQ(roi.step(0), src.step(0));
    EXPECT_EQ(roi.pixelPtr(0, 0), src.pixelPtr(1, 2));

    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_EQ(roi.at<uchar>(y, x, ch), src.at<uchar>(y + 1, x + 2, ch));

                const size_t off =
                        static_cast<size_t>(y) * roi.step(0) +
                        static_cast<size_t>(x) * roi.elemSize() +
                        static_cast<size_t>(ch) * roi.elemSize1();
                EXPECT_EQ(roi.data[off], roi.at<uchar>(y, x, ch));
            }
        }
    }
}

TEST(MatLayoutSemantics_TEST, chw_reorder_roundtrip_matches_interleaved_for_contiguous_and_roi)
{
    Mat src({2, 3}, CV_32FC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src.at<float>(y, x, ch) = static_cast<float>(100 * ch + 10 * y + x) + 0.25f;
            }
        }
    }

    const std::vector<float> chw = interleaved_to_chw(src);
    ASSERT_EQ(chw.size(), static_cast<size_t>(18));
    EXPECT_FLOAT_EQ(chw[(0 * 2 + 1) * 3 + 2], src.at<float>(1, 2, 0));
    EXPECT_FLOAT_EQ(chw[(1 * 2 + 0) * 3 + 1], src.at<float>(0, 1, 1));
    EXPECT_FLOAT_EQ(chw[(2 * 2 + 1) * 3 + 0], src.at<float>(1, 0, 2));

    const Mat restored = chw_to_interleaved(chw, 2, 3, 3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_FLOAT_EQ(restored.at<float>(y, x, ch), src.at<float>(y, x, ch));
            }
        }
    }

    const Mat roi = src.colRange(1, 3);
    ASSERT_FALSE(roi.isContinuous());
    const std::vector<float> chw_roi = interleaved_to_chw(roi);
    const Mat roi_restored = chw_to_interleaved(chw_roi, 2, 2, 3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_FLOAT_EQ(roi_restored.at<float>(y, x, ch), roi.at<float>(y, x, ch));
            }
        }
    }
}
