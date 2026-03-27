#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace cvh;

namespace
{

std::string fixture_path(const std::string& relative)
{
    return std::string(M_ROOT_PATH) + "/test/imgproc/data/opencv_extra/" + relative;
}

void require_fixture(const std::string& relative)
{
    ASSERT_TRUE(std::filesystem::exists(fixture_path(relative)))
        << "Missing fixture: " << fixture_path(relative);
}

uint32_t adler32(const Mat& src)
{
    static constexpr uint32_t kMod = 65521u;
    uint32_t a = 1u;
    uint32_t b = 0u;
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        a = (a + src.data[i]) % kMod;
        b = (b + a) % kMod;
    }
    return (b << 16) | a;
}

int count_nonzero_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U);
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    int nz = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (src.data[i] != 0)
        {
            ++nz;
        }
    }
    return nz;
}

void assert_binary_u8(const Mat& src)
{
    ASSERT_EQ(src.depth(), CV_8U);
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        ASSERT_TRUE(src.data[i] == 0 || src.data[i] == 255);
    }
}

}  // namespace

TEST(ImgprocPipeline_TEST, fixed_pipeline_matches_snapshot_hashes_from_opencv_extra_images)
{
    struct Case
    {
        const char* relative;
        uint32_t expected_hash;
        int expected_nonzero;
    };

    const std::vector<Case> cases = {
        {"cv/shared/pic1.png", 678252563u, 1879},
        {"cv/imgproc/stuff.jpg", 606059218u, 2757},
        {"cv/grabcut/image1652.ppm", 1741072570u, 1968},
        {"cv/grabcut/mask1652.ppm", 201326593u, 0},
    };

    for (const Case& c : cases)
    {
        SCOPED_TRACE(c.relative);
        require_fixture(c.relative);

        const Mat input = imread(fixture_path(c.relative), IMREAD_COLOR);
        ASSERT_FALSE(input.empty());
        ASSERT_EQ(input.type(), CV_8UC3);

        Mat resized;
        resize(input, resized, Size(64, 48), 0.0, 0.0, INTER_LINEAR);
        ASSERT_EQ(resized.type(), CV_8UC3);

        Mat gray;
        cvtColor(resized, gray, COLOR_BGR2GRAY);
        ASSERT_EQ(gray.type(), CV_8UC1);

        Mat binary;
        const double ret = threshold(gray, binary, 96.0, 255.0, THRESH_BINARY);
        EXPECT_DOUBLE_EQ(ret, 96.0);
        ASSERT_EQ(binary.type(), CV_8UC1);
        ASSERT_EQ(binary.size[0], 48);
        ASSERT_EQ(binary.size[1], 64);
        assert_binary_u8(binary);

        const uint32_t hash = adler32(binary);
        const int nz = count_nonzero_u8(binary);
        EXPECT_EQ(hash, c.expected_hash) << "snapshot hash mismatch, got=" << hash;
        EXPECT_EQ(nz, c.expected_nonzero) << "snapshot nonzero mismatch, got=" << nz;
    }
}
