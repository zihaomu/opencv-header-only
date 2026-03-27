#include "cvh.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace cvh;

namespace
{

std::string fixture_path(const std::string& relative)
{
    return std::string(M_ROOT_PATH) + "/test/imgcodecs/data/opencv_extra/" + relative;
}

void require_fixture(const std::string& relative)
{
    ASSERT_TRUE(std::filesystem::exists(fixture_path(relative)))
        << "Missing fixture: " << fixture_path(relative);
}

std::string make_temp_file(const std::string& ext)
{
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::string name = "cvh_imgcodecs_" + std::to_string(now) + "_" +
                             std::to_string(std::rand()) + ext;
    return (std::filesystem::temp_directory_path() / name).string();
}

Mat make_pattern_bgr(int height, int width)
{
    Mat out({height, width}, CV_8UC3);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uchar* px = out.pixelPtr(y, x);
            px[0] = static_cast<uchar>((x * 17 + y * 3) % 256);   // B
            px[1] = static_cast<uchar>((x * 7 + y * 11) % 256);   // G
            px[2] = static_cast<uchar>((x * 13 + y * 19) % 256);  // R
        }
    }
    return out;
}

Mat make_pattern_bgra(int height, int width)
{
    Mat out({height, width}, CV_8UC4);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uchar* px = out.pixelPtr(y, x);
            px[0] = static_cast<uchar>((x * 5 + y * 23) % 256);   // B
            px[1] = static_cast<uchar>((x * 29 + y * 2) % 256);   // G
            px[2] = static_cast<uchar>((x * 31 + y * 17) % 256);  // R
            px[3] = static_cast<uchar>((x * 37 + y * 41) % 256);  // A
        }
    }
    return out;
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    EXPECT_EQ(a.dims, 2);
    EXPECT_EQ(b.dims, 2);
    EXPECT_EQ(a.type(), b.type());
    EXPECT_EQ(a.size[0], b.size[0]);
    EXPECT_EQ(a.size[1], b.size[1]);
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

}  // namespace

TEST(ImgcodecsFormat_TEST, imread_color_supports_png_jpg_bmp_gif_ppm)
{
    const std::vector<std::string> cases = {
        "highgui/readwrite/test_1_c3.png",
        "highgui/readwrite/test_1_c3.jpg",
        "highgui/readwrite/test_rgba_scale.bmp",
        "highgui/gifsuite/g04n3p04.gif",
        "cv/grabcut/image1652.ppm",
    };

    for (const std::string& relative : cases)
    {
        SCOPED_TRACE(relative);
        require_fixture(relative);

        const Mat img = imread(fixture_path(relative), IMREAD_COLOR);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(img.dims, 2);
        EXPECT_EQ(img.depth(), CV_8U);
        EXPECT_EQ(img.channels(), 3);
        EXPECT_GT(img.size[0], 0);
        EXPECT_GT(img.size[1], 0);
    }
}

TEST(ImgcodecsFormat_TEST, imread_grayscale_supports_png_jpg_bmp_gif_ppm)
{
    const std::vector<std::string> cases = {
        "highgui/readwrite/test_1_c3.png",
        "highgui/readwrite/test_1_c3.jpg",
        "highgui/readwrite/test_rgba_scale.bmp",
        "highgui/gifsuite/g04n3p04.gif",
        "cv/grabcut/image1652.ppm",
    };

    for (const std::string& relative : cases)
    {
        SCOPED_TRACE(relative);
        require_fixture(relative);

        const Mat img = imread(fixture_path(relative), IMREAD_GRAYSCALE);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(img.dims, 2);
        EXPECT_EQ(img.depth(), CV_8U);
        EXPECT_EQ(img.channels(), 1);
        EXPECT_GT(img.size[0], 0);
        EXPECT_GT(img.size[1], 0);
    }
}

TEST(ImgcodecsFormat_TEST, imread_unchanged_preserves_expected_channels)
{
    struct Case
    {
        const char* path;
        int expected_channels;
    };

    const std::vector<Case> cases = {
        {"highgui/readwrite/test_1_c3.jpg", 3},
        {"highgui/readwrite/color_palette_alpha.png", 4},
        {"highgui/readwrite/color_palette_no_alpha.png", 3},
        {"highgui/readwrite/test_rgba_scale.bmp", 3},
        {"highgui/gifsuite/g04n3p04.gif", 4},
        {"cv/grabcut/image1652.ppm", 3},
    };

    for (const Case& c : cases)
    {
        SCOPED_TRACE(c.path);
        require_fixture(c.path);

        const Mat img = imread(fixture_path(c.path), IMREAD_UNCHANGED);
        ASSERT_FALSE(img.empty());
        EXPECT_EQ(img.dims, 2);
        EXPECT_EQ(img.depth(), CV_8U);
        EXPECT_EQ(img.channels(), c.expected_channels);
    }
}

TEST(ImgcodecsFormat_TEST, imread_missing_path_returns_empty)
{
    const Mat img = imread(std::string(M_ROOT_PATH) + "/test/imgcodecs/data/opencv_extra/no_such_file.png",
                           IMREAD_COLOR);
    EXPECT_TRUE(img.empty());
}

TEST(ImgcodecsFormat_TEST, imwrite_roundtrip_png_bmp_and_jpg)
{
    const Mat src = make_pattern_bgr(24, 32);
    ASSERT_FALSE(src.empty());

    struct Case
    {
        const char* ext;
        bool expect_lossless;
    };

    const std::vector<Case> cases = {
        {".png", true},
        {".bmp", true},
        {".jpg", false},
    };

    for (const Case& c : cases)
    {
        SCOPED_TRACE(c.ext);
        const std::string filename = make_temp_file(c.ext);
        ASSERT_TRUE(imwrite(filename, src));

        const Mat loaded = imread(filename, IMREAD_COLOR);
        ASSERT_FALSE(loaded.empty());
        EXPECT_EQ(loaded.type(), CV_8UC3);
        EXPECT_EQ(loaded.size[0], src.size[0]);
        EXPECT_EQ(loaded.size[1], src.size[1]);

        const int max_diff = max_abs_diff_u8(src, loaded);
        if (c.expect_lossless)
        {
            EXPECT_EQ(max_diff, 0);
        }
        else
        {
            EXPECT_LE(max_diff, 40);
        }
        std::filesystem::remove(filename);
    }
}

TEST(ImgcodecsFormat_TEST, imwrite_jpg_accepts_bgra_and_drops_alpha)
{
    const Mat src = make_pattern_bgra(16, 20);
    const std::string filename = make_temp_file(".jpg");

    ASSERT_TRUE(imwrite(filename, src));

    const Mat loaded = imread(filename, IMREAD_UNCHANGED);
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.depth(), CV_8U);
    EXPECT_EQ(loaded.channels(), 3);
    EXPECT_EQ(loaded.size[0], src.size[0]);
    EXPECT_EQ(loaded.size[1], src.size[1]);

    std::filesystem::remove(filename);
}

TEST(ImgcodecsFormat_TEST, imwrite_rejects_unsupported_extension_and_invalid_input)
{
    const Mat valid = make_pattern_bgr(8, 8);
    EXPECT_FALSE(imwrite(make_temp_file(".gif"), valid));
    EXPECT_FALSE(imwrite(make_temp_file(".ppm"), valid));
    EXPECT_FALSE(imwrite(make_temp_file(".hdr"), valid));
    EXPECT_FALSE(imwrite(make_temp_file(""), valid));

    const Mat empty;
    EXPECT_FALSE(imwrite(make_temp_file(".png"), empty));

    const Mat fp32({8, 8}, CV_32FC3);
    EXPECT_FALSE(imwrite(make_temp_file(".png"), fp32));

    const Mat rank3({4, 4, 3}, CV_8UC1);
    EXPECT_FALSE(imwrite(make_temp_file(".png"), rank3));

    const Mat ch5({8, 8}, CV_8UC(5));
    EXPECT_FALSE(imwrite(make_temp_file(".png"), ch5));
}

TEST(ImgcodecsFormat_TEST, imread_hdr_is_optional)
{
    const std::string relative = "highgui/readwrite/rle.hdr";
    const std::string path = fixture_path(relative);
    if (!std::filesystem::exists(path))
    {
        GTEST_SKIP() << "HDR fixture is optional: " << path;
    }

    const Mat img = imread(path, IMREAD_UNCHANGED);
    ASSERT_FALSE(img.empty());
    EXPECT_EQ(img.dims, 2);
    EXPECT_EQ(img.depth(), CV_8U);
    EXPECT_EQ(img.channels(), 3);
    EXPECT_GT(img.size[0], 0);
    EXPECT_GT(img.size[1], 0);
}
