#ifndef CVH_IMGCODECS_H
#define CVH_IMGCODECS_H

#include "../core/mat.h"
#include "../detail/stb_bridge.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace cvh {

enum ImreadModes
{
    IMREAD_UNCHANGED = -1,
    IMREAD_GRAYSCALE = 0,
    IMREAD_COLOR = 1,
};

namespace detail {

inline std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline std::string file_extension(const std::string& path)
{
    const size_t pos = path.find_last_of('.');
    if (pos == std::string::npos)
    {
        return std::string();
    }
    return lowercase(path.substr(pos + 1));
}

inline int desired_channels_for_flag(int flags)
{
    if (flags == IMREAD_COLOR)
    {
        return 3;
    }
    if (flags == IMREAD_GRAYSCALE)
    {
        return 1;
    }
    return 0;  // unchanged
}

inline void swap_bgr_rgb_inplace(unsigned char* data, int pixel_count, int channels)
{
    if (channels != 3 && channels != 4)
    {
        return;
    }
    for (int i = 0; i < pixel_count; ++i)
    {
        unsigned char* px = data + static_cast<size_t>(i) * channels;
        std::swap(px[0], px[2]);
    }
}

inline void copy_with_bgr_to_rgb(const uchar* src, uchar* dst, int pixel_count, int channels)
{
    if (channels == 3 || channels == 4)
    {
        for (int i = 0; i < pixel_count; ++i)
        {
            const uchar* in = src + static_cast<size_t>(i) * channels;
            uchar* out = dst + static_cast<size_t>(i) * channels;
            out[0] = in[2];
            out[1] = in[1];
            out[2] = in[0];
            if (channels == 4)
            {
                out[3] = in[3];
            }
        }
        return;
    }
    std::memcpy(dst, src, static_cast<size_t>(pixel_count) * static_cast<size_t>(channels));
}

}  // namespace detail

inline Mat imread(const std::string& filename, int flags = IMREAD_COLOR)
{
    int width = 0;
    int height = 0;
    int channels_in_file = 0;
    const int desired_channels = detail::desired_channels_for_flag(flags);
    unsigned char* decoded = stbi_load(filename.c_str(), &width, &height, &channels_in_file, desired_channels);
    if (!decoded || width <= 0 || height <= 0)
    {
        return Mat();
    }

    const int channels = desired_channels > 0 ? desired_channels : channels_in_file;
    if (channels <= 0 || channels > CV_CN_MAX)
    {
        stbi_image_free(decoded);
        return Mat();
    }

    Mat out(std::vector<int>{height, width}, CV_MAKETYPE(CV_8U, channels));
    const size_t total_bytes =
        static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channels);
    std::memcpy(out.data, decoded, total_bytes);
    stbi_image_free(decoded);

    // OpenCV color convention is BGR/BGRA; stb decodes RGB/RGBA.
    if (flags != IMREAD_GRAYSCALE)
    {
        detail::swap_bgr_rgb_inplace(out.data, width * height, channels);
    }

    return out;
}

inline bool imwrite(const std::string& filename, const Mat& img)
{
    if (img.empty() || img.depth() != CV_8U || img.dims != 2)
    {
        return false;
    }

    Mat continuous;
    if (!img.isContinuous())
    {
        continuous = img.clone();
    }
    const Mat& src = img.isContinuous() ? img : continuous;

    const int height = src.size[0];
    const int width = src.size[1];
    const int channels = src.channels();
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4)
    {
        return false;
    }

    const std::string ext = detail::file_extension(filename);
    const int pixel_count = width * height;

    std::vector<uchar> encoded(static_cast<size_t>(pixel_count) * static_cast<size_t>(channels));
    detail::copy_with_bgr_to_rgb(src.data, encoded.data(), pixel_count, channels);

    if (ext == "png")
    {
        const int stride = width * channels;
        return stbi_write_png(filename.c_str(), width, height, channels, encoded.data(), stride) != 0;
    }

    if (ext == "jpg" || ext == "jpeg")
    {
        if (channels == 4)
        {
            std::vector<uchar> rgb(static_cast<size_t>(pixel_count) * 3);
            for (int i = 0; i < pixel_count; ++i)
            {
                rgb[static_cast<size_t>(i) * 3 + 0] = encoded[static_cast<size_t>(i) * 4 + 0];
                rgb[static_cast<size_t>(i) * 3 + 1] = encoded[static_cast<size_t>(i) * 4 + 1];
                rgb[static_cast<size_t>(i) * 3 + 2] = encoded[static_cast<size_t>(i) * 4 + 2];
            }
            return stbi_write_jpg(filename.c_str(), width, height, 3, rgb.data(), 95) != 0;
        }
        return stbi_write_jpg(filename.c_str(), width, height, channels, encoded.data(), 95) != 0;
    }

    if (ext == "bmp")
    {
        return stbi_write_bmp(filename.c_str(), width, height, channels, encoded.data()) != 0;
    }

    return false;
}

}  // namespace cvh

#endif  // CVH_IMGCODECS_H
