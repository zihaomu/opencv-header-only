#include "cvh/cvh.h"

#include <cstdio>
#include <string>

int main()
{
    const std::string input_path = "cvh_lite_pipeline_input.png";
    const std::string output_path = "cvh_lite_pipeline_output.png";
    std::remove(input_path.c_str());
    std::remove(output_path.c_str());

    cvh::Mat src({4, 4}, CV_8UC3);
    for (int y = 0; y < 4; ++y)
    {
        uchar* row = src.data + static_cast<size_t>(y) * src.step(0);
        for (int x = 0; x < 4; ++x)
        {
            uchar* px = row + static_cast<size_t>(x) * 3;
            px[0] = static_cast<uchar>(x * 40);          // B
            px[1] = static_cast<uchar>(y * 70);          // G
            px[2] = static_cast<uchar>((x + y) * 30);    // R
        }
    }

    if (!cvh::imwrite(input_path, src))
    {
        return 1;
    }

    cvh::Mat loaded = cvh::imread(input_path, cvh::IMREAD_COLOR);
    if (loaded.empty() || loaded.channels() != 3 || loaded.size[0] != 4 || loaded.size[1] != 4)
    {
        return 2;
    }

    cvh::Mat resized;
    cvh::resize(loaded, resized, cvh::Size(2, 2), 0.0, 0.0, cvh::INTER_LINEAR);
    if (resized.empty() || resized.size[0] != 2 || resized.size[1] != 2 || resized.channels() != 3)
    {
        return 3;
    }

    cvh::Mat gray;
    cvh::cvtColor(resized, gray, cvh::COLOR_BGR2GRAY);
    if (gray.empty() || gray.channels() != 1 || gray.size[0] != 2 || gray.size[1] != 2)
    {
        return 4;
    }

    cvh::Mat binary;
    cvh::threshold(gray, binary, 80.0, 255.0, cvh::THRESH_BINARY);
    if (binary.empty() || binary.channels() != 1)
    {
        return 5;
    }

    if (!cvh::imwrite(output_path, binary))
    {
        return 6;
    }

    cvh::Mat loaded_out = cvh::imread(output_path, cvh::IMREAD_GRAYSCALE);
    if (loaded_out.empty() || loaded_out.channels() != 1 || loaded_out.size[0] != 2 || loaded_out.size[1] != 2)
    {
        return 7;
    }

    const uchar* p = loaded_out.data;
    const size_t n = loaded_out.total();
    for (size_t i = 0; i < n; ++i)
    {
        if (!(p[i] == 0 || p[i] == 255))
        {
            return 8;
        }
    }

    std::remove(input_path.c_str());
    std::remove(output_path.c_str());
    return 0;
}
