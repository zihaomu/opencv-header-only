#include "cvh.h"

#include <exception>
#include <iostream>
#include <string>

namespace
{

cvh::Mat make_demo_image()
{
    const int rows = 360;
    const int cols = 640;
    cvh::Mat image({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int band = (x / 80) % 8;
            const uchar r = static_cast<uchar>((x * 255) / (cols - 1));
            const uchar g = static_cast<uchar>((y * 255) / (rows - 1));
            const uchar b = static_cast<uchar>((band % 2) ? 220 : 40);

            image.at<uchar>(y, x, 0) = b;
            image.at<uchar>(y, x, 1) = g;
            image.at<uchar>(y, x, 2) = r;
        }
    }

    return image;
}

}  // namespace

int main(int argc, char** argv)
{
    cvh::Mat image;
    std::string source = "generated";
    int wait_ms = 0;

    if (argc > 1)
    {
        source = argv[1];
        image = cvh::imread(source, cvh::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "[cvh_example_highgui_imshow] imread failed: " << source
                      << ", fallback to generated image\n";
            source = "generated";
            image = make_demo_image();
        }
    }
    else
    {
        image = make_demo_image();
    }

    if (argc > 2)
    {
        try
        {
            wait_ms = std::stoi(argv[2]);
        }
        catch (const std::exception&)
        {
            std::cerr << "[cvh_example_highgui_imshow] invalid wait milliseconds: " << argv[2] << '\n';
            return 2;
        }
    }

    std::cout << "[cvh_example_highgui_imshow] showing " << source << ", size="
              << image.size[1] << "x" << image.size[0] << '\n';
    std::cout << "[cvh_example_highgui_imshow] waitKey(" << wait_ms << ")\n";

    cvh::imshow("cvh::imshow demo", image);
    const int key = cvh::waitKey(wait_ms);
    std::cout << "[cvh_example_highgui_imshow] key=" << key << '\n';
    return 0;
}
