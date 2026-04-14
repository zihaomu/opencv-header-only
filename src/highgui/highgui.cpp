#include "cvh/highgui/highgui.hpp"

#include "cvh/imgproc/imgproc.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__linux__) && !defined(__ANDROID__)
#include "display_fb.h"
#if defined(CVH_HAVE_X11)
#include "display_x11.h"
#endif
#endif

#if defined(_WIN32)
#include "display_win32.h"
#endif

#if defined(__APPLE__) && defined(CVH_HAVE_COCOA)
#include "display_cocoa.h"
#endif

namespace cvh {
namespace detail {

inline Mat bgra_to_bgr(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && src.dims == 2 && src.channels() == 4);
    Mat out({src.size[0], src.size[1]}, CV_8UC3);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = out.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const uchar* srow = src.data + static_cast<size_t>(y) * src_step;
        uchar* drow = out.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            const uchar* in = srow + static_cast<size_t>(x) * 4;
            uchar* out_px = drow + static_cast<size_t>(x) * 3;
            out_px[0] = in[0];
            out_px[1] = in[1];
            out_px[2] = in[2];
        }
    }
    return out;
}

inline Mat gray_to_bgr(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U && src.dims == 2 && src.channels() == 1);
    Mat out;
    cvtColor(src, out, COLOR_GRAY2BGR);
    return out;
}

inline Mat ensure_supported_u8_image(const Mat& input)
{
    CV_Assert(!input.empty() && "imshow: source image can not be empty");
    CV_Assert(input.depth() == CV_8U && "imshow: v1 supports CV_8U only");
    CV_Assert(input.dims == 2 && "imshow: only 2D Mat is supported");

    if (input.channels() == 1 || input.channels() == 3)
    {
        return input;
    }
    if (input.channels() == 4)
    {
        return bgra_to_bgr(input);
    }

    CV_Error_(Error::StsBadArg, ("imshow: unsupported channel count=%d", input.channels()));
    return Mat();
}

inline void throw_imshow_backend_unavailable()
{
#if defined(__linux__) && !defined(__ANDROID__)
    CV_Error(Error::StsError,
             "imshow backend is unavailable at runtime. Linux requires X11 DISPLAY or framebuffer /dev/fb0 access. "
             "Use imwrite(\"out.png\", mat) as temporary replacement.");
#elif defined(__APPLE__)
    CV_Error(Error::StsError,
             "imshow backend is unavailable at runtime. macOS requires a GUI session with WindowServer access. "
             "Use imwrite(\"out.png\", mat) as temporary replacement.");
#else
    CV_Error(Error::StsError,
             "imshow backend is unavailable at runtime. Use imwrite(\"out.png\", mat) as temporary replacement.");
#endif
}

inline bool fb_test_mode_active()
{
    const char* mode = std::getenv("CVH_FB_TEST_MODE");
    return mode && mode[0] != '\0';
}

inline void blit_center(const Mat& src, Mat& dst)
{
    CV_Assert(src.depth() == dst.depth());
    CV_Assert(src.channels() == dst.channels());
    CV_Assert(src.size[0] <= dst.size[0] && src.size[1] <= dst.size[1]);

    const int channels = src.channels();
    const int off_y = (dst.size[0] - src.size[0]) / 2;
    const int off_x = (dst.size[1] - src.size[1]) / 2;
    const size_t copy_bytes = static_cast<size_t>(src.size[1]) * channels;

    for (int y = 0; y < src.size[0]; ++y)
    {
        const uchar* srow = src.data + static_cast<size_t>(y) * src.step(0);
        uchar* drow = dst.data + static_cast<size_t>(y + off_y) * dst.step(0) + static_cast<size_t>(off_x) * channels;
        std::memcpy(drow, srow, copy_bytes);
    }
}

inline Mat letterbox_to_size(const Mat& src, int dst_w, int dst_h)
{
    CV_Assert(dst_w > 0 && dst_h > 0);

    if (src.size[1] == dst_w && src.size[0] == dst_h)
    {
        return src;
    }

    Mat canvas({dst_h, dst_w}, src.type());
    canvas = 0;

    int fit_w = dst_w;
    int fit_h = dst_h;
    if (src.size[1] * dst_h > dst_w * src.size[0])
    {
        fit_h = std::max(1, (dst_w * src.size[0]) / src.size[1]);
    }
    else
    {
        fit_w = std::max(1, (dst_h * src.size[1]) / src.size[0]);
    }

    Mat resized;
    resize(src, resized, Size(fit_w, fit_h), 0.0, 0.0, INTER_LINEAR);
    blit_center(resized, canvas);
    return canvas;
}

#if defined(_WIN32)
#pragma pack(push, 1)
struct BmpFileHeader
{
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BmpInfoHeader
{
    uint32_t biSize;
    int32_t biWidth;
    int32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t biXPelsPerMeter;
    int32_t biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)

inline bool encode_bmp24(const Mat& input, std::vector<uchar>& out)
{
    if (input.empty() || input.depth() != CV_8U || input.dims != 2)
    {
        return false;
    }

    Mat img = ensure_supported_u8_image(input);
    if (img.channels() == 1)
    {
        img = gray_to_bgr(img);
    }
    if (img.channels() != 3)
    {
        return false;
    }

    const int width = img.size[1];
    const int height = img.size[0];
    if (width <= 0 || height <= 0)
    {
        return false;
    }

    const uint32_t row_stride = static_cast<uint32_t>(((width * 3) + 3) & ~3);
    const uint32_t pixel_bytes = row_stride * static_cast<uint32_t>(height);
    const uint32_t total_bytes = static_cast<uint32_t>(sizeof(BmpFileHeader) + sizeof(BmpInfoHeader)) + pixel_bytes;

    out.assign(total_bytes, 0);

    BmpFileHeader* fh = reinterpret_cast<BmpFileHeader*>(out.data());
    fh->bfType = 0x4D42;  // 'BM'
    fh->bfSize = total_bytes;
    fh->bfReserved1 = 0;
    fh->bfReserved2 = 0;
    fh->bfOffBits = static_cast<uint32_t>(sizeof(BmpFileHeader) + sizeof(BmpInfoHeader));

    BmpInfoHeader* ih = reinterpret_cast<BmpInfoHeader*>(out.data() + sizeof(BmpFileHeader));
    ih->biSize = sizeof(BmpInfoHeader);
    ih->biWidth = width;
    ih->biHeight = height;  // bottom-up
    ih->biPlanes = 1;
    ih->biBitCount = 24;
    ih->biCompression = 0;
    ih->biSizeImage = pixel_bytes;
    ih->biXPelsPerMeter = 0;
    ih->biYPelsPerMeter = 0;
    ih->biClrUsed = 0;
    ih->biClrImportant = 0;

    uchar* pixel_base = out.data() + fh->bfOffBits;
    for (int y = 0; y < height; ++y)
    {
        const int src_y = height - 1 - y;
        const uchar* src_row = img.data + static_cast<size_t>(src_y) * img.step(0);
        uchar* dst_row = pixel_base + static_cast<size_t>(y) * row_stride;
        std::memcpy(dst_row, src_row, static_cast<size_t>(width) * 3);
    }

    return true;
}
#endif

void imshow_backend(const std::string& winname, const Mat& mat)
{
    Mat image = ensure_supported_u8_image(mat);

#if defined(_WIN32)
    std::vector<uchar> bmp;
    if (encode_bmp24(image, bmp))
    {
        BitmapWindow::show(winname.c_str(), bmp.data());
        return;
    }
    (void)winname;
    (void)image;
    throw_imshow_backend_unavailable();
#elif defined(__APPLE__) && defined(CVH_HAVE_COCOA)
    if (image.channels() == 1)
    {
        if (display_cocoa::show_gray(winname.c_str(), image.data, image.size[1], image.size[0]) == 0)
        {
            return;
        }
    }
    else if (image.channels() == 3)
    {
        if (display_cocoa::show_bgr(winname.c_str(), image.data, image.size[1], image.size[0]) == 0)
        {
            return;
        }
    }

    (void)winname;
    (void)image;
    throw_imshow_backend_unavailable();
#elif defined(__linux__) && !defined(__ANDROID__)
    const bool force_fb_path = fb_test_mode_active();
#if defined(CVH_HAVE_X11)
    if (!force_fb_path)
    {
        if (image.channels() == 1)
        {
            if (display_x11::show_gray(winname.c_str(), image.data, image.size[1], image.size[0]) == 0)
            {
                return;
            }
        }
        else if (image.channels() == 3)
        {
            if (display_x11::show_bgr(winname.c_str(), image.data, image.size[1], image.size[0]) == 0)
            {
                return;
            }
        }
    }
#endif

    display_fb dpy;
    if (dpy.open() == 0)
    {
        const int dpy_w = dpy.get_width();
        const int dpy_h = dpy.get_height();

        if (dpy_w > 0 && dpy_h > 0)
        {
            Mat to_show = image;
            if (to_show.channels() == 4)
            {
                to_show = bgra_to_bgr(to_show);
            }
            to_show = letterbox_to_size(to_show, dpy_w, dpy_h);

            if (to_show.channels() == 1)
            {
                (void)dpy.show_gray(to_show.data, to_show.size[1], to_show.size[0]);
                return;
            }
            if (to_show.channels() == 3)
            {
                (void)dpy.show_bgr(to_show.data, to_show.size[1], to_show.size[0]);
                return;
            }
        }
    }
    (void)winname;
    (void)image;
    throw_imshow_backend_unavailable();
#else
    (void)winname;
    (void)image;
    throw_imshow_backend_unavailable();
#endif
}

int waitkey_backend(int delay)
{
#if defined(_WIN32)
    const int wait = delay < 0 ? 0 : delay;
    return BitmapWindow::waitKey(static_cast<UINT>(wait));
#elif defined(__APPLE__) && defined(CVH_HAVE_COCOA)
    return display_cocoa::wait_key(delay);
#elif defined(__linux__) && !defined(__ANDROID__) && defined(CVH_HAVE_X11)
    return display_x11::wait_key(delay);
#else
    if (delay > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    return -1;
#endif
}

}  // namespace detail

void register_highgui_backends()
{
    detail::register_imshow_backend(&detail::imshow_backend);
    detail::register_waitkey_backend(&detail::waitkey_backend);
}

}  // namespace cvh
