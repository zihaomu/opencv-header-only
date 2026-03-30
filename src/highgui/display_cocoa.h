#ifndef DISPLAY_COCOA_H
#define DISPLAY_COCOA_H

namespace cvh {

class display_cocoa
{
public:
    static bool supported();

    static int show_bgr(const char* winname, const unsigned char* bgrdata, int width, int height);
    static int show_gray(const char* winname, const unsigned char* graydata, int width, int height);

    static int wait_key(int delay);
};

}  // namespace cvh

#endif  // DISPLAY_COCOA_H
