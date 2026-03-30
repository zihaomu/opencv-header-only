#include "display_cocoa.h"

#import <AppKit/AppKit.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace cvh {
namespace {

template <class Fn>
int run_on_main_int(Fn&& fn)
{
    if ([NSThread isMainThread])
    {
        return fn();
    }

    __block int result = -1;
    dispatch_sync(dispatch_get_main_queue(), ^{
      result = fn();
    });
    return result;
}

template <class Fn>
bool run_on_main_bool(Fn&& fn)
{
    if ([NSThread isMainThread])
    {
        return fn();
    }

    __block bool result = false;
    dispatch_sync(dispatch_get_main_queue(), ^{
      result = fn();
    });
    return result;
}

struct WindowState
{
    NSWindow* window = nil;
    NSImageView* image_view = nil;
    int width = 0;
    int height = 0;
};

class cocoa_context
{
public:
    bool supported()
    {
        return run_on_main_bool([this]() { return ensure_app_ready(); });
    }

    int show_bgr(const char* winname, const unsigned char* bgrdata, int width, int height)
    {
        if (!winname || !bgrdata || width <= 0 || height <= 0)
        {
            return -1;
        }

        std::vector<unsigned char> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
        for (int y = 0; y < height; ++y)
        {
            const unsigned char* src_row = bgrdata + static_cast<size_t>(y) * static_cast<size_t>(width) * 3;
            unsigned char* dst_row = rgba.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4;
            for (int x = 0; x < width; ++x)
            {
                const unsigned char* src = src_row + static_cast<size_t>(x) * 3;
                unsigned char* dst = dst_row + static_cast<size_t>(x) * 4;
                dst[0] = src[2];
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = 255;
            }
        }

        return show_rgba(winname, rgba, width, height);
    }

    int show_gray(const char* winname, const unsigned char* graydata, int width, int height)
    {
        if (!winname || !graydata || width <= 0 || height <= 0)
        {
            return -1;
        }

        std::vector<unsigned char> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
        for (int y = 0; y < height; ++y)
        {
            const unsigned char* src_row = graydata + static_cast<size_t>(y) * static_cast<size_t>(width);
            unsigned char* dst_row = rgba.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 4;
            for (int x = 0; x < width; ++x)
            {
                const unsigned char v = src_row[x];
                unsigned char* dst = dst_row + static_cast<size_t>(x) * 4;
                dst[0] = v;
                dst[1] = v;
                dst[2] = v;
                dst[3] = 255;
            }
        }

        return show_rgba(winname, rgba, width, height);
    }

    int wait_key(int delay)
    {
        return run_on_main_int([this, delay]() {
            if (!ensure_app_ready() || windows_.empty())
            {
                if (delay > 0)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
                }
                return -1;
            }

            if (delay == 0)
            {
                for (;;)
                {
                    const int key = pump_events_once([NSDate distantFuture]);
                    if (key >= 0)
                    {
                        return key;
                    }
                }
            }

            const auto timeout = std::max(0, delay);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
            while (std::chrono::steady_clock::now() < deadline)
            {
                const auto now = std::chrono::steady_clock::now();
                const auto remain_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
                const int step_ms = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(remain_ms, 10)));
                NSDate* until = [NSDate dateWithTimeIntervalSinceNow:static_cast<NSTimeInterval>(step_ms) / 1000.0];

                const int key = pump_events_once(until);
                if (key >= 0)
                {
                    return key;
                }
            }
            return -1;
        });
    }

private:
    bool ensure_app_ready()
    {
        if (app_ready_)
        {
            return true;
        }

        [NSApplication sharedApplication];
        if (NSApp == nil)
        {
            return false;
        }

        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];
        [NSApp activateIgnoringOtherApps:YES];
        app_ready_ = true;
        return true;
    }

    WindowState* get_or_create_window(const char* winname, int width, int height)
    {
        auto it = windows_.find(winname);
        if (it == windows_.end())
        {
            NSRect rect = NSMakeRect(0, 0, width, height);
            const NSUInteger style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                     NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;

            NSWindow* window = [[NSWindow alloc] initWithContentRect:rect
                                                           styleMask:style
                                                             backing:NSBackingStoreBuffered
                                                               defer:NO];
            if (window == nil)
            {
                return nullptr;
            }

            [window setReleasedWhenClosed:NO];
            [window setTitle:[NSString stringWithUTF8String:winname]];
            [window center];

            NSImageView* image_view = [[NSImageView alloc] initWithFrame:rect];
            if (image_view == nil)
            {
                [window release];
                return nullptr;
            }
            [image_view setImageScaling:NSImageScaleNone];
            [image_view setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
            [window setContentView:image_view];

            WindowState ws;
            ws.window = window;
            ws.image_view = image_view;
            ws.width = width;
            ws.height = height;

            it = windows_.emplace(winname, ws).first;

            [window makeKeyAndOrderFront:nil];
            [NSApp activateIgnoringOtherApps:YES];
            [image_view release];
        }
        else if (it->second.width != width || it->second.height != height)
        {
            it->second.width = width;
            it->second.height = height;
            [it->second.window setContentSize:NSMakeSize(width, height)];
            [it->second.image_view setFrame:NSMakeRect(0, 0, width, height)];
        }

        if (![it->second.window isVisible])
        {
            [it->second.window makeKeyAndOrderFront:nil];
            [NSApp activateIgnoringOtherApps:YES];
        }

        return &it->second;
    }

    int update_window_image(WindowState& ws, const unsigned char* rgba, int width, int height)
    {
        NSBitmapImageRep* rep = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:nil
                          pixelsWide:width
                          pixelsHigh:height
                       bitsPerSample:8
                     samplesPerPixel:4
                            hasAlpha:YES
                            isPlanar:NO
                      colorSpaceName:NSDeviceRGBColorSpace
                         bitmapFormat:NSBitmapFormatAlphaNonpremultiplied
                          bytesPerRow:width * 4
                         bitsPerPixel:32];
        if (rep == nil)
        {
            return -1;
        }

        unsigned char* dst = [rep bitmapData];
        if (!dst)
        {
            [rep release];
            return -1;
        }
        std::memcpy(dst, rgba, static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

        NSImage* image = [[NSImage alloc] initWithSize:NSMakeSize(width, height)];
        if (image == nil)
        {
            [rep release];
            return -1;
        }

        [image addRepresentation:rep];
        [ws.image_view setImage:image];
        [ws.window displayIfNeeded];
        [NSApp updateWindows];

        [image release];
        [rep release];
        return 0;
    }

    int show_rgba(const char* winname, const std::vector<unsigned char>& rgba, int width, int height)
    {
        return run_on_main_int([this, winname, width, height, &rgba]() {
            if (!ensure_app_ready())
            {
                return -1;
            }

            WindowState* ws = get_or_create_window(winname, width, height);
            if (!ws)
            {
                return -1;
            }

            return update_window_image(*ws, rgba.data(), width, height);
        });
    }

    int pump_events_once(NSDate* until)
    {
        NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                            untilDate:until
                                               inMode:NSDefaultRunLoopMode
                                              dequeue:YES];
        if (event == nil)
        {
            [NSApp updateWindows];
            return -1;
        }

        int key = -1;
        if ([event type] == NSEventTypeKeyDown)
        {
            NSString* chars = [event charactersIgnoringModifiers];
            if (chars != nil && [chars length] > 0)
            {
                key = static_cast<int>([chars characterAtIndex:0] & 0xFF);
            }
            else
            {
                key = static_cast<int>([event keyCode]);
            }
        }

        [NSApp sendEvent:event];
        [NSApp updateWindows];
        return key;
    }

private:
    bool app_ready_ = false;
    std::map<std::string, WindowState> windows_;
};

cocoa_context& global_cocoa_context()
{
    static cocoa_context ctx;
    return ctx;
}

}  // namespace

bool display_cocoa::supported()
{
    return global_cocoa_context().supported();
}

int display_cocoa::show_bgr(const char* winname, const unsigned char* bgrdata, int width, int height)
{
    return global_cocoa_context().show_bgr(winname, bgrdata, width, height);
}

int display_cocoa::show_gray(const char* winname, const unsigned char* graydata, int width, int height)
{
    return global_cocoa_context().show_gray(winname, graydata, width, height);
}

int display_cocoa::wait_key(int delay)
{
    return global_cocoa_context().wait_key(delay);
}

}  // namespace cvh
