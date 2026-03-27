#ifndef CVH_HIGHGUI_HPP
#define CVH_HIGHGUI_HPP

#include "../core/mat.h"

#include <string>

namespace cvh {

#if defined(CVH_FULL)
CV_EXPORTS void register_highgui_backends();
#endif

namespace detail {

using ImshowFn = void (*)(const std::string&, const Mat&);
using WaitKeyFn = int (*)(int);

inline void imshow_fallback(const std::string& winname, const Mat& mat);
inline int waitkey_fallback(int delay);

inline ImshowFn g_imshow_dispatch = &imshow_fallback;
inline WaitKeyFn g_waitkey_dispatch = &waitkey_fallback;

inline void imshow_fallback(const std::string& winname, const Mat& mat)
{
    (void)winname;
    (void)mat;
    CV_Error(Error::StsNotImplemented,
             "imshow requires CVH_FULL backend. Use imwrite(\"out.png\", mat) as temporary replacement.");
}

inline int waitkey_fallback(int delay)
{
    (void)delay;
    CV_Error(Error::StsNotImplemented,
             "waitKey requires CVH_FULL backend. Use your app event loop or avoid waitKey in CVH_LITE.");
    return -1;
}

inline ImshowFn& imshow_dispatch()
{
    return g_imshow_dispatch;
}

inline WaitKeyFn& waitkey_dispatch()
{
    return g_waitkey_dispatch;
}

inline void register_imshow_backend(ImshowFn fn)
{
    if (fn)
    {
        imshow_dispatch() = fn;
    }
}

inline void register_waitkey_backend(WaitKeyFn fn)
{
    if (fn)
    {
        waitkey_dispatch() = fn;
    }
}

inline bool is_imshow_backend_registered()
{
    return imshow_dispatch() != &imshow_fallback;
}

inline bool is_waitkey_backend_registered()
{
    return waitkey_dispatch() != &waitkey_fallback;
}

inline void ensure_highgui_backends_registered_once()
{
#if defined(CVH_FULL)
    static bool initialized = []() {
        cvh::register_highgui_backends();
        return true;
    }();
    (void)initialized;
#endif
}

}  // namespace detail

inline void imshow(const std::string& winname, const Mat& mat)
{
    detail::ensure_highgui_backends_registered_once();
    detail::imshow_dispatch()(winname, mat);
}

inline int waitKey(int delay = 0)
{
    detail::ensure_highgui_backends_registered_once();
    return detail::waitkey_dispatch()(delay);
}

}  // namespace cvh

#endif  // CVH_HIGHGUI_HPP
