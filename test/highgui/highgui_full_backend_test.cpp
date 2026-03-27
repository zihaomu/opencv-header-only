#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <string>

using namespace cvh;

namespace
{

struct ScopedBackendGuard
{
    ScopedBackendGuard()
        : old_imshow(detail::imshow_dispatch()), old_waitkey(detail::waitkey_dispatch())
    {
    }

    ~ScopedBackendGuard()
    {
        detail::register_imshow_backend(old_imshow);
        detail::register_waitkey_backend(old_waitkey);
    }

    detail::ImshowFn old_imshow;
    detail::WaitKeyFn old_waitkey;
};

struct ScopedEnvVar
{
    explicit ScopedEnvVar(const char* key, const char* value) : key_(key), had_old_(false)
    {
        const char* old = std::getenv(key_);
        if (old)
        {
            had_old_ = true;
            old_value_ = old;
        }
        set(value);
    }

    ~ScopedEnvVar()
    {
        if (had_old_)
        {
            set(old_value_.c_str());
        }
        else
        {
            unset();
        }
    }

    void set(const char* value)
    {
#if defined(_WIN32)
        _putenv_s(key_, value ? value : "");
#else
        if (value)
        {
            setenv(key_, value, 1);
        }
        else
        {
            unsetenv(key_);
        }
#endif
    }

    void unset()
    {
#if defined(_WIN32)
        _putenv_s(key_, "");
#else
        unsetenv(key_);
#endif
    }

    const char* key_;
    bool had_old_;
    std::string old_value_;
};

}  // namespace

TEST(HighguiFull_TEST, register_full_backends_enables_dispatch)
{
    const ScopedBackendGuard guard;

    detail::register_imshow_backend(&detail::imshow_fallback);
    detail::register_waitkey_backend(&detail::waitkey_fallback);
    EXPECT_FALSE(detail::is_imshow_backend_registered());
    EXPECT_FALSE(detail::is_waitkey_backend_registered());

    register_highgui_backends();
    EXPECT_TRUE(detail::is_imshow_backend_registered());
    EXPECT_TRUE(detail::is_waitkey_backend_registered());
}

#if defined(__linux__) && !defined(__ANDROID__)
TEST(HighguiFull_TEST, framebuffer_force_fail_open_throws_runtime_hint)
{
    const ScopedBackendGuard guard;
    const ScopedEnvVar mode("CVH_FB_TEST_MODE", "force_fail_open");

    register_highgui_backends();

    Mat img({8, 8}, CV_8UC3);
    img = 0;

    try
    {
        imshow("fb_force_fail", img);
        FAIL() << "imshow should throw when framebuffer open is forced to fail";
    }
    catch (const Exception& e)
    {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("backend is unavailable"), std::string::npos);
        EXPECT_NE(msg.find("imwrite"), std::string::npos);
    }
}

TEST(HighguiFull_TEST, framebuffer_mock_success_path_returns_without_throw)
{
    const ScopedBackendGuard guard;
    const ScopedEnvVar mode("CVH_FB_TEST_MODE", "mock_success");
    const ScopedEnvVar w("CVH_FB_TEST_WIDTH", "64");
    const ScopedEnvVar h("CVH_FB_TEST_HEIGHT", "48");

    register_highgui_backends();

    Mat img({16, 24}, CV_8UC3);
    img = 128;

    EXPECT_NO_THROW(imshow("fb_mock_success", img));
}
#else
TEST(HighguiFull_TEST, framebuffer_branch_tests_skipped_on_non_linux)
{
    GTEST_SKIP() << "Linux framebuffer branch is only available on Linux non-Android targets.";
}
#endif
