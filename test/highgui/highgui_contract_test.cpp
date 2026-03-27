#include "cvh.h"
#include "gtest/gtest.h"

#include <string>

using namespace cvh;

namespace
{

int g_imshow_calls = 0;
int g_waitkey_calls = 0;
int g_last_rows = -1;
int g_last_cols = -1;
std::string g_last_name;

void test_imshow_backend(const std::string& winname, const Mat& mat)
{
    ++g_imshow_calls;
    g_last_name = winname;
    g_last_rows = mat.size[0];
    g_last_cols = mat.size[1];
}

int test_waitkey_backend(int delay)
{
    ++g_waitkey_calls;
    return delay + 7;
}

}  // namespace

TEST(Highgui_TEST, imshow_requires_full_or_imwrite_hint_in_lite)
{
    Mat img({2, 2}, CV_8UC3);
    img = 0;

    try
    {
        imshow("lite_mode_window", img);
        FAIL() << "imshow should throw in CVH_LITE mode";
    }
    catch (const Exception& e)
    {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("CVH_FULL"), std::string::npos);
        EXPECT_NE(msg.find("imwrite"), std::string::npos);
    }
}

TEST(Highgui_TEST, waitkey_requires_full_in_lite)
{
    try
    {
        (void)waitKey(1);
        FAIL() << "waitKey should throw in CVH_LITE mode";
    }
    catch (const Exception& e)
    {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("CVH_FULL"), std::string::npos);
    }
}

TEST(Highgui_TEST, dispatch_registration_overrides_api_calls)
{
    g_imshow_calls = 0;
    g_waitkey_calls = 0;
    g_last_rows = -1;
    g_last_cols = -1;
    g_last_name.clear();

    const detail::ImshowFn old_imshow = detail::imshow_dispatch();
    const detail::WaitKeyFn old_waitkey = detail::waitkey_dispatch();

    detail::register_imshow_backend(&test_imshow_backend);
    detail::register_waitkey_backend(&test_waitkey_backend);

    Mat img({2, 5}, CV_8UC1);
    img = 9;

    imshow("dispatch_case", img);
    const int key = waitKey(33);

    EXPECT_EQ(g_imshow_calls, 1);
    EXPECT_EQ(g_waitkey_calls, 1);
    EXPECT_EQ(g_last_name, "dispatch_case");
    EXPECT_EQ(g_last_rows, 2);
    EXPECT_EQ(g_last_cols, 5);
    EXPECT_EQ(key, 40);

    detail::register_imshow_backend(old_imshow);
    detail::register_waitkey_backend(old_waitkey);
}

TEST(Highgui_TEST, lite_mode_keeps_backend_unregistered_by_default)
{
    EXPECT_FALSE(detail::is_imshow_backend_registered());
    EXPECT_FALSE(detail::is_waitkey_backend_registered());
    detail::ensure_highgui_backends_registered_once();
    EXPECT_FALSE(detail::is_imshow_backend_registered());
    EXPECT_FALSE(detail::is_waitkey_backend_registered());
}
