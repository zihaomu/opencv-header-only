#include "cvh.h"

float cvh_core_header_odr_peer()
{
    cvh::Mat a({1, 1}, CV_32F);
    cvh::Mat b({1, 1}, CV_32F);
    a = 8.0f;
    b = 2.0f;

    cvh::Mat result;
    cvh::divide(a, b, result);
    return reinterpret_cast<const float*>(result.data)[0];
}
