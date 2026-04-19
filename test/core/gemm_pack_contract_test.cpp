#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>

using namespace cvh;

namespace {

void fill_mat_deterministic(Mat& mat, float amp, float bias)
{
    float* data = reinterpret_cast<float*>(mat.data);
    for (size_t i = 0; i < mat.total(); ++i)
    {
        const float fi = static_cast<float>(i);
        data[i] = std::sin(fi * 0.113f) * amp + std::cos(fi * 0.071f) * (amp * 0.7f) + bias;
    }
}

void expect_mat_close(const Mat& out, const Mat& ref, float abs_tol, float rel_tol)
{
    ASSERT_EQ(out.shape(), ref.shape());
    ASSERT_EQ(out.type(), ref.type());

    const float* out_data = reinterpret_cast<const float*>(out.data);
    const float* ref_data = reinterpret_cast<const float*>(ref.data);
    const size_t count = out.total();

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float abs_diff = std::abs(out_data[i] - ref_data[i]);
        const float denom = std::max(1.0f, std::abs(ref_data[i]));
        const float rel_diff = abs_diff / denom;
        max_abs = std::max(max_abs, abs_diff);
        max_rel = std::max(max_rel, rel_diff);
    }

    EXPECT_TRUE(max_abs <= abs_tol || max_rel <= rel_tol)
        << "max_abs=" << max_abs
        << ", max_rel=" << max_rel
        << ", abs_tol=" << abs_tol
        << ", rel_tol=" << rel_tol;
}

}  // namespace

TEST(GemmPackContract_TEST, fp32_packed_matches_reference)
{
    Mat a({2, 17, 31}, CV_32F);
    Mat b({31, 19}, CV_32F);
    fill_mat_deterministic(a, 0.9f, 0.2f);
    fill_mat_deterministic(b, 0.8f, -0.1f);

    const Mat ref = gemm(a, b, false, false);
    const GemmPackedB packed = gemm_pack_b(b);
    const Mat out = gemm(a, packed);

    expect_mat_close(out, ref, 1e-3f, 1e-4f);
}

TEST(GemmPackContract_TEST, fp32_packed_transposed_input_matches_reference)
{
    Mat a({3, 13, 29}, CV_32F);
    Mat b_nt({11, 29}, CV_32F);
    fill_mat_deterministic(a, 0.7f, 0.0f);
    fill_mat_deterministic(b_nt, 0.6f, 0.1f);

    const Mat ref = gemm(a, b_nt, false, true);
    const GemmPackedB packed = gemm_pack_b(b_nt, true);
    const Mat out = gemm(a, packed);

    expect_mat_close(out, ref, 1e-3f, 1e-4f);
}

TEST(GemmPackContract_TEST, fp16_packed_matches_reference)
{
    Mat a({2, 19, 23}, CV_32F);
    Mat b_fp32({23, 15}, CV_32F);
    fill_mat_deterministic(a, 1.0f, -0.3f);
    fill_mat_deterministic(b_fp32, 0.5f, 0.4f);

    Mat b_fp16;
    b_fp32.convertTo(b_fp16, CV_16F);

    const Mat ref = gemm(a, b_fp16, false, false);
    const GemmPackedB packed = gemm_pack_b(b_fp16);
    const Mat out = gemm(a, packed);

    expect_mat_close(out, ref, 3e-2f, 8e-3f);
}

TEST(GemmPackContract_TEST, packed_weights_can_be_reused_across_calls)
{
    Mat b({27, 21}, CV_32F);
    fill_mat_deterministic(b, 0.9f, 0.2f);
    const GemmPackedB packed = gemm_pack_b(b);

    Mat a0({1, 9, 27}, CV_32F);
    Mat a1({2, 7, 27}, CV_32F);
    fill_mat_deterministic(a0, 0.6f, -0.1f);
    fill_mat_deterministic(a1, 1.1f, 0.3f);

    const Mat ref0 = gemm(a0, b, false, false);
    const Mat ref1 = gemm(a1, b, false, false);

    const Mat out0 = gemm(a0, packed);
    const Mat out1 = gemm(a1, packed);

    expect_mat_close(out0, ref0, 1e-3f, 1e-4f);
    expect_mat_close(out1, ref1, 1e-3f, 1e-4f);
}

TEST(GemmPackContract_TEST, unsupported_weight_type_throws)
{
    Mat b({9, 13}, CV_8S);
    EXPECT_THROW((void)gemm_pack_b(b), Exception);
}
