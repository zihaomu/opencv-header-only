//
// Created by OpenAI Codex on 2026/3/6.
//

#include "cvh.h"
#include "test/utils/mat_load.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace cvh;

namespace {

std::string core_data_path(const std::string& filename)
{
    return std::string(M_ROOT_PATH) + "/test/core/test_data/data/" + filename;
}

void expect_mat_close(const Mat& out, const Mat& ref, float abs_tol, float rel_tol, const std::string& case_name)
{
    ASSERT_EQ(out.shape(), ref.shape()) << "shape mismatch, case=" << case_name;

    const auto total_size = out.total();
    const float* out_data = reinterpret_cast<const float*>(out.data);
    const float* ref_data = reinterpret_cast<const float*>(ref.data);

    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;

    for (size_t i = 0; i < total_size; ++i)
    {
        const double diff = std::abs(static_cast<double>(out_data[i]) - static_cast<double>(ref_data[i]));
        const double denom = std::max(1.0, std::abs(static_cast<double>(ref_data[i])));
        const double rel = diff / denom;

        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        mean_abs += diff;
    }

    mean_abs /= std::max<size_t>(1, total_size);
    const bool pass = (max_abs <= abs_tol) || (max_rel <= rel_tol);

    EXPECT_TRUE(pass)
        << "case=" << case_name
        << ", max_abs=" << max_abs
        << ", max_rel=" << max_rel
        << ", mean_abs=" << mean_abs
        << ", abs_tol=" << abs_tol
        << ", rel_tol=" << rel_tol;
}

enum class BinaryCaseOp
{
    Add,
    Sub,
    Mul,
    Div,
};

struct BinaryCaseConfig
{
    const char* name;
    BinaryCaseOp op;
    float abs_tol;
    float rel_tol;
};

Mat run_binary_case(const BinaryCaseConfig& config, const Mat& a, const Mat& b)
{
    Mat out;

    switch (config.op)
    {
        case BinaryCaseOp::Add:
            add(a, b, out);
            break;
        case BinaryCaseOp::Sub:
            subtract(a, b, out);
            break;
        case BinaryCaseOp::Mul:
            multiply(a, b, out);
            break;
        case BinaryCaseOp::Div:
            divide(a, b, out);
            break;
    }

    return out;
}

}  // namespace

TEST(KernelOp_TEST, binary_generated_cases)
{
    const std::vector<BinaryCaseConfig> cases = {
        {"binary_add_same_shape", BinaryCaseOp::Add, 1e-5f, 1e-5f},
        {"binary_sub_mid_broadcast", BinaryCaseOp::Sub, 1e-5f, 1e-5f},
        {"binary_mul_row_broadcast", BinaryCaseOp::Mul, 1e-5f, 1e-5f},
        {"binary_div_scalar_broadcast", BinaryCaseOp::Div, 1e-5f, 1e-5f},
    };

    for (const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        Mat a = readMatFromNpy(core_data_path(std::string(c.name) + "_a.npy"));
        Mat b = readMatFromNpy(core_data_path(std::string(c.name) + "_b.npy"));
        Mat ref = readMatFromNpy(core_data_path(std::string(c.name) + "_o.npy"));

        Mat out = run_binary_case(c, a, b);
        expect_mat_close(out, ref, c.abs_tol, c.rel_tol, c.name);
    }
}

TEST(KernelOp_TEST, transpose_generated_cases)
{
    {
        const std::string case_name = "transpose_last2_3d";
        Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
        Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
        Mat out = transpose(input);
        expect_mat_close(out, ref, 1e-6f, 1e-6f, case_name);
    }

    {
        const std::string case_name = "transpose_perm_4d";
        Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
        Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
        Mat out = transposeND(input, {0, 2, 1, 3});
        expect_mat_close(out, ref, 1e-6f, 1e-6f, case_name);
    }
}

TEST(KernelOp_TEST, rope_generated_case)
{
    Mat q = readMatFromNpy(core_data_path("rope_small_q_i.npy"));
    Mat k = readMatFromNpy(core_data_path("rope_small_k_i.npy"));
    Mat q_ref = readMatFromNpy(core_data_path("rope_small_q_o.npy"));
    Mat k_ref = readMatFromNpy(core_data_path("rope_small_k_o.npy"));

    rope(q, k, 5);

    expect_mat_close(q, q_ref, 1e-5f, 1e-5f, "rope_small_q");
    expect_mat_close(k, k_ref, 1e-5f, 1e-5f, "rope_small_k");
}

TEST(KernelOp_TEST, softmax_generated_cases)
{
    {
        const std::string case_name = "softmax_lastdim_2d";
        Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
        Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
        Mat out = softmax(input);
        expect_mat_close(out, ref, 1e-5f, 1e-5f, case_name);
    }

    {
        const std::string case_name = "softmax_lastdim_4d";
        Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
        Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
        Mat out = softmax(input);
        expect_mat_close(out, ref, 1e-5f, 1e-5f, case_name);
    }
}

TEST(KernelOp_TEST, softmax_precision_alignment_matches_python_reference)
{
    Mat input = readMatFromNpy(core_data_path("softmax_precision_input.npy"));
    Mat ref_fp32 = readMatFromNpy(core_data_path("softmax_precision_fp32_o.npy"));
    Mat ref_fp16 = readMatFromNpy(core_data_path("softmax_precision_fp16floor_o.npy"));

    Mat out_fp32 = softmax(input);
    expect_mat_close(out_fp32, ref_fp32, 1e-6f, 1e-6f, "softmax_precision_fp32");

    Mat aligned_fp16 = align_precision_sensitive_input(input, RuntimePrecision::FP16);
    Mat out_fp16 = softmax(aligned_fp16);
    expect_mat_close(out_fp16, ref_fp16, 1e-6f, 1e-6f, "softmax_precision_fp16_floor");

    Mat aligned_int8 = align_precision_sensitive_input(input, RuntimePrecision::INT8);
    Mat out_int8 = softmax(aligned_int8);
    expect_mat_close(out_int8, ref_fp16, 1e-6f, 1e-6f, "softmax_precision_int8_floor");
}

TEST(KernelOp_TEST, silu_generated_case)
{
    const std::string case_name = "silu_basic";
    Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
    Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
    Mat out = silu(input);
    expect_mat_close(out, ref, 1e-5f, 1e-5f, case_name);
}

TEST(KernelOp_TEST, rmsnorm_generated_cases)
{
    {
        const std::string case_name = "rmsnorm_basic";
        Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
        Mat weight = readMatFromNpy(core_data_path(case_name + "_w.npy"));
        Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
        Mat out = rmsnorm(input, weight, 1e-6f);
        expect_mat_close(out, ref, 1e-5f, 1e-5f, case_name);
    }

    {
        const std::string case_name = "rmsnorm_rank2";
        Mat input = readMatFromNpy(core_data_path(case_name + "_i.npy"));
        Mat weight = readMatFromNpy(core_data_path(case_name + "_w.npy"));
        Mat ref = readMatFromNpy(core_data_path(case_name + "_o.npy"));
        Mat out = rmsnorm(input, weight, 1e-5f);
        expect_mat_close(out, ref, 1e-5f, 1e-5f, case_name);
    }
}
