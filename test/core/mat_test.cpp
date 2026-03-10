//
// Created by mzh on 2024/10/29.
//

#include "cvh.h"
#include "backend/cpu/kernel/gemm_kernel_xsimd.h"
#include "backend/cpu/layer/runtime_weight.h"
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

struct GemmCaseConfig
{
    const char* name;
    bool trans_a;
    bool trans_b;
    float abs_tol;
    float rel_tol;
};

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

}  // namespace

// TODO add test element equal check. compare two mat, or compare mat and scalar.
TEST(Mat_TEST, loadNpy)
{
    // try to get path from system
    std::string path = std::string(M_ROOT_PATH) + "/test/core/test_data/data/random10x12.npy";

    Mat m = readMatFromNpy(path);

    m.print();

    float * data = (float *)m.data;

    float a0 = 0.5488135;
    float a1 = 0.71518934;
    float a2 = 0.56804454;

    M_Assert(data[0] == a0);
    M_Assert(data[1] == a1);
    M_Assert(data[12] == a2);
}

TEST(Mat_TEST, Op_Test)
{
    Mat m3 = Mat({2, 3, 4}, CV_32F);
    m3.setTo(3.0f);

    Mat m2 = Mat({2, 3, 4}, CV_32F);
    m2.setTo(2.0f);

    Mat m4 = Mat({2, 3, 4}, CV_8S);
    m4.setTo(3.0f);

    Mat m5 = Mat({2, 3, 4}, CV_8S);
    m5.setTo(2.0f);

    m3.print();
    m2.print();

    m4.print();
    m5.print();

//    Mat m = m3 - m2;
//    m.print();

    Mat m22 = m4 - m5;
    m22.print();

//    double v = norm(m, m2, NORM_L1);
//    M_Assert(v < 1e-4);
}


TEST(Mat_TEST, Op_Test2)
{
    std::vector<int> test_shape = {1, 4, 5};
    Mat m3 = Mat(test_shape, CV_32F);
    m3.setTo(3.0f);

    Mat m4 = m3 - 1.f;

    Mat m2 = Mat(test_shape, CV_32F);
    m2.setTo(2.0f);

    double v = norm(m4, m2, NORM_L1);
    M_Assert(v < 1e-4);

    Mat m5 = (1 - m3) * 3;
    Mat m6 = Mat(test_shape, CV_32F);
    m6.setTo(-2.0f * 3);

    double v2 = norm(m5, m6, NORM_L1);
    M_Assert(v2 < 1e-4);

}

TEST(Mat_TEST, transposeND)
{
    Mat m = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_0_i.npy");
    Mat m_cheker_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_1_o.npy");
    Mat m_cheker_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_2_o.npy");
    Mat m_cheker_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_3_o.npy");
    Mat m_cheker_4 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_4_o.npy");

    Mat out0 = transposeND(m, {2, 1, 0});
    Mat out1 = transposeND(m, {0, 2, 1});
    Mat out2 = transposeND(m, {2, 0, 1});
    Mat out3 = transposeND(m, {1, 0, 2});

//    std::cout<<"out.print(10) = "<<std::endl;
//    m.print(30);
//    out0.print(30);
//    m_cheker_1.print(30);
//    out1.print(30);
//    m_cheker_2.print(30);
//    out2.print(30);
//    m_cheker_3.print(30);
//    out3.print(30);
//    m_cheker_4.print(30);

    double v0 = norm(out0, m_cheker_1, NORM_L1);
    double v1 = norm(out1, m_cheker_2, NORM_L1);
    double v2 = norm(out2, m_cheker_3, NORM_L1);
    double v3 = norm(out3, m_cheker_4, NORM_L1);

    M_Assert(v0 < 1e-4);
    M_Assert(v1 < 1e-4);
    M_Assert(v2 < 1e-4);
    M_Assert(v3 < 1e-4);
}


TEST(Mat_TEST, mat_mul)
{
    Mat mi_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_0_i.npy");
    Mat mi_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_i.npy");
    Mat mi_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_i.npy");
    Mat mi_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_i.npy");
    Mat mi_4 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_i.npy");
    Mat mo_checker_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_o.npy");
    Mat mo_checker_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_o.npy");
    Mat mo_checker_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_o.npy");
    Mat mo_checker_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_o.npy");


    Mat out0 = gemm(mi_0, mi_1);
    Mat out1 = gemm(mi_0, mi_2, false, true);
    Mat out2 = gemm(mi_0, mi_3, true, false);
    Mat out3 = gemm(mi_0, mi_4, true, true);

    double v0 = norm(out0, mo_checker_0, NORM_L1);
    double v1 = norm(out1, mo_checker_1, NORM_L1);
    double v2 = norm(out2, mo_checker_2, NORM_L1);
    double v3 = norm(out3, mo_checker_3, NORM_L1);

    std::cout<<"norm 0= "<<v0<<std::endl;
    std::cout<<"norm 1= "<<v1<<std::endl;
    std::cout<<"norm 2= "<<v2<<std::endl;
    std::cout<<"norm 3= "<<v3<<std::endl;

    M_Assert(v0 < 1e-3); // m_cheker_1 is empty, TODO, check why this happen.
    M_Assert(v1 < 1e-3);
    M_Assert(v2 < 1e-3);
    M_Assert(v3 < 1e-3);
}

TEST(Mat_TEST, mat_mul_broad_cast)
{
    Mat mi_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_0_i.npy");
    Mat mi_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_i.npy");
    Mat mi_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_i.npy");
    Mat mi_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_i.npy");
    Mat mi_4 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_i.npy");
    Mat mo_checker_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_o.npy");
    Mat mo_checker_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_o.npy");
    Mat mo_checker_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_o.npy");
    Mat mo_checker_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_o.npy");


    Mat out0 = gemm(mi_0, mi_1);
    Mat out1 = gemm(mi_0, mi_2, false, true);
    Mat out2 = gemm(mi_0, mi_3, true, false);
    Mat out3 = gemm(mi_0, mi_4, true, true);

    double v0 = norm(out0, mo_checker_0, NORM_L1);
    double v1 = norm(out1, mo_checker_1, NORM_L1);
    double v2 = norm(out2, mo_checker_2, NORM_L1);
    double v3 = norm(out3, mo_checker_3, NORM_L1);

    std::cout<<"norm 0= "<<v0<<std::endl;
    std::cout<<"norm 1= "<<v1<<std::endl;
    std::cout<<"norm 2= "<<v2<<std::endl;
    std::cout<<"norm 3= "<<v3<<std::endl;

    M_Assert(v0 < 1e-3); // m_cheker_1 is empty, TODO, check why this happen.
    M_Assert(v1 < 1e-3);
    M_Assert(v2 < 1e-3);
    M_Assert(v3 < 1e-3);
}

TEST(Mat_TEST, gemm_generated_cases)
{
    const std::vector<GemmCaseConfig> cases = {
        {"nn_small_odd", false, false, 1e-3f, 1e-4f},
        {"nn_tail_rect", false, false, 1e-3f, 1e-4f},
        {"nn_rank3", false, false, 1e-3f, 1e-4f},
        {"nn_rank4", false, false, 1e-3f, 1e-4f},
        {"nn_broadcast_a", false, false, 1e-3f, 1e-4f},
        {"nn_broadcast_b", false, false, 1e-3f, 1e-4f},
        {"nn_rank_mismatch_a", false, false, 1e-3f, 1e-4f},
        {"nn_rank_mismatch_b", false, false, 1e-3f, 1e-4f},
        {"nt_small_odd", false, true, 1e-3f, 1e-4f},
        {"nt_tail_rect", false, true, 1e-3f, 1e-4f},
        {"nt_rank3", false, true, 1e-3f, 1e-4f},
        {"nt_broadcast_a", false, true, 1e-3f, 1e-4f},
        {"nt_broadcast_b", false, true, 1e-3f, 1e-4f},
        {"nt_rank_mismatch", false, true, 1e-3f, 1e-4f},
        {"tn_basic", true, false, 1e-3f, 1e-4f},
        {"tn_rank3", true, false, 1e-3f, 1e-4f},
        {"tn_broadcast", true, false, 1e-3f, 1e-4f},
        {"tt_basic", true, true, 1e-3f, 1e-4f},
        {"tt_rank3", true, true, 1e-3f, 1e-4f},
        {"tt_broadcast", true, true, 1e-3f, 1e-4f},
        {"nn_large_value", false, false, 5e-2f, 5e-4f},
        {"nt_small_value", false, true, 1e-6f, 5e-4f},
    };

    for (const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        Mat a = readMatFromNpy(core_data_path(std::string("gemm_") + c.name + "_a.npy"));
        Mat b = readMatFromNpy(core_data_path(std::string("gemm_") + c.name + "_b.npy"));
        Mat ref = readMatFromNpy(core_data_path(std::string("gemm_") + c.name + "_o.npy"));

        Mat out = gemm(a, b, c.trans_a, c.trans_b);
        expect_mat_close(out, ref, c.abs_tol, c.rel_tol, c.name);
    }
}

TEST(Mat_TEST, data_convert_fp16_to_fp32)
{
    std::vector<int> test_shape = {1, 4, 5};
    Mat m0 = Mat(test_shape, CV_32F);
    m0.setTo(3.0f);

    Mat m16;
    m0.convertTo(m16, CV_16F);

    Mat m32;
    m16.convertTo(m32, CV_32F);

    double v0 = norm(m32, m0, NORM_L1);
    M_Assert(v0 < 1e-3);
}

TEST(Mat_TEST, gemm_supports_fp16_weight_matrix)
{
    Mat a({1, 3, 8}, CV_32F);
    Mat b({6, 8}, CV_32F);

    float* pa = reinterpret_cast<float*>(a.data);
    float* pb = reinterpret_cast<float*>(b.data);
    for (size_t i = 0; i < a.total(); ++i)
    {
        pa[i] = std::sin(static_cast<float>(i) * 0.19f) * 0.8f + std::cos(static_cast<float>(i) * 0.07f);
    }
    for (size_t i = 0; i < b.total(); ++i)
    {
        pb[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.6f - std::cos(static_cast<float>(i) * 0.05f) * 0.4f;
    }

    Mat b_fp16;
    b.convertTo(b_fp16, CV_16F);

    Mat ref = gemm(a, b, false, true);
    Mat out = gemm(a, b_fp16, false, true);
    expect_mat_close(out, ref, 5e-2f, 1e-2f, "gemm_fp16_weight_matrix");
}

TEST(Mat_TEST, gemm_supports_int8_weight_matrix)
{
    Mat a({1, 4, 8}, CV_32F);
    Mat b({7, 8}, CV_32F);

    float* pa = reinterpret_cast<float*>(a.data);
    float* pb = reinterpret_cast<float*>(b.data);
    for (size_t i = 0; i < a.total(); ++i)
    {
        pa[i] = std::sin(static_cast<float>(i) * 0.17f) * 0.7f + std::cos(static_cast<float>(i) * 0.03f);
    }
    for (size_t i = 0; i < b.total(); ++i)
    {
        pb[i] = std::sin(static_cast<float>(i) * 0.11f) * 0.9f - std::cos(static_cast<float>(i) * 0.09f) * 0.3f;
    }

    Mat b_int8;
    Mat b_scales;
    quantize_int8_per_row(b, b_int8, b_scales);

    Mat ref = gemm(a, b, false, true);
    Mat out = gemm(a, b_int8, b_scales, false, true);
    expect_mat_close(out, ref, 2.5e-1f, 8e-2f, "gemm_int8_weight_matrix");
}

TEST(Mat_TEST, gemm_kernel_nn_nt_match_reference_with_tails)
{
    constexpr int M = 5;
    constexpr int K = 13;
    constexpr int N = 19;

    Mat a({M, K}, CV_32F);
    Mat b_nt({N, K}, CV_32F);
    Mat b_nn({K, N}, CV_32F);

    float* a_data = reinterpret_cast<float*>(a.data);
    float* b_nt_data = reinterpret_cast<float*>(b_nt.data);
    float* b_nn_data = reinterpret_cast<float*>(b_nn.data);

    for (int idx = 0; idx < M * K; ++idx)
    {
        a_data[idx] = std::sin(static_cast<float>(idx) * 0.17f) * 0.7f + std::cos(static_cast<float>(idx) * 0.11f);
    }

    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < K; ++col)
        {
            const float value = std::sin(static_cast<float>(row * K + col) * 0.13f) * 0.6f -
                                std::cos(static_cast<float>(row * K + col) * 0.07f) * 0.4f;
            b_nt_data[row * K + col] = value;
            b_nn_data[col * N + row] = value;
        }
    }

    Mat ref_nt = gemm(a, b_nt, false, true);
    Mat ref_nn = gemm(a, b_nn, false, false);

    Mat out_nt({M, N}, CV_32F);
    Mat out_nn({M, N}, CV_32F);

    cpu::gemm_kernel_xsimd_nt(reinterpret_cast<const float*>(a.data),
                              reinterpret_cast<const float*>(b_nt.data),
                              reinterpret_cast<float*>(out_nt.data),
                              M,
                              N,
                              K);
    cpu::gemm_kernel_xsimd_nn(reinterpret_cast<const float*>(a.data),
                              reinterpret_cast<const float*>(b_nn.data),
                              reinterpret_cast<float*>(out_nn.data),
                              M,
                              N,
                              K);

    expect_mat_close(out_nt, ref_nt, 1e-4f, 1e-5f, "gemm_kernel_nt_tail_shape");
    expect_mat_close(out_nn, ref_nn, 1e-4f, 1e-5f, "gemm_kernel_nn_tail_shape");
    expect_mat_close(out_nn, out_nt, 1e-4f, 1e-5f, "gemm_kernel_nn_vs_nt");
}

TEST(Mat_TEST, runtime_weight_gemmNT_matches_reference_with_decode_pack)
{
    constexpr int Batch = 3;
    constexpr int K = 13;
    constexpr int N = 17;

    Mat input({Batch, 1, K}, CV_32F);
    Mat weight({N, K}, CV_32F);

    float* input_data = reinterpret_cast<float*>(input.data);
    float* weight_data = reinterpret_cast<float*>(weight.data);

    for (int idx = 0; idx < Batch * K; ++idx)
    {
        input_data[idx] = std::sin(static_cast<float>(idx) * 0.19f) * 0.8f + std::cos(static_cast<float>(idx) * 0.03f);
    }

    for (int idx = 0; idx < N * K; ++idx)
    {
        weight_data[idx] = std::sin(static_cast<float>(idx) * 0.09f) * 0.9f -
                           std::cos(static_cast<float>(idx) * 0.05f) * 0.35f;
    }

    RuntimeWeight rw_fp32;
    rw_fp32.init(weight, Int8QuantScheme::PerRow, true, RuntimePrecision::FP32);
    ASSERT_TRUE(rw_fp32.hasDecodePacked());
    expect_mat_close(rw_fp32.gemmNT(input), gemm(input, rw_fp32.active(), false, true), 1e-4f, 1e-5f, "runtime_weight_fp32_decode_pack");

    RuntimeWeight rw_fp16;
    rw_fp16.init(weight, Int8QuantScheme::PerRow, true, RuntimePrecision::FP16);
    expect_mat_close(rw_fp16.gemmNT(input), gemm(input, rw_fp16.active(), false, true), 5e-2f, 1e-2f, "runtime_weight_fp16_decode_pack");

    RuntimeWeight rw_int8;
    rw_int8.init(weight, Int8QuantScheme::PerRow, true, RuntimePrecision::INT8);
    expect_mat_close(rw_int8.gemmNT(input),
                     gemm(input, rw_int8.active(), rw_int8.int8Scales(), false, true),
                     2.5e-1f,
                     8e-2f,
                     "runtime_weight_int8_decode_pack");
}

TEST(Mat_TEST, test_mat_brodcast)
{
    float f20 = 20.f;
    float f30 = 30.f;
    float f50 = 50.f;
    Mat inpM1 = Mat({4}, CV_32F, reinterpret_cast<int&>(f20));
    Mat inpM2 = Mat({2, 3, 4}, CV_32F, reinterpret_cast<int&>(f30));

    Mat outM  = Mat({2, 3, 4}, CV_32F, reinterpret_cast<int&>(f50));

    Mat out;
    add(inpM1, inpM2, out);

    out.print();
    outM.print();
}

TEST(Mat_TEST, test_mat_trans)
{
    float f20 = 20.f;
    float f30 = 30.f;
    float f50 = 50.f;
    Mat inpM1 = Mat({4}, CV_32F, reinterpret_cast<int&>(f20));
    Mat inpM2 = Mat({2, 3, 4, 2}, CV_32F, reinterpret_cast<int&>(f30));

    Mat inpM3 = transposeND(inpM2, {0, 2, 1, 3});

    inpM2.print();
    inpM3.print();
}
