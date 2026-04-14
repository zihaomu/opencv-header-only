//
// Created by mzh on 2024/3/27.
//

#ifndef CVH_CORE_BASIC_OP_H
#define CVH_CORE_BASIC_OP_H

#include "mat.h"

namespace cvh
{

/*
TODO: all binary op 都需要对齐到opencv。
*/

// BinaryOp
enum BinaryOp
{
    AND = 0,
    EQUAL,
    GREATER,
    GREATER_EQUAL,
    LESS,
    LESS_EQUAL,
    OR,
    POW,
    XOR,
    BITSHIFT,
    MOD,  // Integer Mod. Reminder's sign = Divisor's sign.
    MUL,
    SUB,
    ADD,
    DIV,
    MAX,
    MIN,
    ATAN2,
    HYPOT,
    NOT,  // Bitwise and-not: a & (~b), integral depth only.
    SUM,  // Alias of ADD.
    FMOD, // Floating-point mod. Remainder keeps dividend sign.
    MEAN, // Element-wise arithmetic mean: (a + b) / 2.
};

void binaryFunc(BinaryOp op, const Mat& a, const Mat& b, Mat& c);

// a + b = c
void add(const Mat& a, const Mat& b, Mat& c);
void add(const Mat& a, const Scalar& b, Mat& c);
void add(const Scalar& a, const Mat& b, Mat& c);

// a * alpha + b * beta = c
void addWeighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& c);

// -a = c
void subtract(const Mat& a, Mat& c);

// a - b = c
void subtract(const Mat& a, const Mat& b, Mat& c);
void subtract(const Mat& a, const Scalar& b, Mat& c);
void subtract(const Scalar& a, const Mat& b, Mat& c);
void subtract(const Mat& a, double b, Mat& c);
void subtract(double a, const Mat& b, Mat& c);

// a * b = c
void multiply(const Mat& a, const Mat& b, Mat& c);
void multiply(const Mat& a, const Scalar& b, Mat& c);
void multiply(const Scalar& a, const Mat& b, Mat& c);

// a / b = c
void divide(const Mat& a, const Mat& b, Mat& c);
void divide(const Mat& a, const Scalar& b, Mat& c);
void divide(const Scalar& a, const Mat& b, Mat& c);

void compare(const Mat& a, const Mat& b, Mat& c, int op);
void compare(const Mat& a, const Scalar& b, Mat& c, int op);
void compare(const Scalar& a, const Mat& b, Mat& c, int op);

// Merge multiple source Mats into a multi-channel destination Mat.
void merge(const Mat* src, size_t nsrc, Mat& dst);
void merge(const std::vector<Mat>& src, Mat& dst);

// Split multi-channel source Mat into single-channel destination Mats.
void split(const Mat& src, Mat* dst);
void split(const Mat& src, std::vector<Mat>& dst);

// Apply softmax along the last dimension.
void softmax(const Mat& input, Mat& output);
Mat softmax(const Mat& input);

// Apply SiLU elementwise: x / (1 + exp(-x))
void silu(const Mat& input, Mat& output);
Mat silu(const Mat& input);

// Apply RMSNorm along the last dimension using 1D weight.
void rmsnorm(const Mat& input, const Mat& weight, Mat& output, float eps = 1e-6f);
Mat rmsnorm(const Mat& input, const Mat& weight, float eps = 1e-6f);
void rmsnorm(const Mat& input, const Mat& weight, const Mat& weight_scales, Mat& output, float eps = 1e-6f);
Mat rmsnorm(const Mat& input, const Mat& weight, const Mat& weight_scales, float eps = 1e-6f);

// Apply rotary positional embedding to Q/K in place.
// q shape: [seq_len, head_count, head_dim]
// k shape: [seq_len, head_count_kv, head_dim]
void rope(Mat& q, Mat& k, int start_pos = 0, float freq_base = 10000.0f);

// Transpose Mat last two dimension, if the Mat is one dimension, add the axis to the shape.
Mat transpose(const Mat& input);

// transpose Mat according to the input mat and the given new order.
Mat transposeND(const Mat& input, const std::vector<int> order);

enum NormType
{
    NORM_L1 = 1,
    NORM_L2 = 2,
    NORM_INF = 3,
};

// compute the norm of the Mat.
double norm(const Mat& a, int normType);
double norm(const Mat& a, const Mat& b, int normType);

// reshape Mat according to the given shape.
void reshape(const Mat& input, const std::vector<int>& shape, Mat& out);

#define MAT_AUG_OPERATOR1(op, cvop) \

#define MAT_AUG_OPERATOR(op, cvop) \
static inline Mat &operator op (Mat& a, const Mat& b) {cvop; return a;} \
static inline const Mat &operator op (const Mat& a, const Mat& b) {cvop; return a;}

MAT_AUG_OPERATOR(+=, add(a, b, (Mat &) a))
MAT_AUG_OPERATOR(-=, subtract(a, b, (Mat &) a))
MAT_AUG_OPERATOR(*=, multiply(a, b, (Mat &) a))
MAT_AUG_OPERATOR(/=, divide(a, b, (Mat &) a))

}
#endif //CVH_BASIC_OP_H
