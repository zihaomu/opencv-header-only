#include "cvh.h"

#include "kernel/binary_kernel_xsimd.h"
#include "kernel/normalization_kernel_xsimd.h"
#include "kernel/transpose_kernel.h"

namespace cvh
{

inline void checkIsSameType(const Mat& a, const Mat& b)
{
    if (!(a.type() == b.type() && a.size == b.size))
        assert(0 && "The input Mats have different type or size!");
}

inline void checkIsSameType(const Mat& a, const Mat& b, const Mat& c)
{
    if (!((a.type() == b.type() && a.size == b.size) && (a.type() == c.type() && a.size == c.size)))
        assert(0 && "The input Mats have different type or size!");
}

inline
void preProcessAndCheck(const Mat& a, const Mat& b, Mat& c)
{
    if (c.empty())
    {
        checkIsSameType(a, b);

        c = Mat(a.dims, a.size.p, a.type());
    }
    else
        checkIsSameType(a, b, c);
}

inline
void preProcessAndCheck(const Mat& a, Mat& c)
{
    if (c.empty())
    {
        assert(!a.empty() && "The input mat can not be empty!");
        c = Mat(a.dims, a.size.p, a.type());
    }
    else
        checkIsSameType(a, c);
}

void add(const Mat& a, const Mat& b, Mat& c)
{
    binaryFunc(BinaryOp::ADD, a, b, c);
}

void addWeighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& c)
{
    preProcessAndCheck(a, b, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == CV_32F)
    {
        const float* ap = (const float*)a.data;
        const float* bp = (const float*)b.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = (float )(ap[i] * alpha + bp[i] * beta);
        }
    }
    else if (type == CV_32S)
    {
        const int* ap = (const int*)a.data;
        const int* bp = (const int*)b.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = (int )(ap[i] * alpha + bp[i] * beta);
        }
    }
    else
        M_Error_(Error::Code::StsBadType, ("Unsupported format at function \" addWeighted \" type = %d!", type));
}

void subtract(const Mat& a, const Mat& b, Mat& c)
{
    if (a.empty() && !b.empty())
    {
        subtract(b, c);
        return;
    }

    binaryFunc(BinaryOp::SUB, a, b, c);
}

void subtract(const Mat& a, Mat& c)
{
    preProcessAndCheck(a, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == CV_32F)
    {
        const float* ap = (const float*)a.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = -ap[i];
        }
    }
    else if (type == CV_32S)
    {
        const int* ap = (const int*)a.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = -ap[i];
        }
    }
    else
        M_Error_(Error::Code::StsBadType, ("Unsupported format at function \" subtract \" type = %d!", type));
}

void multiply(const Mat& a, const Mat& b, Mat& c)
{
    binaryFunc(BinaryOp::MUL, a, b, c);
}

void divide(const Mat& a, const Mat& b, Mat& c)
{
    binaryFunc(BinaryOp::DIV, a, b, c);
}

void softmax(const Mat& input, Mat& output)
{
    M_Assert(!input.empty() && "Softmax input mat can not be empty!");
    M_Assert(input.type() == CV_32F && "Currently only FP32 softmax is supported!");
    M_Assert(input.dims >= 1 && "Softmax expects at least 1 dimension!");

    if (output.empty())
    {
        output = Mat(input.dims, input.size.p, input.type());
    }
    else
    {
        M_Assert(output.type() == input.type());
        M_Assert(output.shape() == input.shape());
    }

    const size_t inner = input.size[input.dims - 1];
    const size_t outer = input.total() / inner;
    cpu::softmax_lastdim_xsimd(reinterpret_cast<const float*>(input.data),
                               reinterpret_cast<float*>(output.data),
                               outer,
                               inner);
}

Mat softmax(const Mat& input)
{
    Mat output;
    softmax(input, output);
    return output;
}

void silu(const Mat& input, Mat& output)
{
    M_Assert(!input.empty() && "SiLU input mat can not be empty!");
    M_Assert(input.type() == CV_32F && "Currently only FP32 SiLU is supported!");

    if (output.empty())
    {
        output = Mat(input.dims, input.size.p, input.type());
    }
    else
    {
        M_Assert(output.type() == input.type());
        M_Assert(output.shape() == input.shape());
    }

    cpu::silu_kernel_xsimd(reinterpret_cast<const float*>(input.data),
                          reinterpret_cast<float*>(output.data),
                          input.total());
}

Mat silu(const Mat& input)
{
    Mat output;
    silu(input, output);
    return output;
}

void rmsnorm(const Mat& input, const Mat& weight, Mat& output, float eps)
{
    M_Assert(!input.empty() && !weight.empty() && "RMSNorm input/weight can not be empty!");
    M_Assert(input.type() == CV_32F && "Currently only FP32 RMSNorm input is supported!");
    M_Assert((weight.type() == CV_32F || weight.type() == CV_16F) && "RMSNorm weight supports FP32/FP16!");
    M_Assert(weight.dims == 1 && "RMSNorm weight must be 1D!");
    M_Assert(weight.size[0] == input.size[input.dims - 1] && "RMSNorm weight size must match input last dimension!");

    if (output.empty())
    {
        output = Mat(input.dims, input.size.p, input.type());
    }
    else
    {
        M_Assert(output.type() == input.type());
        M_Assert(output.shape() == input.shape());
    }

    const size_t channels = input.size[input.dims - 1];
    const size_t outer = input.total() / channels;
    if (weight.type() == CV_32F)
    {
        cpu::rmsnorm_lastdim_xsimd(reinterpret_cast<const float*>(input.data),
                                   reinterpret_cast<const float*>(weight.data),
                                   reinterpret_cast<float*>(output.data),
                                   outer,
                                   channels,
                                   eps);
    }
    else
    {
        cpu::rmsnorm_lastdim_xsimd_fp16_weight(reinterpret_cast<const float*>(input.data),
                                               reinterpret_cast<const hfloat*>(weight.data),
                                               reinterpret_cast<float*>(output.data),
                                               outer,
                                               channels,
                                               eps);
    }
}

Mat rmsnorm(const Mat& input, const Mat& weight, float eps)
{
    Mat output;
    rmsnorm(input, weight, output, eps);
    return output;
}

void rmsnorm(const Mat& input, const Mat& weight, const Mat& weight_scales, Mat& output, float eps)
{
    M_Assert(!input.empty() && !weight.empty() && !weight_scales.empty() && "RMSNorm quantized input/weight can not be empty!");
    M_Assert(input.type() == CV_32F && "Currently only FP32 RMSNorm input is supported!");
    M_Assert(weight.type() == CV_8S && "Quantized RMSNorm expects INT8 weight!");
    M_Assert(weight_scales.type() == CV_32F && weight_scales.total() == 1 && "Quantized RMSNorm expects a single FP32 scale!");
    M_Assert(weight.dims == 1 && "RMSNorm weight must be 1D!");
    M_Assert(weight.size[0] == input.size[input.dims - 1] && "RMSNorm weight size must match input last dimension!");

    if (output.empty())
    {
        output = Mat(input.dims, input.size.p, input.type());
    }
    else
    {
        M_Assert(output.type() == input.type());
        M_Assert(output.shape() == input.shape());
    }

    const size_t channels = input.size[input.dims - 1];
    const size_t outer = input.total() / channels;
    cpu::rmsnorm_lastdim_xsimd_i8_weight(reinterpret_cast<const float*>(input.data),
                                         reinterpret_cast<const int8_t*>(weight.data),
                                         reinterpret_cast<const float*>(weight_scales.data),
                                         reinterpret_cast<float*>(output.data),
                                         outer,
                                         channels,
                                         eps);
}

Mat rmsnorm(const Mat& input, const Mat& weight, const Mat& weight_scales, float eps)
{
    Mat output;
    rmsnorm(input, weight, weight_scales, output, eps);
    return output;
}

void rope(Mat& q, Mat& k, int start_pos, float freq_base)
{
    M_Assert(!q.empty() && !k.empty() && "RoPE input mat can not be empty!");
    M_Assert(q.type() == CV_32F && k.type() == CV_32F && "Currently only FP32 RoPE is supported!");
    M_Assert(q.dims == 3 && k.dims == 3 && "RoPE expects [seq_len, head_count, head_dim] tensors!");
    M_Assert(q.size[0] == k.size[0] && "Q/K seq_len must match on RoPE!");
    M_Assert(q.size[2] == k.size[2] && "Q/K head_dim must match on RoPE!");
    M_Assert(q.size[2] % 2 == 0 && "RoPE head_dim must be even!");
    M_Assert(start_pos >= 0 && "RoPE start_pos must be non-negative!");

    cpu::rope_kernel_inplace(reinterpret_cast<float*>(q.data),
                             reinterpret_cast<float*>(k.data),
                             q.size[0],
                             start_pos,
                             q.size[1],
                             k.size[1],
                             q.size[2],
                             freq_base);
}

void compare(const Mat& a, const Mat& b, Mat& c, int op)
{
    // TODO add implementation!
    M_Error(NULL, "Un-implemented function at compare!");
}

Mat transpose(const Mat& input)
{
    M_Assert(!input.empty() && "The transpose function get empty input!");
    MatShape inpShape = input.shape();
    Mat out;
    if (inpShape.size() == 1)
    {
        out = input.clone();
        out.setSize({1, inpShape[0]});
    }
    else
    {
        int dim = inpShape.size();
        std::vector<int> out_order(dim);

        for (int i = 0; i < dim; i++)
        {
            out_order[i] = i;
        }

        std::swap(out_order[dim - 2], out_order[dim - 1]);
        out = transposeND(input, out_order);
    }

    return out;
}

Mat transposeND(const Mat& input, const std::vector<int> order)
{
    if (input.dims != order.size())
    {
        M_Error_(Error::StsBadSize, ("In transposeND, the input dimension is not equal to the order size! input dim = %d, order size = %d!", input.dims, static_cast<int>(order.size())));
    }

    auto order_ = order;
    std::sort(order_.begin(), order_.end());

    for (int i = 0; i < order_.size(); i++)
    {
        if (order_[i] != i)
        {
            M_Error(Error::StsBadSize, "New order shold be a valid permutation of the old one.");
        }
    }

    const auto& oldShape = input.shape();
    std::vector<int> newShape(order.size());
    for (int i = 0; i < order.size(); i++)
    {
        newShape[i] = oldShape[order[i]];
    }

    Mat out = Mat(newShape, input.type());

    bool is_last_two_swap = order.size() >= 2;
    for (int i = 0; i < order.size() && is_last_two_swap; ++i)
    {
        int expect = i;
        if (i == order.size() - 2)
            expect = order.size() - 1;
        else if (i == order.size() - 1)
            expect = order.size() - 2;

        if (order[i] != expect)
            is_last_two_swap = false;
    }

    if (is_last_two_swap)
    {
        const int rows = oldShape[oldShape.size() - 2];
        const int cols = oldShape[oldShape.size() - 1];
        const size_t elem_size = CV_ELEM_SIZE(input.type());
        const size_t plane_bytes = static_cast<size_t>(rows) * cols * elem_size;
        const size_t batch = input.total() / static_cast<size_t>(rows * cols);

        const unsigned char* src = input.data;
        unsigned char* dst = out.data;
        for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx)
        {
            cpu::transpose2d_kernel_blocked(src + batch_idx * plane_bytes,
                                            dst + batch_idx * plane_bytes,
                                            rows,
                                            cols,
                                            elem_size);
        }

        return out;
    }

    int continuous_idx = 0;
    for (int i = order.size() - 1; i >= 0; --i)
    {
        if (order[i] != i)
        {
            continuous_idx = i + 1;
            break;
        }
    }

    // minimum continuous size
    size_t continuous_size = 1;
    if (continuous_idx != order.size())
        continuous_size = total(newShape, continuous_idx);
    if (continuous_idx == 0)
        continuous_size = total(newShape);

    size_t step_in = 1;
    size_t step_out = 1;

    std::vector<size_t> inp_steps(order.size());
    std::vector<size_t> inp_steps_old(order.size());

    for (int i = order.size() - 1; i >= 0; --i)
    {
        inp_steps_old[i] = step_in;

        step_in *= oldShape[i];
    }

    for (int i = order.size() - 1; i >= 0; --i)
    {
        inp_steps[i] = inp_steps_old[order[i]];
    }

    size_t out_size = input.total() / continuous_size;

    auto* src = input.data;
    auto* dst = out.data;

    size_t src_offset = 0;
    size_t es = CV_ELEM_SIZE(out.type());
    size_t continuous_size_es = es * continuous_size;

    for (size_t i = 0; i < out_size; i++)
    {
//        src_offset = 0;
//        size_t src_left = i;
//
//        for (int j = 0; j < continuous_idx; j++)
//        {
//            int k = src_left/inp_steps[j];
//            src_left -= k * inp_steps[j];
//            src_offset += k * inp_steps[j];
//        }

        std::memcpy(dst, src + es * src_offset, continuous_size_es);
        dst += continuous_size_es;

        for (int j = continuous_idx - 1; j >= 0; --j)
        {
            src_offset += inp_steps[j];
            if ((src_offset / inp_steps[j]) % out.size[j] != 0)
            {
                break;
            }
            src_offset -= inp_steps[j] * out.size[j];
        }
    }

    return out;
}

/****************************************************************************************\
*                                  Mat normalization Implementation                                *
\****************************************************************************************/
template<typename _Tp> static double
norm_(const _Tp* src1, const _Tp* src2, size_t total, int normType, double startval)
{
    double result = startval;

    if (normType == NORM_INF)
    {
        for (size_t i = 0; i < total; i++)
        {
            result = std::max(result, std::abs((double )src1[i] - (double )src2[i]));
        }
    }
    else if (normType == NORM_L1)
    {
        for (size_t i = 0; i < total; i++)
        {
            result += std::abs((double )src1[i] - (double )src2[i]);

            if (std::abs((double )src1[i] - (double )src2[i]) > 0.1) {

            // Debug code
            // std::cout<<"src1["<<i<<"] = "<<(double )src1[i]<<", src2["<<i<<"] = "<<(double )src2[i]<<std::endl;
            }
        }
    }
    else if (normType == NORM_L2)
    {
        for (size_t i = 0; i < total; i++)
        {
            double s = (double )src1[i] - (double )src2[i];
            result += s * s;
        }
    }
    else
    {
        M_Error(Error::StsBadArg, "Unknown/unsupported norm type");
    }

    return result;
}

double norm(const Mat& a, const Mat& b, int normType)
{
    M_Assert(!a.empty() && !b.empty() && "The input mat can not be empty!");

    M_Assert(a.type() == b.type() && "Input data type is different!");

    M_Assert(a.total() == b.total() && "Input data total is different!");

    size_t total_size = a.total();

    double result = 0;
    switch (a.type()) 
    {
        case CV_8U:
            result = norm_((const uchar*)a.data, (const uchar*)b.data, total_size, normType, 0);
            break;
        case CV_8S:
            result = norm_((const char*)a.data, (const char*)b.data, total_size, normType, 0);
            break;
        case CV_16U:
            result = norm_((const ushort*)a.data, (const ushort*)b.data, total_size, normType, 0);
            break;
        case CV_16S:
            result = norm_((const short*)a.data, (const short*)b.data, total_size, normType, 0);
            break;
        case CV_32S:
            result = norm_((const int*)a.data, (const int*)b.data, total_size, normType, 0);
            break;
        case CV_32F:
            result = norm_((const float*)a.data, (const float*)b.data, total_size, normType, 0);
            break;
        case CV_64F:
            result = norm_((const double*)a.data, (const double*)b.data, total_size, normType, 0);
            break;
        default:
            M_Error(Error::StsBadArg, "Unknown/unsupported data type");
    };

    return result;
}

template<typename _Tp> static double
norm_(const _Tp* src1, size_t total, int normType, double startval)
{
    double result = startval;

    if (normType == NORM_INF)
    {
        for (size_t i = 0; i < total; i++)
        {
            result = std::max(result, std::abs((double )src1[i]));
        }
    }
    else if (normType == NORM_L1)
    {
        for (size_t i = 0; i < total; i++)
        {
            result += std::abs((double )src1[i]);
        }
    }
    else if (normType == NORM_L2)
    {
        for (size_t i = 0; i < total; i++)
        {
            double s = (double )src1[i];
            result += s * s;
        }
    }
    else
    {
        M_Error(Error::StsBadArg, "Unknown/unsupported norm type");
    }

    return result;
}

double norm(const Mat& a, int normType)
{
    M_Assert(!a.empty() && "The input mat can not be empty!");

    size_t total_size = a.total();

    double result = 0;

    switch (a.type())
    {
        case CV_8U:
            result = norm_((const uchar*)a.data, total_size, normType, 0);
            break;
        case CV_8S:
            result = norm_((const char*)a.data, total_size, normType, 0);
            break;
        case CV_16U:
            result = norm_((const ushort*)a.data, total_size, normType, 0);
            break;
        case CV_16S:
            result = norm_((const short*)a.data, total_size, normType, 0);
            break;
        case CV_32S:
            result = norm_((const int*)a.data, total_size, normType, 0);
            break;
        case CV_32F:
            result = norm_((const float*)a.data, total_size, normType, 0);
            break;
        case CV_64F:
            result = norm_((const double*)a.data, total_size, normType, 0);
            break;
        default:
            M_Error(Error::StsBadArg, "Unknown/unsupported data type");

    };

    return result;
}

/****************************************************************************************\
*                                  Reshape Implementation                                *
\****************************************************************************************/

void reshape(const Mat& input, const std::vector<int>& shape, Mat& out)
{
    M_Assert(!input.empty() && "The input mat can not be empty!");

    size_t total_size = input.total();
    size_t new_total = total(shape);

    M_Assert(total_size == new_total && "The total size of input mat is not equal to the new shape!");

    out = input;
    out.setSize(shape);
}

/****************************************************************************************\
*                                  Binary Op Implementation                              *
\****************************************************************************************/

// Helper contains all information used in binary BinaryOp.
class BinaryOpHelper
{
public:
    int max_dims;
    MatShape inp0_shape_align;   // contains new shape of input 0
    MatShape inp1_shape_align;
    MatShape out_shape;

    MatShape inp0_steps; // steps store all dimension jumping numbers of pointer.
    MatShape inp1_steps;
    MatShape out_steps;

    /* Reorganize mat for [block_num x block_size] based on out_shape.
     * For example: out shape is [a x b x c], the [a x b] is block_num, and c is inner block_size
     * */
    size_t block_size;
    size_t block_num;

    bool isInit = false;

    void init(const Mat& a, const Mat& b)
    {
        M_Assert(a.type() == b.type() && "Input data type is different!");

        MatShape shape0 = a.shape();
        MatShape shape1 = b.shape();

        int dim_0 = shape0.size();
        int dim_1 = shape1.size();

        max_dims = std::max(shape0.size(), shape1.size());

        // broadcasting the shape
        inp0_shape_align.resize(max_dims, 1);
        inp1_shape_align.resize(max_dims, 1);
        out_shape.resize(max_dims, 1);

        int idx_0 = dim_0 - 1;
        int idx_1 = dim_1 - 1;
        int idx = max_dims - 1; // shape loop cur

        while (idx >= 0)
        {
            const int s0 = idx_0 >= 0 ? shape0[idx_0] : 1;
            const int s1 = idx_1 >= 0 ? shape1[idx_1] : 1;

            if (s0 == s1)
            {
                out_shape[idx] = s0;
                inp0_shape_align[idx] = s0;
                inp1_shape_align[idx] = s1;
            }
            else if (s0 == 1)
            {
                out_shape[idx] = s1;
                inp0_shape_align[idx] = s0;
                inp1_shape_align[idx] = s1;
            }
            else if (s1 == 1)
            {
                out_shape[idx] = s0;
                inp0_shape_align[idx] = s0;
                inp1_shape_align[idx] = s1;
            }
            else
            {
                std::string str_0 = shape_to_str(a);
                std::string str_1 = shape_to_str(b);
                std::string log_info = "Broadcasting error! The two input shape are" + str_0 + " and " + str_1;
                M_Error(Error::StsBadSize, log_info.c_str());
            }

            idx_0--;
            idx_1--;
            idx--;
        }

        // set dim steps
        auto get_step_func = [](const MatShape& i_s, MatShape& o_s) {
            o_s.resize(i_s.size(), 1);
            size_t step = 1;
            for (int i = o_s.size() - 2; i >= 0; i--)
            {
                o_s[i] = 1;
            }

            for (int i = static_cast<int>(i_s.size()) - 1; i >= 0; --i)
            {
                if (i_s[i] == 1)
                {
                    o_s[i] = 0;
                }
                else
                {
                    o_s[i] = step;
                }

                step *= i_s[i];
            }
        };

        get_step_func(inp0_shape_align, inp0_steps);
        get_step_func(inp1_shape_align, inp1_steps);
        get_step_func(out_shape, out_steps);

        block_num = total(out_shape, 0, max_dims - 1);
        block_size = out_shape[max_dims - 1];
        isInit = true; // set isInit as true.
    }
};

inline bool to_kernel_op(BinaryOp op, cpu::BinaryKernelOp& kernel_op)
{
    switch (op)
    {
        case BinaryOp::ADD:
            kernel_op = cpu::BinaryKernelOp::Add;
            return true;
        case BinaryOp::SUB:
            kernel_op = cpu::BinaryKernelOp::Sub;
            return true;
        case BinaryOp::MUL:
            kernel_op = cpu::BinaryKernelOp::Mul;
            return true;
        case BinaryOp::DIV:
            kernel_op = cpu::BinaryKernelOp::Div;
            return true;
        default:
            return false;
    }
}

inline bool is_row_broadcast(const MatShape& aligned_shape, const MatShape& out_shape)
{
    if (aligned_shape.empty() || aligned_shape.size() != out_shape.size())
        return false;

    if (aligned_shape.back() != out_shape.back())
        return false;

    for (size_t i = 0; i + 1 < aligned_shape.size(); ++i)
    {
        if (aligned_shape[i] != 1)
            return false;
    }

    return true;
}

inline bool try_fast_binary_float(BinaryOp op, const BinaryOpHelper& helper, const Mat& a, const Mat& b, Mat& c)
{
    if (a.type() != CV_32F)
        return false;

    cpu::BinaryKernelOp kernel_op;
    if (!to_kernel_op(op, kernel_op))
        return false;

    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);
    float* pc = reinterpret_cast<float*>(c.data);
    const size_t total_num = total(helper.out_shape);
    const size_t inner = helper.out_shape.back();
    const size_t outer = total_num / inner;

    if (a.shape() == helper.out_shape && b.shape() == helper.out_shape)
    {
        cpu::binary_broadcast_xsimd(kernel_op, pa, inner, 1, pb, inner, 1, pc, outer, inner);
        return true;
    }

    if (a.total() == 1)
    {
        cpu::binary_broadcast_xsimd(kernel_op, pa, 0, 0, pb, inner, 1, pc, outer, inner);
        return true;
    }

    if (b.total() == 1)
    {
        cpu::binary_broadcast_xsimd(kernel_op, pa, inner, 1, pb, 0, 0, pc, outer, inner);
        return true;
    }

    if (is_row_broadcast(helper.inp0_shape_align, helper.out_shape) && b.shape() == helper.out_shape)
    {
        cpu::binary_broadcast_xsimd(kernel_op, pa, 0, 1, pb, inner, 1, pc, outer, inner);
        return true;
    }

    if (a.shape() == helper.out_shape && is_row_broadcast(helper.inp1_shape_align, helper.out_shape))
    {
        cpu::binary_broadcast_xsimd(kernel_op, pa, inner, 1, pb, 0, 1, pc, outer, inner);
        return true;
    }

    return false;
}

// TODO Optimized the following code.
template<typename T, typename Func>
void binary_forward(const Func& op, const BinaryOpHelper& helper,  const uchar* inp0, const uchar* inp1, uchar* out)
{
    M_Assert(helper.isInit && "BinaryOp has not been inited!");

    int max_dims = helper.max_dims;
    int block_size = helper.out_shape[helper.max_dims - 1];
    size_t total_num = total(helper.out_shape);
    int block_num = total_num / block_size;

    M_Assert(total_num % block_size == 0);
    const int esz = sizeof(T); // element size

    const int inner_0 = helper.inp0_shape_align[max_dims - 1] == 1 ? 0 : 1;
    const int inner_1 = helper.inp1_shape_align[max_dims - 1] == 1 ? 0 : 1;

    for (int bi = 0; bi < block_num; bi++)
    {
        // step 0: get output pointer
        T* p_o = (T*)(out + bi * block_size * esz);
        size_t jump0 = 0;
        size_t jump1 = 0;

        int idx = bi;
        for (int k = max_dims - 2; k >= 0; k--)
        {
            int next_idx = idx / helper.out_shape[k];
            int ik = idx - next_idx * helper.out_shape[k];
            jump0 += ik * helper.inp0_steps[k];
            jump1 += ik * helper.inp1_steps[k];
            idx = next_idx;
        }

        T* p_i0 = (T* )(inp0 + jump0 * esz);
        T* p_i1 = (T* )(inp1 + jump1 * esz);

        for (int i = 0; i < block_size; i++, p_o++, p_i0 += inner_0, p_i1 += inner_1)
        {
            *p_o = op(*p_i0, *p_i1);
            // std::cout<<"p_o["<<i<<"] = "<<*p_i0<<", "<<*p_i1<<", "<<*p_o<<std::endl;
        }
    }
}

template<typename T, typename... Args>
inline void opDispatch(const BinaryOp op, Args&&... args)
{
    if (std::is_same<T, float>::value)
    {
        M_Assert(op != BinaryOp::MOD && op != BinaryOp::AND && op != BinaryOp::OR && op != BinaryOp::XOR);
    }

    switch (op)
    {
        case BinaryOp::EQUAL:
        {
            auto equal = [](const T &a, const T &b) { return a == b; };
            binary_forward<T>(equal, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::GREATER:
        {
            auto greater = [](const T &a, const T &b) { return a > b; };
            binary_forward<T>(greater, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::GREATER_EQUAL:
        {
            auto greater_equal = [](const T &a, const T &b) { return a >= b; };
            binary_forward<T>(greater_equal, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::LESS:
        {
            auto less = [](const T &a, const T &b) { return a < b; };
            binary_forward<T>(less, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::LESS_EQUAL:
        {
            auto less_equal = [](const T &a, const T &b) { return a <= b; };
            binary_forward<T>(less_equal, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::POW:
        {
            auto pow = [] (const T& a, const T& b) { return std::pow(a, b); };
            binary_forward<T>(pow, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::BITSHIFT:
        {
            auto bitshift = [] (const uint8_t &a, const uint8_t &b) { return a << b; };
            binary_forward<T>(bitshift, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::MOD:
        {
            auto mod = [](const uint8_t &a, const uint8_t &b) { return a % b; };
            binary_forward<T>(mod, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::MUL:
        {
            auto mul = [](const T &a, const T &b) { return a * b; };
            binary_forward<T>(mul, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::SUB:
        {
            auto sub = [](const T &a, const T &b) { return a - b; };
            binary_forward<T>(sub, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::ADD:
        {
            auto add = [](const T &a, const T &b) { return a + b; };
            binary_forward<T>(add, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::DIV:
        {
            auto div = [](const T &a, const T &b) { return a / b; };
            binary_forward<T>(div, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::AND:
        {
            auto op_and = [](const uint8_t &a, const uint8_t &b) { return a & b; };
            binary_forward<T>(op_and, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::OR:
        {
            auto op_or = [](const uint8_t &a, const uint8_t &b) { return a | b; };
            binary_forward<T>(op_or, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::XOR:
        {
            auto op_xor = [](const uint8_t &a, const uint8_t &b) { return a ^ b; };
            binary_forward<T>(op_xor, std::forward<Args>(args)...);
            break;
        }
        default:
            M_Error_(Error::StsBadType, ("Unsupported op on Mat binary function! Op = %d!", (int)op));
    }
}

template<typename... Args>
inline void typeDispatch(const int type, Args&&... args)
{
    switch (type)
    {
        case CV_8U:
            opDispatch<uint8_t>(std::forward<Args>(args)...);
            break;
        case CV_8S:
            opDispatch<int8_t>(std::forward<Args>(args)...);
            break;
        case CV_16U:
            opDispatch<uint16_t>(std::forward<Args>(args)...);
            break;
        case CV_16S:
            opDispatch<int16_t>(std::forward<Args>(args)...);
            break;
        case CV_32S:
            opDispatch<int32_t>(std::forward<Args>(args)...);
            break;
        case CV_32F:
            opDispatch<float>(std::forward<Args>(args)...);
            break;
        default:
            M_Error_(Error::StsBadType, ("Unsupported type on Mat binary function! Type = %d!", type));
    }
}

void binaryFunc(BinaryOp op, const Mat& a, const Mat& b, Mat& c)
{
    M_Assert(a.type() == b.type());
    BinaryOpHelper helper = BinaryOpHelper();
    helper.init(a, b);

    if (c.empty())
    {
        c = Mat(helper.out_shape, a.type());
    }
    else
    {
        M_Assert(c.shape() == helper.out_shape);
        M_Assert(c.type() == a.type());
    }

    if (try_fast_binary_float(op, helper, a, b, c))
    {
        return;
    }

    typeDispatch(a.type(), op, helper, a.data, b.data, c.data);
}

}
