//
// Created by mzh on 2024/8/5. 考虑把这个文件放到core中
//

#ifndef CVH_UTILS_H
#define CVH_UTILS_H

#include <cstdint> // uint32_t, uint64_t, etc.
#include <string>
#include <vector>
#include "context.h"
#include "mat.h"

namespace cvh{

// taken from https://gist.github.com/zhuker/b4bd1fb306c7b04975b712c37c4c4075
float fp16_to_fp32(const uint16_t in);

// taken from https://gist.github.com/zhuker/b4bd1fb306c7b04975b712c37c4c4075
uint16_t fp32_to_fp16(const float in);

// convert the mat shape to char*, it is used to conveniently print mat dimension info.
std::string shape_to_str(const Mat& m);
std::string shape_to_str(const MatShape& shape);

// shape inferen of multi-dim gemm op
MatShape get_gemm_shape(const Mat& A, const Mat& B);
MatShape get_gemm_shape(const MatShape&A, const MatShape& B);

// get the argmax tokens from the logits
std::vector<int> argmax_tokens(const float* logits, int batch, int seq_len, int vocab_size);

const char* runtime_precision_name(RuntimePrecision precision);
bool parse_runtime_precision(const std::string& text, RuntimePrecision& precision);
RuntimePrecision parse_runtime_precision(const std::string& text);

void align_precision_sensitive_input(const Mat& input, RuntimePrecision precision, Mat& output);
Mat align_precision_sensitive_input(const Mat& input, RuntimePrecision precision);

}

#endif //CVH_UTILS_H
