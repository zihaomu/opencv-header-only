import os
import numpy as np

ROOT_PATH = "./test/core/test_data/data"


def softmax_lastdim(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(np.float32, copy=False)


def quantize_rowwise(src: np.ndarray):
    src = np.asarray(src, dtype=np.float32)
    rows = src.shape[0]
    scales = np.empty((rows,), dtype=np.float32)
    dst = np.empty_like(src, dtype=np.int8)
    for row in range(rows):
        max_abs = np.max(np.abs(src[row]))
        scale = max_abs / 127.0 if max_abs > np.finfo(np.float32).eps else 1.0
        scales[row] = np.float32(scale)
        q = np.round(src[row] / scale)
        dst[row] = np.clip(q, -127.0, 127.0).astype(np.int8)
    return dst, scales


def generate_softmax_cases():
    input_shape = (2, 3, 7)
    indices = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    softmax_input = (np.sin(indices * 0.17).astype(np.float32) * 1.25 +
                     np.cos(indices * 0.11).astype(np.float32) * 0.65)
    softmax_input = np.ascontiguousarray(softmax_input.astype(np.float32))

    softmax_fp32 = np.ascontiguousarray(softmax_lastdim(softmax_input))
    softmax_fp16_aligned_input = np.ascontiguousarray(softmax_input.astype(np.float16).astype(np.float32))
    softmax_fp16 = np.ascontiguousarray(softmax_lastdim(softmax_fp16_aligned_input))

    np.save(os.path.join(ROOT_PATH, "softmax_precision_input.npy"), softmax_input)
    np.save(os.path.join(ROOT_PATH, "softmax_precision_fp32_o.npy"), softmax_fp32)
    np.save(os.path.join(ROOT_PATH, "softmax_precision_fp16floor_o.npy"), softmax_fp16)


def generate_linear_cases():
    input_shape = (1, 3, 8)
    weight_shape = (6, 8)
    bias_shape = (6,)

    input_idx = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    weight_idx = np.arange(np.prod(weight_shape), dtype=np.float32).reshape(weight_shape)
    bias_idx = np.arange(np.prod(bias_shape), dtype=np.float32)

    linear_input = np.ascontiguousarray(
        (np.sin(input_idx * 0.13).astype(np.float32) * 0.8 +
         np.cos(input_idx * 0.07).astype(np.float32) * 0.35).astype(np.float32)
    )
    linear_weight = np.ascontiguousarray(
        (np.sin(weight_idx * 0.09).astype(np.float32) * 0.55 -
         np.cos(weight_idx * 0.05).astype(np.float32) * 0.25).astype(np.float32)
    )
    linear_bias = np.ascontiguousarray(
        (np.sin(bias_idx * 0.15).astype(np.float32) * 0.2 +
         np.float32(0.05)).astype(np.float32)
    )

    linear_weight_fp16 = np.ascontiguousarray(linear_weight.astype(np.float16))
    linear_weight_i8, linear_weight_i8_scales = quantize_rowwise(linear_weight)
    linear_weight_i8_dequant = np.ascontiguousarray(
        linear_weight_i8.astype(np.float32) * linear_weight_i8_scales[:, None]
    )

    linear_fp32_o = np.ascontiguousarray(np.matmul(linear_input, linear_weight.T) + linear_bias)
    linear_fp16_o = np.ascontiguousarray(
        np.matmul(linear_input, linear_weight_fp16.astype(np.float32).T) + linear_bias
    )
    linear_int8_o = np.ascontiguousarray(
        np.matmul(linear_input, linear_weight_i8_dequant.T) + linear_bias
    )

    np.save(os.path.join(ROOT_PATH, "linear_precision_input.npy"), linear_input)
    np.save(os.path.join(ROOT_PATH, "linear_precision_weight.npy"), linear_weight)
    np.save(os.path.join(ROOT_PATH, "linear_precision_bias.npy"), linear_bias)
    np.save(os.path.join(ROOT_PATH, "linear_precision_fp32_o.npy"), linear_fp32_o)
    np.save(os.path.join(ROOT_PATH, "linear_precision_fp16_o.npy"), linear_fp16_o)
    np.save(os.path.join(ROOT_PATH, "linear_precision_int8_o.npy"), linear_int8_o)


def main():
    os.makedirs(ROOT_PATH, exist_ok=True)
    generate_softmax_cases()
    generate_linear_cases()
    print(f"Generated precision cases into {ROOT_PATH}")


if __name__ == "__main__":
    main()
