import os
import numpy as np

ROOT_PATH = "./test/core/test_data/data"


def ensure_root_path():
    os.makedirs(ROOT_PATH, exist_ok=True)


def save_case(name, a, b, out):
    np.save(os.path.join(ROOT_PATH, f"{name}_a.npy"), np.array(a, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, f"{name}_b.npy"), np.array(b, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, f"{name}_o.npy"), np.array(out, dtype=np.float32, order="C", copy=True))


def generate_binary_cases(rng):
    a = rng.standard_normal((2, 3, 4), dtype=np.float32)
    b = rng.standard_normal((2, 3, 4), dtype=np.float32)
    save_case("binary_add_same_shape", a, b, a + b)

    a = rng.standard_normal((2, 1, 4), dtype=np.float32)
    b = rng.standard_normal((1, 3, 1), dtype=np.float32)
    save_case("binary_sub_mid_broadcast", a, b, a - b)

    a = rng.standard_normal((4,), dtype=np.float32)
    b = rng.standard_normal((2, 3, 4), dtype=np.float32)
    save_case("binary_mul_row_broadcast", a, b, a * b)

    a = rng.standard_normal((2, 3, 4), dtype=np.float32)
    b = np.array([2.5], dtype=np.float32)
    save_case("binary_div_scalar_broadcast", a, b, a / b)


def generate_transpose_cases(rng):
    data = rng.standard_normal((2, 5, 7), dtype=np.float32)
    np.save(os.path.join(ROOT_PATH, "transpose_last2_3d_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(
        os.path.join(ROOT_PATH, "transpose_last2_3d_o.npy"),
        np.array(np.swapaxes(data, -1, -2), dtype=np.float32, order="C", copy=True),
    )

    data = rng.standard_normal((2, 3, 4, 5), dtype=np.float32)
    np.save(os.path.join(ROOT_PATH, "transpose_perm_4d_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(
        os.path.join(ROOT_PATH, "transpose_perm_4d_o.npy"),
        np.array(np.transpose(data, (0, 2, 1, 3)), dtype=np.float32, order="C", copy=True),
    )


def apply_rope(x, start_pos, freq_base):
    seq_len, head_count, head_dim = x.shape
    assert head_dim % 2 == 0

    half_dim = head_dim // 2
    inv_freq = 1.0 / (freq_base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    out = np.array(x, dtype=np.float32, order="C", copy=True)

    for pos in range(seq_len):
        cur_pos = start_pos + pos
        angles = cur_pos * inv_freq
        cos_v = np.cos(angles).astype(np.float32)
        sin_v = np.sin(angles).astype(np.float32)

        for h in range(head_count):
            real = x[pos, h, 0::2]
            imag = x[pos, h, 1::2]
            out[pos, h, 0::2] = real * cos_v - imag * sin_v
            out[pos, h, 1::2] = real * sin_v + imag * cos_v

    return out


def generate_rope_case(rng):
    q = rng.standard_normal((3, 2, 8), dtype=np.float32)
    k = rng.standard_normal((3, 1, 8), dtype=np.float32)
    np.save(os.path.join(ROOT_PATH, "rope_small_q_i.npy"), np.array(q, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rope_small_k_i.npy"), np.array(k, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rope_small_q_o.npy"), np.array(apply_rope(q, 5, 10000.0), dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rope_small_k_o.npy"), np.array(apply_rope(k, 5, 10000.0), dtype=np.float32, order="C", copy=True))


def generate_softmax_cases(rng):
    data = rng.standard_normal((4, 7), dtype=np.float32) * np.float32(2.0)
    ref = np.exp(data - np.max(data, axis=-1, keepdims=True))
    ref = ref / np.sum(ref, axis=-1, keepdims=True)
    np.save(os.path.join(ROOT_PATH, "softmax_lastdim_2d_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "softmax_lastdim_2d_o.npy"), np.array(ref, dtype=np.float32, order="C", copy=True))

    data = rng.standard_normal((2, 3, 4, 5), dtype=np.float32) * np.float32(1.5)
    ref = np.exp(data - np.max(data, axis=-1, keepdims=True))
    ref = ref / np.sum(ref, axis=-1, keepdims=True)
    np.save(os.path.join(ROOT_PATH, "softmax_lastdim_4d_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "softmax_lastdim_4d_o.npy"), np.array(ref, dtype=np.float32, order="C", copy=True))


def generate_silu_case(rng):
    data = rng.standard_normal((2, 3, 8), dtype=np.float32) * np.float32(2.0)
    ref = data / (1.0 + np.exp(-data))
    np.save(os.path.join(ROOT_PATH, "silu_basic_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "silu_basic_o.npy"), np.array(ref, dtype=np.float32, order="C", copy=True))


def generate_rmsnorm_cases(rng):
    data = rng.standard_normal((1, 5, 8), dtype=np.float32)
    weight = rng.standard_normal((8,), dtype=np.float32)
    eps = np.float32(1e-6)
    ref = data / np.sqrt(np.mean(data * data, axis=-1, keepdims=True) + eps) * weight
    np.save(os.path.join(ROOT_PATH, "rmsnorm_basic_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rmsnorm_basic_w.npy"), np.array(weight, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rmsnorm_basic_o.npy"), np.array(ref, dtype=np.float32, order="C", copy=True))

    data = rng.standard_normal((6, 16), dtype=np.float32) * np.float32(0.5)
    weight = rng.standard_normal((16,), dtype=np.float32)
    eps = np.float32(1e-5)
    ref = data / np.sqrt(np.mean(data * data, axis=-1, keepdims=True) + eps) * weight
    np.save(os.path.join(ROOT_PATH, "rmsnorm_rank2_i.npy"), np.array(data, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rmsnorm_rank2_w.npy"), np.array(weight, dtype=np.float32, order="C", copy=True))
    np.save(os.path.join(ROOT_PATH, "rmsnorm_rank2_o.npy"), np.array(ref, dtype=np.float32, order="C", copy=True))


def main():
    ensure_root_path()
    rng = np.random.default_rng(20260306)
    generate_binary_cases(rng)
    generate_transpose_cases(rng)
    generate_rope_case(rng)
    generate_softmax_cases(rng)
    generate_silu_case(rng)
    generate_rmsnorm_cases(rng)
    print(f"Generated operator cases into {ROOT_PATH}")


if __name__ == "__main__":
    main()
