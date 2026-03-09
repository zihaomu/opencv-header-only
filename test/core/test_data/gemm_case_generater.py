import os
import numpy as np

ROOT_PATH = "./test/core/test_data/data"


def make_case(name, batch_a, batch_b, m, k, n, trans_a=False, trans_b=False, scale=1.0):
    return {
        "name": name,
        "batch_a": list(batch_a),
        "batch_b": list(batch_b),
        "m": int(m),
        "k": int(k),
        "n": int(n),
        "trans_a": bool(trans_a),
        "trans_b": bool(trans_b),
        "scale": float(scale),
    }


CASES = [
    make_case("nn_small_odd", [], [], 3, 5, 4),
    make_case("nn_tail_rect", [], [], 7, 11, 13),
    make_case("nn_rank3", [2], [2], 4, 6, 5),
    make_case("nn_rank4", [2, 3], [2, 3], 3, 7, 4),
    make_case("nn_broadcast_a", [1, 3], [2, 3], 5, 9, 4),
    make_case("nn_broadcast_b", [2, 1], [2, 4], 6, 8, 3),
    make_case("nn_rank_mismatch_a", [3], [2, 3], 4, 5, 6),
    make_case("nn_rank_mismatch_b", [2, 3], [3], 4, 5, 6),
    make_case("nt_small_odd", [], [], 3, 5, 4, False, True),
    make_case("nt_tail_rect", [], [], 9, 15, 7, False, True),
    make_case("nt_rank3", [2], [2], 5, 7, 6, False, True),
    make_case("nt_broadcast_a", [1, 3], [2, 3], 4, 9, 5, False, True),
    make_case("nt_broadcast_b", [2, 1], [2, 4], 4, 8, 5, False, True),
    make_case("nt_rank_mismatch", [3], [2, 3], 6, 10, 4, False, True),
    make_case("tn_basic", [], [], 5, 7, 4, True, False),
    make_case("tn_rank3", [2], [2], 4, 6, 5, True, False),
    make_case("tn_broadcast", [1, 3], [2, 3], 4, 6, 5, True, False),
    make_case("tt_basic", [], [], 3, 5, 4, True, True),
    make_case("tt_rank3", [2], [2], 4, 9, 6, True, True),
    make_case("tt_broadcast", [2, 1], [1, 3], 5, 7, 4, True, True),
    make_case("nn_large_value", [], [], 8, 16, 9, False, False, 100.0),
    make_case("nt_small_value", [], [], 8, 16, 9, False, True, 1e-3),
]


def make_tensor(shape, scale, rng):
    data = rng.standard_normal(shape).astype(np.float32)
    data *= np.float32(scale)
    return np.array(data, dtype=np.float32, order="C", copy=True)


def effective_shape(batch_shape, m, k, n, trans_a, trans_b):
    if trans_a:
        shape_a = tuple(batch_shape["a"] + [k, m])
    else:
        shape_a = tuple(batch_shape["a"] + [m, k])

    if trans_b:
        shape_b = tuple(batch_shape["b"] + [n, k])
    else:
        shape_b = tuple(batch_shape["b"] + [k, n])
    return shape_a, shape_b


def generate_case_data():
    os.makedirs(ROOT_PATH, exist_ok=True)

    # cleanup old generated files for this suite
    for fname in os.listdir(ROOT_PATH):
        if fname.startswith("gemm_") and fname.endswith(".npy"):
            os.remove(os.path.join(ROOT_PATH, fname))
    manifest_path = os.path.join(ROOT_PATH, "gemm_cases_manifest.txt")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    rng = np.random.default_rng(2026)
    lines = []
    for case in CASES:
        name = case["name"]
        m, k, n = case["m"], case["k"], case["n"]
        trans_a, trans_b = case["trans_a"], case["trans_b"]
        batch_shape = {"a": case["batch_a"], "b": case["batch_b"]}

        shape_a, shape_b = effective_shape(batch_shape, m, k, n, trans_a, trans_b)

        a = make_tensor(shape_a, case["scale"], rng)
        b = make_tensor(shape_b, case["scale"], rng)

        a_eff = np.swapaxes(a, -1, -2) if trans_a else a
        b_eff = np.swapaxes(b, -1, -2) if trans_b else b
        c_ref = np.array(np.matmul(a_eff, b_eff), dtype=np.float32, order="C", copy=True)

        np.save(os.path.join(ROOT_PATH, f"gemm_{name}_a.npy"), a)
        np.save(os.path.join(ROOT_PATH, f"gemm_{name}_b.npy"), b)
        np.save(os.path.join(ROOT_PATH, f"gemm_{name}_o.npy"), c_ref)

        lines.append(
            f"{name},transA={int(trans_a)},transB={int(trans_b)},"
            f"A={shape_a},B={shape_b},O={tuple(c_ref.shape)}"
        )
        print(f"[ok] {lines[-1]}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Generated {len(CASES)} GEMM cases into {ROOT_PATH}")


if __name__ == "__main__":
    generate_case_data()
