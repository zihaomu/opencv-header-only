# Scalar 优先接入 + Mat/Scalar 协作规划（/plan-ceo-review）

- 日期：2026-03-26
- 分支：`main`
- 模式：`HOLD_SCOPE`（按你明确要求：先支持 Scalar，再让 Mat 支持 Scalar）
- 目标：优先补齐 OpenCV-like 的 `Scalar` 使用链路，作为后续 channel 相关兼容能力的基础能力

## 1. Step 0 结论

### 1.1 Premise Challenge（问题重构）

- 真实问题不是“Mat 的 channel 编码”，而是“缺少 `Scalar` 导致大量 OpenCV 常用入口无法表达”。
- 业务结果是：先解锁 `Mat = Scalar` / `setTo(Scalar)` / `add-subtract-compare` 的 `Mat-Scalar` 组合，推动 channel pending 用例落地。
- 若什么都不做：`subtract(scalar, mat)`、`compare(..., scalar)`、`Mat::operator=(Scalar)` 这类典型 API 继续空缺，上游兼容测试持续 pending。

### 1.2 Existing Code Leverage（现有复用）

- 已有 `Mat::setTo(float)` 与 `Mat::operator=(float/int)` 可复用为 `Scalar` 路径底座。
- 已有 `basic_op` 的 `Mat-Mat` 路径可复用内核与输出分配逻辑。
- 已有 `channel` 合同与 layout 语义测试可作为回归基线。

### 1.3 Dream State Mapping

```text
CURRENT STATE                         THIS PLAN                               12-MONTH IDEAL
Mat 仅支持 float/int 赋值     --->    Scalar 类型 + Mat/Scalar API      --->   OpenCV core 高频算子的
Mat-Scalar 算子入口缺失               + 明确错误语义 + 覆盖测试                   Scalar 兼容稳定层
```

## 2. 0C-bis：实现方案对比（必选）

### APPROACH A（最小可交付）

- Summary：新增 `Scalar` 类型，先覆盖 `Mat=Scalar`、`setTo(Scalar)`、`add/subtract/compare` 的标量重载，限定 `Mat` 通道数 `1..4`。
- Effort：`M`
- Risk：`Low`
- Pros：
  - 直接解决当前兼容阻塞点。
  - diff 小，回归面可控。
  - 与当前架构（Mat 主体 + basic_op）一致。
- Cons：
  - `channels > 4` 的标量语义需明确拒绝。
  - 后续若扩展更多算子，需要持续补重载。
- Reuses：
  - `Mat::setTo(float)`、`basic_op`、现有 core 测试框架。

### APPROACH B（理想架构）

- Summary：一次性建立 `ScalarAdapter`/`ScalarExpr`，统一所有 `Mat-Scalar`、`Scalar-Mat`、`MatExpr-Scalar` 路径。
- Effort：`L`
- Risk：`Medium`
- Pros：
  - 长期扩展成本低。
  - 能减少后续重复实现。
- Cons：
  - 当前仓库规模下偏重，首步落地慢。
  - 引入新抽象，短期调试复杂度更高。
- Reuses：
  - 复用 `MatExpr` 框架，但需要较多重构。

### APPROACH C（折中）

- Summary：不新增 `Scalar` 类型，临时用 `Mat(1x1xC)` 代替标量。
- Effort：`S`
- Risk：`High`
- Pros：
  - 开发快。
- Cons：
  - 对外 API 不兼容 OpenCV 心智。
  - 语义隐晦，长期债务高。
- Reuses：
  - 仅复用现有 Mat 路径。

**RECOMMENDATION：选 APPROACH A。**  
理由：满足你“先支持 Scalar”的目标，且最符合“最小 diff + 高可验证性”的工程偏好。

## 3. 范围定义（本次规划）

## In Scope

1. 新增 `cvh::Scalar` 类型（4 lane，double 存储，含 `all()`）。
2. `Mat` 支持：
   - `operator=(const Scalar&)`
   - `setTo(const Scalar&)`
3. `basic_op` 新增重载：
   - `add(Mat, Scalar, Mat&)`
   - `add(Scalar, Mat, Mat&)`
   - `subtract(Mat, Scalar, Mat&)`
   - `subtract(Scalar, Mat, Mat&)`
   - `compare(Mat, Scalar, Mat&, op)`
   - `compare(Scalar, Mat, Mat&, op)`
4. 错误语义与测试覆盖（含 ROI/non-contiguous + 多通道）。

## NOT in scope

1. `MatExpr` 全量 `Scalar` 运算融合（原因：范围过大，首步非阻塞）。
2. `merge/split/reinterpret` 实现（原因：属于下一批 channel 关键能力）。
3. `channels > 4` 的 Scalar 广播扩展（原因：先锁定 v1 行为，避免语义分叉）。

## 4. 语义合同（新增/修订）

### 4.1 Scalar 语义

- `Scalar` 是 4 元向量，默认 `[0,0,0,0]`。
- 构造：
  - `Scalar(v0, v1=0, v2=0, v3=0)`
  - `Scalar::all(v)`
- lane 访问：`operator[](i)`，`i in [0,3]`。

### 4.2 Mat 与 Scalar 协作规则

- `Mat::operator=(Scalar)`：将 scalar 按通道写入所有像素。
- `Mat::setTo(Scalar)`：同上，返回与 `setTo(float)` 一致的类型转换行为（saturate cast）。
- 对 `channels <= 4`：
  - 使用 `scalar[ch]`。
- 对 `channels > 4`：
  - v1 明确抛错：`StsBadArg`（避免隐式重复/截断歧义）。

### 4.3 二元算子规则

- `add/subtract/compare` 中，`Scalar` 先转换到目标 `Mat` depth，再按通道逐元素处理。
- 处理顺序与多通道规则：每个通道独立运算。

## 5. 系统架构与数据流

### 5.1 架构图

```text
User API
  |
  |-- Mat::operator=(Scalar) / Mat::setTo(Scalar)
  |-- add/subtract/compare (Mat, Scalar) overloads
  v
Scalar Normalize Layer
  - validate channels
  - cast scalar lanes to dst depth
  v
Core Compute Path
  - continuous fast path
  - stride-aware fallback path
  v
Mat data buffer
```

### 5.2 数据流（以 `add(Mat, Scalar, dst)` 为例）

```text
INPUT(src, scalar) -> VALIDATE -> CAST scalar lanes -> ALLOC/REUSE dst -> COMPUTE -> OUTPUT
      |                |             |                    |                 |         |
      | nil src        | bad channels| cast overflow      | alloc fail      | op err  | dst
      v                v             v                    v                 v         v
   StsBadArg        StsBadArg     saturate rules      StsNoMem         StsBadType  correct
```

## 6. Error & Rescue Registry（首批）

| Codepath | Failure Mode | Exception | Rescue Action | User Sees |
|---|---|---|---|---|
| `Mat::setTo(Scalar)` | empty mat | `StsBadArg` | 直接报错 | 明确错误 |
| `Mat::setTo(Scalar)` | `channels > 4` | `StsBadArg` | 直接报错 | 明确错误 |
| `add/subtract/compare` scalar path | depth 不支持 | `StsNotImplemented` | 直接报错 | 明确错误 |
| scalar cast | 数值溢出 | 无异常（saturate） | 饱和转换 | 稳定输出 |
| dst create | 分配失败 | `StsNoMem` | 抛错 | 明确错误 |

## 7. Failure Modes Registry

| Codepath | Failure Mode | Rescued? | Test? | User Sees? | Logged? |
|---|---|---|---|---|---|
| `setTo(Scalar)` | `channels > 4` | Y | Y | Error | Y |
| `add(Mat,Scalar)` | src empty | Y | Y | Error | Y |
| `compare(Scalar,Mat)` | type unsupported | Y | Y | Error | Y |
| scalar cast | NaN/Inf 输入 | Y（按 cast 规则） | Y | Deterministic result | Y |

## 8. 测试计划（先做完再扩）

### 8.1 新增单测文件建议

- `test/core/scalar_contract_test.cpp`
- `test/core/mat_scalar_ops_test.cpp`

### 8.2 必测清单

1. `Scalar` 构造、`all()`、索引访问。
2. `Mat::operator=(Scalar)` 在 `C1/C2/C3/C4` 的正确性。
3. `Mat::setTo(Scalar)` 在连续与 ROI/non-contiguous 上一致性。
4. `add/subtract/compare` 的 `Mat-Scalar` 与 `Scalar-Mat` 对称性。
5. `channels > 4` 明确报错路径。
6. 边界值：负数、溢出、NaN/Inf（float depth）。

### 8.3 Upstream pending 对齐

- 优先解锁：
  - `Subtract.scalarc1_matc3`
  - `Subtract.scalarc4_matc4`
- 仍待后续：
  - `compare.*`（若本次 compare 重载完成，可同步推进一部分）

## 9. 执行分阶段（人类/CC 双刻度）

1. P1：`Scalar` 类型与 `Mat` 赋值支持  
人类：~0.5 天 / CC：~20-30 分钟
2. P2：`basic_op` 的 Mat-Scalar/Scalar-Mat 重载  
人类：~1 天 / CC：~30-45 分钟
3. P3：测试补齐 + pending 升级  
人类：~1 天 / CC：~30-45 分钟
4. P4：文档合同同步（`mat-contract` + gap tracker）  
人类：~0.5 天 / CC：~15-20 分钟

## 10. 需要你拍板的唯一语义决策

### 决策：`channels > 4` 时 `Scalar` 行为

- 方案 A（推荐）：v1 直接报错，后续版本再扩展。
- 方案 B：按 `i % 4` 循环广播。
- 方案 C：只用 `scalar[0]` 广播全部通道。

推荐 A，原因：最符合“显式优于隐式”，可避免 silent wrong result。

### 已确认决策（2026-03-26）

- 你已选择：**A**
- 生效规则：在 v1 中，当 `Mat::channels() > 4` 且调用任意 `Mat-Scalar` / `Scalar-Mat` 路径时，统一返回 `StsBadArg`，不做隐式循环广播或单值广播。
- 影响范围：`Mat::operator=(Scalar)`、`Mat::setTo(Scalar)`、`add/subtract/compare` 的 Scalar 重载。
