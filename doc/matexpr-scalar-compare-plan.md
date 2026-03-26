# MatExpr Scalar + Compare 两阶段规划（/plan-ceo-review）

- 日期：2026-03-26
- 分支：`main`
- 建议模式：`HOLD_SCOPE`
- 目标：按你指定顺序完成
  1. 先扩展 MatExpr 到 `Scalar`
  2. 再实现 `> < >= <= !=`（并补齐比较语义）

## 0. 系统审计结论

### 0.1 当前状态（关键事实）

1. `MatExpr` 主实现在 [`src/core/mat_expr.cpp`](/home/moo/work/my_lab/opencv-header-only/src/core/mat_expr.cpp)，但未纳入构建源（`CMakeLists.txt` 里没有它）。
2. 比较底层里，`compare(const Mat&, const Mat&, ...)` 仍是 TODO（[`src/core/basic_op.cpp:551`](/home/moo/work/my_lab/opencv-header-only/src/core/basic_op.cpp:551)）。
3. `mat.h` 里比较运算符声明块仍在注释区（[`include/cvh/core/mat.h:318`](/home/moo/work/my_lab/opencv-header-only/include/cvh/core/mat.h:318)）。
4. 上游兼容点 `Core_MatExpr.issue_16655` 仍 pending（`!=` 结果类型语义未打通）。

### 0.2 What Already Exists（可复用）

1. `Scalar` 基础类型与 `Mat::setTo(Scalar)` 已具备。
2. `add/subtract/compare` 的 `Mat-Scalar/Scalar-Mat` 已实现并有单测（`channels>4` 抛错语义已冻结）。
3. `MatExpr` 已有 `+ - * /` 的 float 版本实现骨架（在 `src/core/mat_expr.cpp`）。

### 0.3 Dream State Mapping

```text
CURRENT STATE
MatExpr 对 Scalar 不完整；比较算子缺失；比较结果类型不稳定
      |
      v
THIS PLAN
两阶段补齐 Scalar 运算 + 比较运算，锁定 CV_8UC(ch) 输出语义
      |
      v
12-MONTH IDEAL
MatExpr 的高频算子（算术+比较）与 OpenCV 使用心智基本一致，可用于上游核心用例迁移
```

## 1. 0C-bis 实现路径对比

### APPROACH A（推荐，最小可交付）

- Summary：保持当前 MatExpr 结构不重构；先做 `Scalar` 重载，再做比较运算；实现以“可验证行为优先”，必要处可先走 eager（求值后再封装 MatExpr）。
- Effort：M（human: ~1.5-2 天 / CC: ~45-75 分钟）
- Risk：Low
- Pros：
  - 改动面最小，符合当前仓库节奏。
  - 与已落地 `Mat-Scalar` 语义复用度高。
  - 可快速解锁上游 pending 用例。
- Cons：
  - 短期不追求完整 OpenCV MatExpr 内部架构一致性。
  - 可能保留少量非关键性能债。
- Reuses：
  - `basic_op_scalar`、现有 `Scalar` 语义与单测框架。

### APPROACH B（中期平衡）

- Summary：把 `mat_expr.cpp` 正式接入构建，补齐 `MatOp_Cmp::assign/type`，并以临时 Mat 方式承载 Scalar。
- Effort：L（human: ~3-4 天 / CC: ~2-3 小时）
- Risk：Medium
- Pros：
  - 保留 MatExpr 懒求值主路径，结构更接近 OpenCV。
  - 后续扩展 `min/max/bitwise` 更顺滑。
- Cons：
  - 需先处理 `basic_op.cpp` 当前未构建/未完工依赖链。
  - 一次性触碰面较大，回归成本上升。

### APPROACH C（理想架构）

- Summary：完全对齐 OpenCV `MatExpr` 模型（含 Scalar 成员、完整 compare/op family）。
- Effort：XL（human: ~1-2 周 / CC: ~1 天）
- Risk：High
- Pros：
  - 长期最干净，最少技术债。
- Cons：
  - 超出“先落地两阶段目标”的必要范围。
  - 当前仓库状态下回报周期过长。

**RECOMMENDATION：A。**  
原因：你已经明确“两步走”，A 是最稳、最快、最可验收的路径。

## 2. 范围定义

## In Scope

1. MatExpr 增加 `Scalar` 重载（第一步）。
2. 实现比较运算符 `!= > < >= <=`（第二步）。
3. 比较结果类型语义固定：`CV_8UC(channels)`，mask 值 `0/255`。
4. 补齐对应单测与 upstream pending 文案更新。

## NOT in scope

1. `& | ^ min max abs` 全量落地。
2. MatExpr 全面重构到 OpenCV 同构内部实现。
3. merge/split/reinterpret 等非本主题能力。

## 3. 两阶段落地计划

## P1：MatExpr 扩展到 Scalar

### 改动点

1. 在 [`include/cvh/core/mat.h`](/home/moo/work/my_lab/opencv-header-only/include/cvh/core/mat.h) 增加 `Scalar` 运算符声明：
   - `+ - * /` 的 `Mat-Scalar`、`Scalar-Mat`、`MatExpr-Scalar`、`Scalar-MatExpr`。
2. 在 `MatExpr` 实现处增加统一辅助：
   - `materializeScalarLike(type, Scalar)`（按目标 type 生成 1 元 Mat 或直接走标量 op）。
   - `evalExpr(expr) -> Mat`（确保 MatExpr 与 Mat 混合输入可统一处理）。
3. 第一阶段优先保证语义正确与测试通过，可接受 eager 路径（表达式先求值再封装 MatExpr）。

### 验收标准

1. `Mat + Scalar`、`Scalar - MatExpr` 等组合可编译、可运行。
2. 多通道情况下 lane 对齐正确。
3. `channels>4` 统一抛 `StsBadArg`（继承既有规则）。

## P2：实现比较运算（`!= > < >= <=`）

### 改动点

1. 实现 `compare(Mat, Mat, dst, op)`（作为 MatExpr compare 的底座）。
2. 打开并落地 `mat.h` 中比较运算声明块（去掉 TODO 注释式占位，改成真实声明）。
3. 新增并实现：
   - `operator!=, >, <, >=, <=` 的 `Mat-Mat`、`Mat-Scalar`、`Scalar-Mat`（必要时覆盖 MatExpr 组合）。
4. `MatExpr` 比较表达式 `type()` 必须返回 `CV_8UC(channels)`，与 materialized Mat 一致。

### 验收标准

1. `MatExpr ab_expr = a != b;` 的 `ab_expr.type()` 与 `Mat(ab_expr).type()` 都是 `CV_8UC(ch)`。
2. 比较输出全部是 `0/255`。
3. upstream case `Core_MatExpr.issue_16655` 可从 pending 迁移到 runnable（或至少语义前置完成）。

## 4. Error & Rescue Registry

| Codepath | Failure Mode | Exception | Rescue Action | User Sees |
|---|---|---|---|---|
| `MatExpr + Scalar` | 输入 MatExpr 空 | `StsBadArg` | 立即抛错 | 明确错误 |
| `MatExpr + Scalar` | `channels>4` | `StsBadArg` | 立即抛错 | 明确错误 |
| `compare(Mat,Mat)` | shape/type 不匹配 | `StsBadArg` / `StsBadType` | 立即抛错 | 明确错误 |
| `compare(..., op)` | 非法 op | `StsBadArg` | 立即抛错 | 明确错误 |
| compare 输出 | dst type 错误 | `StsBadType` | 立即抛错 | 明确错误 |

## 5. Failure Modes Registry

| Codepath | Failure Mode | Rescued? | Test? | User Sees? | Logged? |
|---|---|---|---|---|---|
| `MatExpr !=` | compare 输出不是 `CV_8UC(ch)` | Y | Y | Error/TestFail | Y |
| `MatExpr op Scalar` | 标量 lane 映射错误 | Y | Y | Error/TestFail | Y |
| `MatExpr op Scalar` | ROI 非连续路径错误 | Y | Y | Error/TestFail | Y |
| `compare` | 非法 op 被静默忽略 | Y | Y | Error | Y |

## 6. 测试计划

1. 新增 `test/core/mat_expr_scalar_ops_test.cpp`：
   - `Mat + Scalar`、`Scalar - Mat`、`MatExpr * Scalar`、`Scalar / MatExpr`。
2. 新增 `test/core/mat_expr_compare_ops_test.cpp`：
   - `!= > < >= <=` 的双向重载。
   - 输出 type 与 mask 语义。
   - 多通道 + ROI + `channels>4` 异常。
3. 迁移/解锁 upstream 关键用例：
   - `Core_MatExpr.issue_16655`。

## 7. 部署与回滚

1. 合入顺序：先 P1（Scalar）后 P2（比较）。
2. 每步独立可回滚，避免一步改太大。
3. 每步要求 `cvh_test_core` 全量通过后再进入下一步。

## 8. 待你拍板（唯一关键决策）

### 决策：本轮是否接受 P1 的 eager MatExpr 路径（推荐）

- A（推荐）：先用 eager 保证行为与测试，后续再优化懒求值细节。  
- B：本轮必须懒求值完整落地（范围和风险显著上升）。

推荐 A，原因：与你当前“先打通语义再迭代”的节奏一致，且可最快把比较算子从 TODO 变成可验收功能。
