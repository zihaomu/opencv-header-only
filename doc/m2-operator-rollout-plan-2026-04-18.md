# 测试来源（优先复用 OpenCV 官方用例）

- 测试数据来源：`/Users/moo/workssd/my_project/opencv_test/opencv_extra-4.x/testdata`
- 测试代码来源：`/Users/moo/workssd/my_project/opencv_test/opencv-4.13.0/modules/*/test`
- 复用策略：
  - 接口语义对齐优先移植 upstream `test` case（先做 contract 子集）。
  - 性能对比继续沿用 `opencv_compare`（CSV + Markdown 报告）。

# M2 新增算子落地计划（2周）

更新时间：2026-04-18  
状态：`Completed`（`Day1-Day10` 已完成）

## 1. 目标

在当前已完成 `Sobel/erode/dilate/morphologyEx/Canny` 的基础上，完成下一批“高频但缺口仍在”的算子闭环：

- `copyMakeBorder`
- `LUT`
- `warpAffine`
- `filter2D`
- `sepFilter2D`

每个算子都要满足：`API可用 + 合同测试 + upstream子集 + compare可观测`。

## 2. 边界

- 本轮优先 correctness，不把 SIMD/并行优化作为阻塞项。
- 先覆盖 `CV_8U/CV_32F` 主路径，复杂 dtype 逐步补。
- 先做 `imgproc` 高频参数组合，不追求 OpenCV 全参数面一次到位。

## 3. 两周排期（10个工作日）

### Week 1

1. Day 1：`copyMakeBorder` API + fallback + lite/full 编译接入（已完成）  
   验收：基础类型和常用 border 模式可跑通，header layout test 通过。
2. Day 2：`copyMakeBorder` 合同测试 + upstream 子集移植（已完成）  
   验收：`cvh_test_imgproc` 新增用例通过；同步脚本与 case manifest 更新。
3. Day 3：`LUT` API + fallback（`CV_8U` 主路径）（已完成）  
   验收：`C1/C3/C4` + 连续/ROI 路径正确。
4. Day 4：`LUT` 合同测试 + upstream 子集移植（已完成）  
   验收：异常输入、边界值、通道一致性全部覆盖。
5. Day 5：Week1 收口（文档 + compare 可观测行）  
   验收：`opencv_compare_quick.md` 能看到 `COPYMAKEBORDER/LUT` 行；全量 tests pass。

### Week 2

1. Day 6：`warpAffine` API + fallback（先 `INTER_NEAREST/LINEAR` + 常用 border）（已完成）  
   验收：基础仿射矩阵路径、ROI 路径可用。
2. Day 7：`warpAffine` 合同测试 + upstream 子集移植（已完成）  
   验收：几何正确性、边界模式、异常参数全部锁住。
3. Day 8：`filter2D` API + fallback（`CV_8U/CV_32F`）（已完成）  
   验收：常见 kernel 尺寸 + anchor/border 正确。
4. Day 9：`sepFilter2D` API + fallback + 与 `filter2D` 交叉校验（已完成）  
   验收：可分离卷积结果与 reference 一致。
5. Day 10：Week2 收口（upstream、compare、文档、风险清单）（已完成）  
   验收：新增算子全链路通过并形成对外可读报告。

## 4. 逐项验收标准（DoD）

每个新增算子必须同时满足：

1. 接口验收：
   - `include/cvh/imgproc/imgproc.h` 可见；
   - lite/full 模式均可编译。
2. 正确性验收：
   - 合同测试覆盖正常路径 + ROI/non-contiguous + 异常输入；
   - 至少 1 个 upstream case 子集对齐并标记 `PASS_NOW`。
3. 对比验收：
   - `opencv_compare` 中有对应 `op` 行，`status=OK`；
   - Markdown 报告可见该算子结果。
4. 工程验收：
   - `./build-m1/cvh_test_imgproc` 全量通过；
   - smoke 与现有 gate 不回归。

## 5. 统一执行命令（每个阶段收口时）

```bash
cmake --build build-m1 --target cvh_test_imgproc cvh_resize_dispatch_full_smoke -j8
./build-m1/cvh_test_imgproc
./build-m1/cvh_resize_dispatch_full_smoke

./opencv_compare/run_compare.sh --profile quick
```

## 6. 风险与对策

1. 风险：`warpAffine/filter2D` 参数面太大导致延期。  
   对策：先锁主路径参数，剩余参数分 M2.x 迭代。
2. 风险：upstream case 依赖大体量数据。  
   对策：优先选无大数据依赖的 TEST 子集，必要时补最小 fixture。
3. 风险：compare 时间过长影响迭代速度。  
   对策：开发期间固定 quick，收口时再跑 full。

## 7. 本轮交付定义

`M2 完成` 定义为：

- 上述 5 个算子全部具备 `API + test + upstream + compare`；
- `cvh_test_imgproc` 全量通过；
- `opencv_compare_quick.md` 可直接展示新增算子对比结果。

## 8. Day10 风险清单（2026-04-18）

1. 风险：`FILTER2D_3X3/SEPFILTER2D_3X3` 与 OpenCV 性能差距较大（当前 quick 报告 `OpenCV/CVH` 约 `0.002~0.02`）。  
   对策：M2.x 优先做可分离卷积与通道向量化优化，先攻 `CV_8U/CV_32F` + `C1/C3/C4` 热路径。
2. 风险：`LUT/COPYMAKEBORDER/WARP_AFFINE` 仍明显慢于 OpenCV，影响对外“可用即快”的感知。  
   对策：建立“correctness 通过后再优化”的二阶段基线，先补并行切分与内存访问优化，再引入 SIMD。
3. 风险：收口阶段 compare 耗时随算子扩展增长，迭代反馈变慢。  
   对策：开发期固定 `quick + small iter`，发布前补 `quick stable` 与 `full` 两档报告。
4. 关闭项：`Day7` warpAffine upstream 子集已补齐（`Imgproc_WarpAffine.accuracy`、`Imgproc_Warp.regression_19566` 已入 `case_manifest`）。  
   后续：M2.x 聚焦性能优化，不再阻塞 correctness 收口。
