# 项目总目标

[
\textbf{Goal A: } \text{构建两套实现（Non-Fused vs Fully-Fused）并验证数值一致性与性能提升。}
]

* 一致性：在相同输入上，fused 与 non-fused 的残差与梯度输出一致（在严格容差内）。
* 性能：相对 non-fused 达到显著加速与更低显存；可选与 PyTorch autograd 版对比（作为参考基线）。

[
\textbf{Goal B: } \text{将 fused 物理核与编码/MLP 接口打通，形成无 autograd 的一阶反向训练闭环。}
]

---

# 技术背景（核对公式）

连续/输运物理项残差：
[
R_{\sigma}=\partial_t\sigma+u\cdot\nabla\sigma+\sigma,(\nabla\cdot u)
]

简化动量项残差：
[
\mathbf{R}_{u}=\partial_t u+(u\cdot\nabla)u
]

总损失（权重 (w_\sigma,w_u)）：
[
L=w_\sigma,\frac{1}{N}\sum R_\sigma^2;+;w_u,\frac{1}{N}\sum \lVert \mathbf R_u\rVert_2^2
]

显式 VJP 系数（仅一阶反向）：
[
g_\sigma=2w_\sigma R_\sigma,\qquad \mathbf g_u=2w_u\mathbf R_u
]

---

# 里程碑（Milestones）与验收标准

## M0. 仓库初始化与基础设施

**目标**：建立可构建、可测试、可基准的最小工程。

* 交付物

    * CMake 工程（C++23 / CUDA 12.x），`/include`, `/src`, `/tests`, `/bench`, `/scripts`。
    * 编码规范：clang-format，警告视为错误。
    * 单元测试框架：GoogleTest；基准脚本：CUDA events + CSV 输出。
* 验收（定量）

    * 本地与 CI（Release）均能构建通过（0 失败）；`ctest` 通过率 100%。

---

## M1. 数据与度量基线（无内核）

**目标**：定义网格、权重、缓冲区接口；生成可控的测试场。

* API（头文件约定）

    * `GridSpec{nx,ny,nz,hx,hy,hz,dt}`，`PhysWeights{w_sigma,w_u}`。
    * `PhysBuf{sigma,u,sigma_prev,next,u_prev,next,d_sigma,d_u,loss_sigma,loss_u}`。
* 测试集

    1. 解析场（零残差）：(\sigma=\sin(x+y+z-t),\ u=(1,1,1))。
    2. 解析场（非零）：(\sigma=\sin(2x)+\cos(3y)-t,\ u=(\sin z,\cos x,\sin y))。
    3. 随机场（固定种子）：Uniform/Gaussian，范围可控。
       尺度：(64^3, 96^3, 128^3)，时间帧 (t-1,t,t+1)。
* 误差度量

    * 相对误差：(\mathrm{rel_err}(a,b)=\frac{\lVert a-b\rVert_2}{\lVert a\rVert_2+\varepsilon})。
    * 最大绝对误差：(\max_i |a_i-b_i|)。
* 验收（定量）

    * 数据生成与加载 0 失败；CSV 输出包含 case id / 规模 / dtype。

---

## M2. Non-Fused 基线（多核，一阶反向）

**目标**：用有限差分 + 显式 VJP，分多步 kernel 完成：导数 → 残差 → 反向。

* 内核（建议拆分）

    * `k_diff_all`：计算 (\partial_t\sigma,\ \nabla\sigma,\ \nabla\cdot u,\ \partial_t u,\ \nabla u) 并写入中间缓冲。
    * `k_residuals`：计算 (R_\sigma,\ \mathbf R_u) 与分项损失。
    * `k_backward_sigma` 与 `k_backward_u`：按显式 VJP 规则写 `d_sigma,d_u`。
* 边界处理与一致性

    * 缺邻点使用单边差分；同一规则将用于 fused 版本。
* 验收（定量）

    * 对 M1 的三类测试，FP32 残差与梯度数值稳定（零残差场中 (\max|R|<10^{-6})）。
    * 生成 Non-Fused 的耗时与显存基线（CSV）。

---

## M3. Fully-Fused 物理核（单核，一阶反向）

**目标**：在一个 kernel 内完成：装载邻域 → 差分（一次）→ 两个残差 → 块内规约 → 显式 VJP → 一次写回。

* 共享内存布局

    * Tile（例如 (8\times8\times4)）+ 三轴 (\pm 1) halo；必要时加载 (t\pm1) 对应 tile。
* 规约与写回

    * warp 规约损失；SMEM 累计梯度；块尾一次写回全局。
* 验收（定量）

    * 与 M2 对比（同数据、同差分策略、同 dtype）：

        * 数值一致性（FP32）：

            * (\mathrm{rel_err}(L_\sigma,L_\sigma^*)\le 1\mathrm{e}{-7})，(\mathrm{rel_err}(L_u,L_u^*)\le 1\mathrm{e}{-7})。
            * (\mathrm{rel_err}(d\sigma,d\sigma^*)\le 1\mathrm{e}{-6})，(\mathrm{rel_err}(d u,d u^*)\le 1\mathrm{e}{-6})，(\max|\cdot|\le 1\mathrm{e}{-6})。
        * 性能：速度提升 (\ge 1.5\times)（vox/s 或每步耗时对比）；显存峰值下降 (\ge 30%)。
    * 记录 Nsight 指标：SM 利用率、L2 命中、DRAM Bytes、原子冲突。

---

## M4. 精度与稳态扩展

**目标**：混合精度与稳定性开关。

* FP16/BF16 支持

    * 差分与规约内部保留 FP32 累加；输入/输出可 FP16。
    * 验收（定量）：FP16/BF16 一致性阈值 (\mathrm{rel_err}\le 1\mathrm{e}{-3})；速度较 FP32 (\ge 1.2\times)。
* 对流项迎风差分开关

    * 与中心差分一致性（在小速度场时差异应趋近 0）。
    * 在随机大速度场上提高稳定性（不爆 NaN）。

---

## M5. Profiling 与报告固化

**目标**：固定方法学，自动生成报告。

* `bench_compare` 程序

    * 输入参数：网格、dtype、差分策略、迭代次数。
    * 输出 CSV：case、规模、dtype、(T_\text{fused})、(T_\text{nonfused})、speedup、显存峰值、SM 利用率、L2 hit、误差指标。
* 自动图表

    * 速度对比柱状（不同网格/精度），误差箱线。
* 验收（定量）

    * 一键脚本产出 `report/`，包含 CSV + PNG + 简要结论（文本）。

---

## M6. 与编码/MLP 的无缝对接（闭环）

**目标**：将 fused 物理核与 tiny-cuda-nn 的 HashGrid+Fused-MLP 对接形成训练闭环（不含渲染）。

* 接口

    * 前向：((x,y,z,t)\overset{\text{encoding}}{\to}\text{feat}\overset{\text{MLP}}{\to}(\sigma,u))。
    * 物理核：读 ((\sigma,u)) 与 (t\pm1)；返还 (\partial L/\partial\sigma,\partial L/\partial u)。
    * MLP 反向：将上述梯度传给 MLP 权重；优化器更新。
* 收敛性验证

    * 在解析非零残差场上，最小化 (L)（或在构造数据上拟合目标速度/密度演化）；记录 (L) 随迭代下降曲线。
* 验收（定量）

    * 训练 (K) 步内，(L) 下降 (\ge 90%)（相对初始值）；吞吐量与 M3 保持同量级，无显著退化。

---

# 统一编码与接口规范

* 目录

  ```
  /include (public headers)
  /src     (kernels & impl)
  /tests   (gtest)
  /bench   (benchmark harness)
  /scripts (nsight, plots)
  /report  (csv, figures)
  ```
* 关键头文件（示例）

    * `physics_types.h`：`GridSpec`，`PhysWeights`，`PhysBuf`。
    * `physics_nonfused.h/.cu`：`run_nonfused(const GridSpec&, const PhysWeights&, const PhysBuf&)`。
    * `physics_fused.h/.cu`：`run_fused(const GridSpec&, const PhysWeights&, const PhysBuf&)`。
* 日志/可复现性

    * 固定随机种子；版本号写入 CSV；设备信息落盘。

---

# 验证方法细则与容差

* 误差度量

    * (L_\sigma,L_u) 对比：(\mathrm{rel_err}\le 1\mathrm{e}{-7})。
    * 梯度数组：(\mathrm{rel_err}\le 1\mathrm{e}{-6})，(\max|\cdot|\le 1\mathrm{e}{-6})。
    * FP16/BF16 放宽 (\mathrm{rel_err}\le 1\mathrm{e}{-3})。
* 计时

    * 预热 50 次；统计 200 次；报告均值与 P95。
    * vox/s 与每步毫秒需同时记录。
* Nsight 指标

    * 目标：SM 利用率 (\ge 80%)，L2 hit (\ge 90%)，显存峰值相对 non-fused 降 (\ge 30%)。

---

# 常见风险与约束（Agent 必须遵守）

1. **边界处理一致**：中心/单边差分与权重掩码在两路实现必须完全一致。
2. **规约确定性**：块内先 SMEM 规约再有限次原子写回，减少合并误差与非确定性。
3. **时间对齐**：(t\pm 1) 的 tile 必须空间对齐；stride/pitch 严格一致。
4. **混合精度**：中间量（残差/规约）用 FP32 累加；只在 I/O 层做 FP16。
5. **迎风/中心差分开关**：对比时必须保持一致。
6. **代码风格**：内核参数检查、错误返回、断言与越界保护不可省略。

---
