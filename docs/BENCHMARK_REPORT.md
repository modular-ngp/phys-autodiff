# Benchmark Report: 3D Smoke Physics Loss (Fused vs Non‑Fused)

## 1) Executive Summary
- Physics residuals stage: Fully‑Fused is 1.4–1.7x faster end‑to‑end (up to 1.77x kernel‑only) than Non‑Fused.
- With MLP inference in the loop (MLP→Physics end‑to‑end), total latency is dominated by MLP, so overall speedup is 1.03–1.08x, while the physics stage itself still gains 1.40–1.59x.
- All numerical checks pass: CPU↔CUDA and Fused↔Non‑Fused match within tight tolerances for residuals and backward.

## 2) Environment
- OS/Toolchain: Windows, CMake 3.31.6, PowerShell 7.5.4
- CUDA Toolkit: 13.0 (V13.0.88)
- GPU: GeForce RTX 5090 (driver 581.57, 32 GB)
- CPU: AMD Ryzen 9 9950X 16‑Core
- Build: Release, `CMAKE_CUDA_ARCHITECTURES=75;86;89;90`

## 3) What We Measured
- Physics residuals (transport‑style) using central differences in space/time.
  - Non‑Fused: multiple kernels, intermediate derivatives written to global memory.
  - Fully‑Fused: single kernel computes all derivatives/residuals (with single‑kernel backward variant).
- MLP‑driven fields at t−dt, t, t+dt: input [x,y,z,t], output [sigma, ux, uy, uz]; random weights with fixed seed.
- Timing scopes:
  - Kernel‑only (physics): CUDA events around kernels.
  - Physics E2E: wall‑clock around wrappers (includes device alloc/free and H2D/D2H copies).
  - Total E2E (MLP+Physics): MLP grid inference + physics wrapper.
- Grids: 64×64×64, 96×96×64, 128×96×96; iters=10, warmup=2.

## 4) Correctness (All Pass)
- `test_phys_cpu_ref`: manufactured solution vs discrete analytic (central differences).
- `test_phys_cuda_nonfused_vs_cpu`: Non‑Fused vs CPU.
- `test_phys_cuda_fused_vs_nonfused`: Fused vs Non‑Fused (residuals and backward).
- `test_mlp_grid_infer`: MLP grid inference CPU vs CUDA.
- `test_mlp_phys_integration_inputs`: MLP‑generated field sizes and NaN checks.

## 5) Headline Results
- Physics‑only E2E (no MLP): Fused improves 29.9%–41.4%.
- MLP+Physics Total E2E: Fused improves 3.2%–8.0% (total is MLP‑dominated).
- Kernel‑only (physics): Fused up to 1.77x (+43.6%).

## 6) Detailed Results
Physics‑only (no MLP), executable: `test_phys_perf`

| Grid | Non‑Fused ms | Fused ms | Speedup | Gain |
|---|---:|---:|---:|---:|
| 64×64×64 | 3.61493 | 2.20080 | 1.64x | +39.1% |
| 96×96×64 | 7.06326 | 4.13635 | 1.71x | +41.4% |
| 128×96×96 | 9.54905 | 6.69626 | 1.43x | +29.9% |

MLP + Physics (end‑to‑end), executable: `test_mlp_phys_perf`

| Grid | Kernel‑only NF | Kernel‑only F | Speedup | Physics E2E NF | Physics E2E F | Speedup | Total E2E NF | Total E2E F | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64×64×64 | 0.07864 | 0.06750 | 1.16x | 3.8968 | 2.4523 | 1.59x | 23.0356 | 21.3113 | 1.08x |
| 96×96×64 | 0.13000 | 0.07349 | 1.77x | 7.1276 | 4.5631 | 1.56x | 47.0586 | 44.1871 | 1.07x |
| 128×96×96 | 0.18340 | 0.11997 | 1.53x | 10.2412 | 7.2931 | 1.40x | 87.5883 | 84.7648 | 1.03x |

Note: Total E2E = MLP grid inference + physics wrapper. As grids grow, the MLP portion dominates, masking some physics‑stage gains.

## 7) Interpretation and Recommendations
- Why Fused wins: fewer global memory round trips for intermediates; more locality.
- Why Total E2E gains are modest: in this pipeline, three MLP evaluations dominate total time; physics speedups are a smaller share of the total.
- Next steps (high impact first):
  - Keep data on device for MLP inference (persistent weights/coords) or fuse MLP and physics into a single mega‑kernel.
  - Add shared‑memory tiling (+ halo) in the fused kernel to cut global neighborhood loads.
  - Expand grids and try mixed precision (FP16/BF16) to probe bandwidth limits and throughput.

## 8) How to Reproduce
- Build (Release):
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja`
  - `cmake --build build --config Release -j`
- Correctness:
  - `test_phys_cpu_ref`, `test_phys_cuda_nonfused_vs_cpu`, `test_phys_cuda_fused_vs_nonfused`, `test_mlp_grid_infer`, `test_mlp_phys_integration_inputs`
- Performance:
  - Physics‑only: `test_phys_perf`
  - MLP+Physics: `test_mlp_phys_perf`
- Programs print CSV to stdout; redirect to files as needed.

## 9) Raw CSV (local run)
- Physics‑only:
  - phys,residuals_nonfused,64,64,64,10,3.61493
  - phys,residuals_fused,64,64,64,10,2.20080
  - phys,residuals_nonfused,96,96,64,10,7.06326
  - phys,residuals_fused,96,96,64,10,4.13635
  - phys,residuals_nonfused,128,96,96,10,9.54905
  - phys,residuals_fused,128,96,96,10,6.69626
- MLP+Physics:
  - mlp_phys,nonfused,64,64,64,10,0.07864,3.8968,19.1388,23.0356
  - mlp_phys,fused,64,64,64,10,0.0675008,2.45234,18.859,21.3113
  - mlp_phys,nonfused,96,96,64,10,0.13,7.12756,39.9311,47.0586
  - mlp_phys,fused,96,96,64,10,0.0734944,4.56314,39.6239,44.1871
  - mlp_phys,nonfused,128,96,96,10,0.183395,10.2412,77.3471,87.5883
  - mlp_phys,fused,128,96,96,10,0.119968,7.29306,77.4718,84.7648
