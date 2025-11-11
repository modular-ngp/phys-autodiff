# Plan: Fully-Fused Physics Loss Kernel

## Goals
- Implement a physics loss kernel in two variants:
  - Non-fused: multiple kernels with intermediates in global memory.
  - Fully-fused: a single kernel computing residuals and VJP end-to-end with shared memory tiling.
- Validate numerical equivalence and quantify performance gains across grid sizes and GPUs.
- Provide isolated test executables in `test/` for correctness and performance.

## Problem Setup (from REQUIREMENT.md, distilled)
- Fields on a structured grid with spacing `(hx, hy, hz)` and time step `dt`.
- Scalars: `sigma(t, x, y, z)`; Vectors: `u(t, x, y, z)`.
- Residuals (example forms):
  - R_sigma = ∂t sigma + u · ∇sigma + sigma · (∇·u)
  - R_u     = ∂t u + (u · ∇) u
- Weighted loss: L = w_sigma * mean(R_sigma^2) + w_u * mean(||R_u||^2)
- VJP factors: g_sigma = 2 w_sigma R_sigma, g_u = 2 w_u R_u

## Milestones

M0. Repo Plumbing & Baseline
- Create core headers for physics loss API: grid spec, weights, and buffer structs.
- Set up test executables under `test/` and a perf harness skeleton.
- Deliverable: Compiles; `test_mlp_compare` remains functional; placeholder perf test runs.

M1. Math & Reference Implementations (CPU)
- Decide discretization: centered differences for space, first-order finite difference for time.
- Implement CPU reference for residuals and loss (forward) and its backward (VJP) for sanity.
- Add correctness tests using analytic fields (manufactured solutions) and randomized fields.
- Metrics: relative L2 error and max-abs error thresholds.

M2. Non-Fused CUDA Path
- Kernels: compute spatial/temporal derivatives, residuals, and VJP in separate steps, writing intermediates to global memory.
- Boundary handling: zero-gradient or periodic (configurable); match CPU reference.
- Measure: end-to-end time including global writes; record CSV.

M3. Fully-Fused CUDA Kernel
- Tiling: shared-memory tiles with halo for stencil ops; e.g., 8x8x4 threads per block with ±1 halo.
- In-kernel pipeline: load tile (t-1, t, t+1 as needed), compute derivatives, residuals, accumulate loss/VJP, single write of outputs.
- Warp-level reductions for per-block loss accumulation, then global atomic add.
- Validate bitwise- or tolerance-level equivalence vs non-fused and CPU.

M4. Performance Study
- Bench matrix: grid sizes (e.g., 64^3, 96^3, 128^3, 192^3), FP32; sweeps over boundary modes.
- Timers: CUDA events, warmup + N iterations; report mean/percentiles.
- Outputs: CSV with config, kernel type, runtime ms, bandwidth/throughput estimates, speedup.

M5. Integration & Documentation
- Clean API surface for forward/backward calls; optional hooks for autograd.
- Write usage docs and caveats (SMEM limits, occupancy, arch-specific tuning hints).

## APIs (proposed)
- GridSpec { nx, ny, nz, hx, hy, hz, dt, periodic (bool) }
- PhysWeights { w_sigma, w_u }
- PhysBuf { sigma_t0, sigma_t1, sigma_t2, u_t0, u_t1, u_t2, d_sigma, d_u, loss_sigma, loss_u }
- Functions:
  - cpu_phys_loss_forward(...), cpu_phys_loss_backward(...)
  - cuda_phys_loss_forward_nonfused(...), cuda_phys_loss_backward_nonfused(...)
  - cuda_phys_loss_forward_fused(...), cuda_phys_loss_backward_fused(...)

## Tests (all under test/, each a separate exe)
- test_mlp_compare: existing CPU vs CUDA gradient parity; sanity for build and CUDA env.
- test_phys_cpu_ref: manufactured solutions vs finite-diff expected residuals.
- test_phys_cuda_nonfused_vs_cpu: numerical parity within tolerance.
- test_phys_cuda_fused_vs_nonfused: equivalence within tolerance.
- test_phys_perf: prints CSV lines comparing fused vs non-fused across sizes.

## Acceptance & Metrics
- Correctness: relative L2 error < 1e-6 for residuals and gradients on smooth fields; max-abs < 1e-6.
- Performance: fully-fused shows measurable speedup vs non-fused (goal: >=1.3x, target 2–3x depending on memory subsystem and tile tuning).

## Risks & Mitigations
- SMEM pressure limiting occupancy: tune tile sizes and compute grouping; prefer 2D tiles if needed.
- Bank conflicts: pad shared arrays; align to 32 banks.
- Boundary handling divergence: separate boundary kernel or branchless clamping.
- Atomic contention on loss: per-warp reductions then one atomic per block.

## Next Steps
- Implement M1 CPU reference to lock down math and tests.
- Implement M2 non-fused CUDA and tests; then M3 fused kernel.
- Run M4 perf matrix and iterate on tiling and memory layout.

