# Plan: MLP‑Driven 3D Smoke Physics Loss Benchmark

## Goals
- Use the repo’s MLP to synthesize 3D density (sigma) and velocity (u) fields on a grid at times t−dt, t, t+dt.
- Feed these fields into CUDA physics loss kernels (Non‑Fused vs Fully‑Fused) and report performance deltas.
- Keep tests isolated under `test/` as standalone executables; produce reproducible CSV.

## Scope and Assumptions
- MLP inputs: normalized coordinates (x, y, z, t) ∈ [0, 1] or [−1, 1] (configurable); batch = grid points.
- MLP outputs: 4 channels per sample: [sigma, ux, uy, uz]. We will drive both CPU and CUDA MLP backends, favoring CUDA for E2E GPU pipelines.
- Physics kernels already implemented: `cuda_phys_residuals_nonfused` and `cuda_phys_residuals_fused` (+ backward variants).
- This plan focuses on inference + physics benchmarks (not training). A later milestone covers optional physics‑only training.

## Milestones

M0. Design & Plumbing
- Decide coordinate normalization and grid indexing convention; add a small config struct for MLP inference.
- Define a unified MLP driver API: `mlp_infer_{cpu,cuda}(coords -> [sigma,u])` producing contiguous arrays for a grid.
- Add a validation helper to check shapes and bounds; add simple unit sanity tests under `test/`.

M1. MLP Grid Inference
- CPU baseline: loop over grid points and call `mlp_forward<ExecCpu>` in batches.
- CUDA path: batch all grid coordinates and call `mlp_forward<ExecCuda>`; or implement a device kernel wrapper that evaluates the 2‑layer MLP directly for better throughput.
- Output layout: scalars `sigma_t{m1,0,p1}` as 3 × N; vectors `u_t{m1,0,p1}` as channel‑major 3 × N per time slice (total 12 × N floats).
- Tests: `test_mlp_grid_infer` prints a checksum and verifies consistent ranges and continuity across t−dt, t, t+dt.

M2. Field Generation Harness
- Implement a small generator that builds grid coordinates for requested (nx, ny, nz) and times (t−dt, t, t+dt), normalizes them, and invokes the MLP inference.
- Add a cache to avoid re‑allocating buffers across runs; expose CLI args for sizes, dt, and seeds.
- Tests: `test_mlp_phys_integration_inputs` checks that arrays have expected sizes and non‑NaN values.

M3. Physics Loss: Non‑Fused Pipeline (baseline)
- Wire `cuda_phys_residuals_nonfused` (and backward if needed) on the generated fields.
- Timer: CUDA events for kernel‑only and wall‑clock for E2E (including H2D/D2H + alloc/free).
- CSV: emit lines `mlp_phys,nonfused,nx,ny,nz,iters,ms_kernel,ms_e2e`.
- Tests: `test_mlp_phys_nonfused_perf` runs a small matrix (e.g., 64^3, 96^3, 128^3).

M4. Physics Loss: Fully‑Fused Pipeline
- Wire `cuda_phys_residuals_fused` on the same inputs.
- Timer: same methodology as M3; CSV with `mlp_phys,fused,...`.
- Tests: `test_mlp_phys_fused_vs_nonfused` checks numerical equivalence (rel L2) and prints paired CSV.

M5. Report & Reproducibility
- Aggregate CSV into a single artifacts directory; include the full command lines used.
- Provide a short README with setup instructions and sample results; plot optional.
- Deliver speedup table: fused vs non‑fused, both kernel‑only and end‑to‑end.

M6. Optional Enhancements (post‑baseline)
- Streaming/Tiled Inference: generate fields tile‑by‑tile to reduce peak memory (e.g., 128^3 → 8 tiles).
- Divergence‑Free Velocity: optional projection or param via vector potential for more realistic smoke velocity fields.
- End‑to‑End Fusing (stretch): fuse MLP inference and physics residuals into a single mega‑kernel to minimize global memory traffic.
- Physics‑Only Training: backprop g_sigma/g_u through the MLP (requires analytic Jacobians or autograd integration), beyond current scope.

## Data & Memory Planning
- For grid N = nx·ny·nz, the physics loss needs 12·N floats (sigma×3 + u×3×3). At 128^3 (≈2.1M points): ~12×2.1e6×4B ≈ 96 MB for inputs; intermediate buffers depend on non‑fused path but are transient.
- Use channel‑major layout for vectors to match existing kernels.

## Metrics & Outputs
- Primary: average milliseconds (ms/iter) and speedup (fused vs non‑fused).
- Modes: kernel‑only vs end‑to‑end (includes H2D/D2H + allocations) to reflect real usage.
- CSV schema: `mlp_phys,{nonfused|fused},nx,ny,nz,iters,ms_kernel,ms_e2e`.

## Risks & Mitigations
- Peak memory with large grids: use tiled inference and on‑the‑fly physics evaluation.
- PCIe overhead in E2E timing: prefer CUDA MLP inference to keep data on device between steps.
- Numerical drift: verify tolerance with manufactured solutions and regression compare fused/non‑fused.

## Deliverables
- New test executables under `test/`: `test_mlp_grid_infer`, `test_mlp_phys_nonfused_perf`, `test_mlp_phys_fused_vs_nonfused`.
- CSV reports and a short README in `bench/` (or `artifacts/`) with speedup summary.

