# Quality Optimisation Guide

## Key Finding

Our initial test run at 7k iterations was **massively undertrained**. The LichtFeld default is 30,000 iterations, and 7k is just an intermediate checkpoint. The loss of 0.2757 indicates training was stopped before convergence.

## Recommended Training Command

### Balanced Quality (30k iterations, ~10 min on A6000)

```bash
lichtfeld-studio --headless \
    --data-path /path/to/colmap/undistorted \
    --output-path /path/to/output \
    --iter 30000 \
    --strategy mrnf \
    --pose-opt mlp \
    --ppisp \
    --mip-filter \
    --eval \
    --sh-degree 3
```

### Maximum Quality (60k iterations, ~20 min on A6000)

```bash
lichtfeld-studio --headless \
    --data-path /path/to/colmap/undistorted \
    --output-path /path/to/output \
    --iter 60000 \
    --strategy mrnf \
    --pose-opt mlp \
    --ppisp \
    --mip-filter \
    --bilateral-grid \
    --eval \
    --sh-degree 3 \
    --steps-scaler 2.0
```

## Parameter Recommendations

| Parameter | Our v1 | Recommended | Why |
|-----------|--------|-------------|-----|
| Iterations | 7,000 | 30,000-60,000 | 7k is checkpoint, 30k is convergence |
| Strategy | MCMC | **MRNF** | Error-guided densification, 5M max cap |
| Pose optimisation | Off | **--pose-opt mlp** | Video COLMAP poses are noisy |
| PPISP | Off | **--ppisp** | Auto-exposure correction |
| Mip filter | Off | **--mip-filter** | Anti-aliasing for multi-scale |
| Eval mode | Off | **--eval** | PSNR metrics on held-out views |
| SH degree | 3 | 3 | Full view-dependent colour |
| Max gaussians | 1M (MCMC) | 5M (MRNF default) | More capacity for detail |
| Bilateral grid | Off | Optional | For mixed lighting scenes |

## Strategy Selection Guide

| Strategy | Best For | Max Cap | Features |
|----------|----------|---------|----------|
| **MRNF** | Indoor, gallery, detailed objects | 5M | Error + edge maps, refined LR |
| MCMC | Outdoor, large coverage | 1M | Stochastic, noise injection |
| IGS+ | Appearance variation, bilateral grid | 4M | TV loss, higher scaling LR |

## Quality Gates (Automated)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Loss at 5k iter | > 0.1 | Enable pose optimisation |
| Eval PSNR | < 22 dB | FAIL — retrain |
| Eval PSNR | 22-25 dB | Continue +10k iterations |
| Eval PSNR | > 25 dB | PASS |
| Loss plateau | Before 15k | Try pose opt or increase LR |
| Loss oscillation | Any | Reduce learning rate |

## Frame Extraction

| Quality | Frames | FPS | COLMAP Time |
|---------|--------|-----|-------------|
| Minimum | 50-100 | 0.5 | 5 min |
| Balanced | 120-180 | 1.0 | 20 min |
| High | 300-500 | 2.0 | 1-3 hours |

## Sources

- LichtFeld defaults: `src/core/include/core/parameters.hpp` (line 71: `iterations = 30'000`)
- Strategy presets: `src/core/parameters.cpp` (lines 210-258)
- PPISP docs: `docs/docs/features/ppisp.md`
- Pose opt docs: `docs/docs/features/poseopt.md`
