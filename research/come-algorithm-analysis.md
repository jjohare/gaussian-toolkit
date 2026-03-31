# CoMe Algorithm Analysis for Rust Implementation

**Paper**: "CoMe: Confidence-Based Mesh Extraction from 3D Gaussians" (arXiv:2603.24725)
**Authors**: Radl, Windisch, Kurz, Kohler, Steiner, Steinberger (Graz University of Technology / Huawei)
**Built on**: SOF (Sorted Opacity Fields, arXiv:2506.19139) and GOF (Gaussian Opacity Fields, arXiv:2404.10772)
**Date of analysis**: 2026-03-31

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Input Format: 3DGS PLY Specification](#2-input-format-3dgs-ply-specification)
3. [Core Data Structures](#3-core-data-structures)
4. [Rendering: Alpha-Blending and Ray Evaluation](#4-rendering-alpha-blending-and-ray-evaluation)
5. [Confidence Mechanism](#5-confidence-mechanism)
6. [Loss Functions](#6-loss-functions)
7. [Confidence-Steered Densification](#7-confidence-steered-densification)
8. [Decoupled Appearance Embedding](#8-decoupled-appearance-embedding)
9. [Opacity Field Construction (from SOF)](#9-opacity-field-construction-from-sof)
10. [Mesh Extraction: Marching Tetrahedra](#10-mesh-extraction-marching-tetrahedra)
11. [Vertex Attribute Computation](#11-vertex-attribute-computation)
12. [Post-Processing](#12-post-processing)
13. [Output Format](#13-output-format)
14. [Hyperparameters Reference](#14-hyperparameters-reference)
15. [Rust Implementation Strategy](#15-rust-implementation-strategy)

---

## 1. Pipeline Overview

CoMe is a **training method**, not a post-hoc extraction tool. It trains 3D Gaussians from
scratch with confidence-aware optimization, then extracts a mesh from the trained opacity field.

```
COLMAP dataset (images/ + sparse/)
    |
    v
[Phase 1] Confidence-Aware 3DGS Training (~18 min on RTX 4090)
    |  - Standard 3DGS optimisation with added:
    |    - Per-primitive learnable confidence
    |    - Confidence loss (L_conf)
    |    - Color variance loss (L_color-var)
    |    - Normal variance loss (L_normal-var)
    |    - Confidence-steered densification
    |    - Decoupled appearance embedding
    |
    v
Trained 3DGS model (PLY with confidence values)
    |
    v
[Phase 2] Mesh Extraction via Marching Tetrahedra (~7 min)
    |  - Construct tetrahedral grid from Gaussian bounding boxes
    |  - Evaluate opacity field at tet vertices
    |  - Binary search along edges for 0.5 level-set
    |  - Standard marching tetrahedra case table
    |
    v
Geometry-only PLY mesh (vertices + faces, vertex colours, normals)
```

**For a Rust implementation focused on mesh extraction only**: Phase 2 is the target.
Phase 1 requires CUDA differentiable rendering and is better left in Python/CUDA.
A Rust tool can accept a trained CoMe/SOF model and perform the mesh extraction step.

---

## 2. Input Format: 3DGS PLY Specification

### Standard 3DGS PLY Header (62 properties per vertex)

```
ply
format binary_little_endian 1.0
element vertex <N>
property float x              # Position
property float y
property float z
property float nx             # Normal (often zero in vanilla 3DGS)
property float ny
property float nz
property float f_dc_0         # SH DC component (RGB base colour)
property float f_dc_1
property float f_dc_2
property float f_rest_0       # SH higher-order coefficients
property float f_rest_1
...
property float f_rest_44      # 45 coefficients = 3 channels x 15 (bands 1-3)
property float opacity        # Logit-space opacity (apply sigmoid to get [0,1])
property float scale_0        # Log-space scale (apply exp to get actual scale)
property float scale_1
property float scale_2
property float rot_0          # Quaternion rotation (w, x, y, z)
property float rot_1
property float rot_2
property float rot_3
end_header
```

**Total**: 3 (pos) + 3 (normal) + 3 (f_dc) + 45 (f_rest) + 1 (opacity) + 3 (scale) + 4 (rot) = 62 floats = 248 bytes per Gaussian.

### CoMe Additional Field (Expected)

CoMe adds a per-primitive confidence scalar:

```
property float confidence     # Logit-space confidence gamma_i (apply exp to get gamma_tilde)
```

This may be stored in a separate file or appended to the PLY. The exact format
depends on the unreleased code, but SOF stores all per-Gaussian attributes in PLY.

### Fields Required for Mesh Extraction

The mesh extraction step (Phase 2) needs only:

| Field | Type | Transform | Purpose |
|-------|------|-----------|---------|
| x, y, z | f32 x 3 | none | Gaussian centre (mean) |
| scale_0/1/2 | f32 x 3 | exp() | Gaussian scale along each axis |
| rot_0/1/2/3 | f32 x 4 | normalise | Quaternion -> rotation matrix |
| opacity | f32 | sigmoid() | Per-Gaussian base opacity o_i |
| f_dc_0/1/2 | f32 x 3 | SH0_C0 transform | Base colour for vertex colouring |
| nx, ny, nz | f32 x 3 | normalise | Gaussian normal (if trained with normal loss) |

Higher-order SH coefficients (f_rest_*) are only needed if view-dependent vertex
colouring is desired. For a geometry-only mesh, f_dc is sufficient.

---

## 3. Core Data Structures

### Gaussian Primitive

```rust
struct Gaussian {
    mean: [f32; 3],           // mu_i = (x, y, z)
    scale: [f32; 3],          // s_i = exp(scale_raw) -- already transformed
    rotation: [f32; 4],       // quaternion (w, x, y, z) -> rotation matrix R_i
    opacity: f32,             // o_i = sigmoid(opacity_raw) in [0, 1]
    colour_dc: [f32; 3],      // f_dc -> base RGB via SH C0 coefficient
    normal: [f32; 3],         // n_i (shortest axis of covariance ellipsoid)
    confidence: f32,          // gamma_tilde_i = exp(gamma_i), optional for mesh extraction
}
```

### Covariance Matrix

The 3x3 covariance matrix for each Gaussian is:

```
Sigma_i = R_i * S_i * S_i^T * R_i^T
```

where:
- `R_i` is the 3x3 rotation matrix from the quaternion
- `S_i = diag(scale_0, scale_1, scale_2)`

The inverse covariance (precision matrix) is:

```
Sigma_i^{-1} = R_i * diag(1/s0^2, 1/s1^2, 1/s2^2) * R_i^T
```

Stored as a symmetric 3x3 matrix (6 unique values).

---

## 4. Rendering: Alpha-Blending and Ray Evaluation

### Per-Ray Alpha Contribution

For a ray r(t) = o + t*d passing near Gaussian i, the point of maximum contribution is:

```
x_i* = o + ((d^T Sigma_i^{-1} (mu_i - o)) / (d^T Sigma_i^{-1} d)) * d
                                                                    [Eq. 3 in CoMe]
```

The alpha contribution at that ray:

```
alpha_i(r) = o_i * exp(-0.5 * (x_i* - mu_i)^T * Sigma_i^{-1} * (x_i* - mu_i))
                                                                    [Eq. 2 in CoMe]
```

### Blending Weights

Standard front-to-back alpha compositing:

```
w_i(r) = alpha_i(r) * prod_{j=0}^{i-1} (1 - alpha_j(r))          [Eq. 1 in CoMe]
```

### Rendered Image

```
I_hat(r) = sum_{i=0}^{N-1} w_i(r) * sh(theta_i, d)               [Eq. 1 in CoMe]
```

where `sh(theta_i, d)` evaluates the spherical harmonics of Gaussian i at view direction d.

---

## 5. Confidence Mechanism

### Per-Primitive Confidence

Each Gaussian has a learnable scalar gamma_i, initialised to 0:

```
gamma_tilde_i = exp(gamma_i)                                       [Eq. 10]
```

At initialisation, gamma_tilde_i = exp(0) = 1 (full confidence).

### Rendered Confidence Map

Confidence is rendered via the same alpha-blending as colour:

```
C_hat(r) = sum_{i=0}^{N-1} w_i(r) * gamma_tilde_i                [Eq. 11]
```

This produces a per-pixel confidence value. Low-confidence regions correspond to
areas where the model is uncertain (reflections, view-dependent effects, thin structures).

### Confidence Clamping

```
C_hat is clamped to [0.001, 5.0]
```

### Confidence Gradient

```
dL_conf / dC_hat = L_rgb - beta / C_hat                           [Eq. 18]
```

When L_rgb < beta/C_hat, the gradient pushes confidence down (model admits uncertainty).
When L_rgb > beta/C_hat, the gradient pushes confidence up.

---

## 6. Loss Functions

### 6.1 Base Photometric Loss

```
L_rgb = (1 - lambda_rgb) * L1(I, I_hat) + lambda_rgb * L_D-SSIM(I, I_hat)
                                                                    [Eq. 4]
lambda_rgb = 0.2
```

### 6.2 Confidence Loss

Replaces L_rgb in the total loss:

```
L_conf = L_rgb * C_hat - beta * log(C_hat)                        [Eq. 9]

beta = 0.075
```

When C_hat = 1, this reduces to L_rgb (no effect). The second term is a
regulariser that prevents confidence from collapsing to zero everywhere (which
would trivially minimise L_rgb * C_hat).

### 6.3 Colour Variance Loss

Penalises blending of Gaussians with different colours at the same pixel:

```
L_color-var = sum_{i=0}^{N-1} w_i(r) * ||sh(theta_i, d) - I||_2^2
                                                                    [Eq. 13]
```

This measures the weighted variance of per-Gaussian colours around the ground-truth
image value. High variance means the rendered colour results from averaging dissimilar
Gaussians (a sign of poor geometry).

**Gradient with respect to opacity** (for the optimiser):

```
dL_color-var / d_alpha_i =
    ||sh(theta_i, d) - I||_2^2 * prod_{j=0}^{i-1}(1 - alpha_j)
    - (1/(1 - alpha_i)) * sum_{j=i+1}^{N-1} w_j * ||sh(theta_j, d) - I||_2^2
                                                                    [Eq. 27]
```

### 6.4 Normal Variance Loss

Penalises blending of Gaussians with different normals:

```
L_normal-var = sum_{i=0}^{N-1} w_i(r) * ||n_i - N||_2^2 = 1 - ||N||_2^2
                                                                    [Eq. 14]
```

where the blended normal is:

```
N = sum_{i=0}^{N-1} w_i(r) * n_i
```

The simplification to `1 - ||N||^2` follows from the identity that the weighted
variance of unit vectors equals 1 minus the squared norm of their weighted mean.

### 6.5 Geometric Losses (from SOF)

**Distortion loss** (encourages compact depth distributions):

```
L_dist = sum_{i,j} w_i * w_j * ||NDC(t_i) - NDC(t_j)||^2         [SOF Eq. 17]
```

**Depth-normal consistency**:

```
L_normal = sum_i w_i * (1 - n_i^T * N)                            [SOF Eq. 18]
```

**Direct opacity supervision** (encourages opacity field to reach 0.5 at surface):

```
L_opa = ||O_N(o + t_r* * d) - 0.5||^2                             [SOF Eq. 20]
```

**Normal smoothness** (edge-aware):

```
L_smooth = ||grad(N)|| * exp(-||grad(I)||)                         [SOF Eq. 22]
```

### 6.6 Total Loss

```
L = L_conf + L_geom + lambda_color-var * L_color-var + lambda_normal-var * L_normal-var
                                                                    [Eq. 15]

lambda_color-var = 0.5
lambda_normal-var = 0.005
```

where L_geom encompasses the SOF geometric losses (distortion, depth-normal
consistency, opacity supervision, smoothness).

---

## 7. Confidence-Steered Densification

Standard 3DGS densifies (clones/splits) Gaussians whose view-space positional
gradient exceeds a threshold tau_grad. CoMe modifies this threshold per-Gaussian:

```
tau_bar_grad = tau_grad / min(gamma_tilde_i, 1)                    [Eq. 12]
```

**Effect**:
- Confident Gaussians (gamma_tilde >= 1): threshold stays at tau_grad (normal densification)
- Low-confidence Gaussians (gamma_tilde < 1): threshold increases (harder to densify)

This prevents over-densification in ambiguous regions (reflections, sky, glass) where
repeated cloning of tiny Gaussians creates floater artifacts.

**Activation**: Confidence densification enabled at iteration 500.
**Learning rate**: gamma_i optimised with lr = 2.5e-4.

---

## 8. Decoupled Appearance Embedding

Handles per-image exposure/lighting variation without baking it into geometry.

### Architecture

```
M_i = F_Theta([ds_32(I_hat), rho_i])                              [Eq. 6]
```

where:
- `ds_32()` downsamples the rendered image by factor 32
- `rho_i` is a per-image learnable latent vector in R^64
- `F_Theta` is a small CNN
- Input is augmented with positional encoding [u, v, r(u,v)] -> total 70 channels

### Appearance-Corrected Image

```
I_hat_app = I_hat * sigmoid(M_i)                                   [Eq. 7]
```

Element-wise multiplication with a sigmoid mask.

### Decoupled D-SSIM

```
L_D-SSIM_dec = 1 - l(I, I_hat_app) * c(I, I_hat) * s(I, I_hat)   [Eq. 8]
```

Only the **luminance** term `l()` uses the appearance-corrected image.
The **contrast** `c()` and **structure** `s()` terms use the raw rendered image.

This prevents the appearance network from compensating for geometric errors.

---

## 9. Opacity Field Construction (from SOF)

This is the core of the mesh extraction step and the primary target for Rust implementation.

### 1D Gaussian Along a Ray

For Gaussian i with centre mu_i, given ray r(t) = o + t*d, define:

```
o_g = Sigma_i^{-1/2} * R_i^T * (o - mu_i)    # ray origin in Gaussian local space
d_g = Sigma_i^{-1/2} * R_i^T * d              # ray direction in Gaussian local space
```

Then:
```
A_i = d_g^T * d_g
B_i = 2 * d_g^T * o_g
C_i = o_g^T * o_g
```

The 1D Gaussian evaluation along the ray:

```
G_i^{1D}(t) = exp(-0.5 * (A_i * t^2 + B_i * t + C_i))            [SOF Eq. 8]
```

The depth of maximum contribution:

```
t_i* = -B_i / (2 * A_i)
```

### Accumulated Opacity Along a Ray

```
O_N(x) = sum_{i=0}^{N-1} o_i * G_i^{1D}(t_eval) * prod_{j=0}^{i-1}(1 - o_j * G_j^{1D}(t_eval))
                                                                    [SOF Eq. 12]
where t_eval = min(t_i*, t)
```

### View-Independent Opacity Field

```
O(x) = min_{all training views (o,d)} O_N(o + t*d)                [SOF Eq. 13]
```

The minimum across all training views makes the field solely position-dependent.

### Exact Depth for 0.5 Level-Set

The depth t_r* where accumulated opacity equals 0.5:

```
T_i * (1 - o_i * G_i^{1D}(t_r*)) = 0.5                           [SOF Eq. 15]
```

Solving the quadratic:

```
t_r* = t_r - sqrt(B_i^2 - 4*A_i*(C_i + 2*ln((T_i - 0.5) / alpha_i))) / (2*A_i)
                                                                    [SOF Eq. 16]
```

---

## 10. Mesh Extraction: Marching Tetrahedra

### Step 1: Gaussian Filtering

Filter out Gaussians that do not contribute to the surface:

```
Keep Gaussian i if: o_i >= 1/255
```

Compute opacity-adaptive bounding extent:

```
E_i = sqrt(2 * ln(255 * o_i))                                     [SOF Eq. 3]
```

### Step 2: Tetrahedral Grid Construction

1. For each surviving Gaussian i, compute a 3D bounding box:
   - Centre: mu_i
   - Half-extents along each axis: E_i * scale_j for j in {0,1,2} (in local frame)
   - 8 corners of the oriented bounding box

2. Collect all centres and corners as vertices

3. Perform **Delaunay triangulation** (3D) on these vertices using CGAL
   - This produces a tetrahedral mesh filling the convex hull
   - Each tetrahedron has 4 vertices and 6 edges

4. **Filter cells**: Remove any tetrahedron whose edges connect non-overlapping Gaussians.
   Two Gaussians are non-overlapping when the edge length exceeds the sum of their
   maximum scales.

### Step 3: Opacity Field Evaluation at Vertices

For each vertex v of the tetrahedral grid:

1. Find all Gaussians whose bounding box contains v
2. Evaluate the accumulated opacity O(v) using the formula from Section 9
3. For SOF: use tile-based parallel evaluation for efficiency
   - Points assigned to tiles (not pixels)
   - Sort by [tile_id, depth] for balanced workload

### Step 4: Level-Set Extraction (Binary Search)

For each edge of each tetrahedron where the two endpoint opacity values straddle
the 0.5 level-set (one vertex has O > 0.5 and the other O < 0.5):

```
Algorithm: Binary Search Level-Set Finding
Input: edge endpoints p0, p1 with O(p0) < 0.5 < O(p1)
Output: surface point p* where O(p*) ~= 0.5

lo = p0
hi = p1
for iter in 0..8:
    mid = (lo + hi) / 2
    o_mid = evaluate_opacity_field(mid)
    if o_mid > 0.5:
        hi = mid
    else:
        lo = mid
return (lo + hi) / 2
```

8 iterations of binary search effectively simulate 256 dense evaluations.
This is more accurate than the linear interpolation used in standard marching
tetrahedra/cubes.

### Step 5: Triangle Generation

Apply the standard marching tetrahedra case table. Each tetrahedron has 4 vertices,
each classified as inside (O >= 0.5) or outside (O < 0.5). This gives 2^4 = 16
cases, reduced to 8 by symmetry. Each case generates 0, 1, or 2 triangles from
the edge intersection points found in Step 4.

The case table (canonical, from Gueziec & Hummel 1995):

| Inside vertices | Triangles |
|-----------------|-----------|
| 0               | 0 triangles |
| 1 (e.g. v0)    | 1 triangle from edges (v0,v1), (v0,v2), (v0,v3) |
| 2 adjacent      | 2 triangles (quad split) |
| 2 opposite      | 2 triangles |
| 3               | 1 triangle (complement of case 1) |
| 4               | 0 triangles |

### Step 6: Mesh Assembly

Collect all generated triangles. Merge duplicate vertices (vertices on shared
tetrahedron edges should be identical). Build vertex and index buffers.

---

## 11. Vertex Attribute Computation

### Vertex Normals

Two approaches, both valid:

**A. From opacity field gradient** (preferred for accuracy):

```
n(x) = -grad(O(x)) / ||grad(O(x))||
```

The gradient can be computed via finite differences at the binary-search vertex
positions, or analytically from the Gaussian field.

**B. From nearest Gaussian** (simpler):

The normal of the nearest Gaussian is the eigenvector corresponding to the
smallest eigenvalue of its covariance matrix (the "thinnest" direction).

```
n_i = R_i * e_min
```

where e_min is the unit vector along the axis with smallest scale.

### Vertex Colours

**From SH DC component** (view-independent base colour):

```
colour_i = SH_C0 * f_dc + 0.5
```

where `SH_C0 = 0.28209479177387814` (the zeroth-order SH basis function value = 1/(2*sqrt(pi))).

For each vertex of the extracted mesh, find the K nearest Gaussians and compute
a distance-weighted average of their base colours:

```
colour(v) = sum_k (w_k * colour_k) / sum_k w_k
where w_k = exp(-||v - mu_k||^2 / (2 * sigma_k^2))
```

Alternatively, assign the colour of the single nearest Gaussian (simpler, noisier).

**From alpha-blended rendering** (higher quality, more expensive):

Render the trained Gaussians from a virtual camera positioned at the vertex,
looking along the vertex normal. Use the rendered colour.

### Vertex Confidence (optional)

If confidence is included in the output, it can be interpolated the same way as
colour from the nearest Gaussians' gamma_tilde values.

---

## 12. Post-Processing

The papers do not describe extensive post-processing. Based on SOF code and
standard practice:

### Component Filtering

Remove small disconnected components below a size threshold (e.g. < 100 faces).

### Laplacian Smoothing (optional)

```
v_i' = v_i + lambda * sum_{j in N(i)} (v_j - v_i) / |N(i)|
```

where N(i) are the 1-ring neighbours of vertex i. Typically 1-3 iterations
with lambda = 0.5.

### Mesh Decimation (optional)

Quadric error metric decimation to reduce face count while preserving shape.
Target: reduce to 50-80% of original face count.

### No UV Mapping or Texture Atlas

CoMe/SOF produce geometry-only meshes with per-vertex colours. For textured
output, a separate pass is needed (xatlas UV unwrapping + reprojection from
training images).

---

## 13. Output Format

### PLY Mesh (default output)

```
ply
format binary_little_endian 1.0
element vertex <V>
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face <F>
property list uchar int vertex_indices
end_header
```

### Alternative: OBJ/GLB

For downstream use, convert PLY to OBJ (with MTL for materials) or GLB
(binary glTF). Both require per-vertex or per-face colours to be preserved.

---

## 14. Hyperparameters Reference

### CoMe-Specific

| Parameter | Symbol | Value | Purpose |
|-----------|--------|-------|---------|
| Confidence balance | beta | 0.075 | L_conf regularisation strength |
| Colour variance weight | lambda_color-var | 0.5 | Weight of colour variance loss |
| Normal variance weight | lambda_normal-var | 0.005 | Weight of normal variance loss |
| Confidence learning rate | lr_gamma | 2.5e-4 | Optimiser LR for gamma_i |
| Confidence activation iter | - | 500 | When to start confidence optimisation |
| Confidence clamp range | - | [0.001, 5.0] | Prevents numerical issues |
| RGB loss weight | lambda_rgb | 0.2 | D-SSIM vs L1 balance |
| Appearance latent dim | - | 64 | Per-image latent vector dimension |
| Appearance downsample | - | 32x | CNN input resolution reduction |

### Mesh Extraction (SOF)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Level-set threshold | 0.5 | Opacity isosurface value |
| Binary search iterations | 8 | Precision of vertex placement |
| Minimum opacity | 1/255 | Gaussian inclusion threshold |
| Bounding extent | sqrt(2*ln(255*o_i)) | Adaptive bounding box size |

---

## 15. Rust Implementation Strategy

### Scope: Mesh Extraction Only (Phase 2)

The training phase (Phase 1) requires CUDA differentiable rasterisation and
backpropagation through millions of Gaussians. This is impractical in pure Rust
and should remain in Python/CUDA.

The mesh extraction phase (Phase 2) is computationally intensive but
embarrassingly parallel and does not require gradient computation. It is an
excellent target for Rust.

### Required Crates

| Crate | Purpose |
|-------|---------|
| `ply-rs` or custom parser | Read/write PLY files (Gaussian input, mesh output) |
| `nalgebra` | Linear algebra (3x3 matrices, quaternions, eigendecomposition) |
| `rayon` | Parallel iteration over Gaussians, tetrahedra, edges |
| `kiddo` or `rstar` | KD-tree/R-tree for spatial queries (nearest Gaussian lookup) |
| `spade` | 3D Delaunay triangulation (alternative to CGAL) |
| `meshopt` | Mesh optimisation (vertex cache, overdraw) |

### Core Algorithm in Rust (Pseudocode)

```rust
fn extract_mesh(gaussians: &[Gaussian], cameras: &[Camera]) -> Mesh {
    // 1. Filter low-opacity Gaussians
    let active: Vec<&Gaussian> = gaussians.iter()
        .filter(|g| g.opacity >= 1.0 / 255.0)
        .collect();

    // 2. Compute bounding boxes and collect vertices for Delaunay
    let mut tet_points: Vec<Point3> = Vec::new();
    for g in &active {
        let extent = (2.0 * (255.0 * g.opacity).ln()).sqrt();
        tet_points.push(g.mean);  // centre
        // Add 8 OBB corners
        for corner in g.oriented_bounding_box(extent) {
            tet_points.push(corner);
        }
    }

    // 3. Delaunay triangulation -> tetrahedral mesh
    let tets = delaunay_3d(&tet_points);

    // 4. Filter tets connecting non-overlapping Gaussians
    let tets = filter_non_overlapping(tets, &active);

    // 5. Evaluate opacity at each tet vertex
    let opacity_values: Vec<f32> = tet_points.par_iter()
        .map(|p| evaluate_opacity_field(p, &active, cameras))
        .collect();

    // 6. For each tet edge straddling 0.5, binary search for surface point
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();

    for tet in &tets {
        let case = classify_tet(tet, &opacity_values);
        let edge_vertices = binary_search_edges(tet, &opacity_values,
                                                 &tet_points, &active, cameras);
        let triangles = marching_tet_case_table(case, &edge_vertices);
        // ... accumulate vertices and indices
    }

    // 7. Assign vertex colours and normals
    assign_vertex_attributes(&mut vertices, &active);

    // 8. Post-process
    remove_small_components(&mut vertices, &mut indices, min_faces: 100);

    Mesh { vertices, indices }
}

fn evaluate_opacity_field(
    point: &Point3,
    gaussians: &[&Gaussian],
    cameras: &[Camera],
) -> f32 {
    let mut min_opacity = f32::MAX;

    for cam in cameras {
        let ray_origin = cam.position;
        let ray_dir = (point - ray_origin).normalize();
        let t_target = (point - ray_origin).norm();

        // Sort Gaussians by depth along this ray
        let mut contributions: Vec<(f32, f32)> = Vec::new(); // (depth, alpha)
        for g in gaussians {
            let (t_star, alpha) = evaluate_gaussian_on_ray(g, &ray_origin, &ray_dir);
            if alpha > 1e-6 {
                contributions.push((t_star.min(t_target), alpha));
            }
        }
        contributions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Accumulate opacity
        let mut transmittance = 1.0_f32;
        let mut accumulated = 0.0_f32;
        for (_, alpha) in &contributions {
            accumulated += transmittance * alpha;
            transmittance *= 1.0 - alpha;
            if transmittance < 1e-4 { break; } // early stop
        }

        min_opacity = min_opacity.min(accumulated);
    }

    min_opacity
}
```

### Performance Considerations

1. **Spatial indexing**: Use a BVH or KD-tree to find relevant Gaussians for each
   query point, avoiding O(N) scans over all Gaussians.

2. **Tile-based evaluation**: SOF's parallelisation groups vertices into tiles and
   sorts Gaussians by [tile_id, depth]. This maps well to Rayon's parallel iterators.

3. **Memory**: For 1M Gaussians at 248 bytes each, the input is ~248 MB. The
   tetrahedral grid may have 5-10x as many vertices. Budget 2-4 GB for the
   extraction pipeline.

4. **Binary search**: 8 iterations per edge, with each iteration requiring an
   opacity field evaluation. This is the hot loop and benefits most from SIMD
   and cache-friendly data layout.

5. **Delaunay triangulation**: The 3D Delaunay step is the most complex algorithmic
   component. The `spade` crate handles 2D; for 3D, consider binding to CGAL via
   `cxx` or using a simpler uniform grid approach for bounded scenes.

### Simplified Alternative: TSDF Fusion in Rust

If 3D Delaunay is too complex to implement:

1. Render depth maps from the trained Gaussians at 32-64 viewpoints
2. Fuse into a voxel TSDF grid
3. Extract mesh via marching cubes (well-supported in Rust ecosystem)

This trades mesh quality for implementation simplicity. The `isosurface` crate
provides marching cubes in Rust.

---

## Sources

- [CoMe Paper (arXiv:2603.24725)](https://arxiv.org/abs/2603.24725)
- [CoMe HTML (full text)](https://arxiv.org/html/2603.24725)
- [CoMe Project Page](https://r4dl.github.io/CoMe/)
- [CoMe GitHub (code coming soon)](https://github.com/r4dl/CoMe)
- [SOF Paper (arXiv:2506.19139)](https://arxiv.org/abs/2506.19139)
- [SOF GitHub (code available)](https://github.com/r4dl/SOF)
- [GOF Paper (arXiv:2404.10772)](https://arxiv.org/abs/2404.10772)
- [GOF GitHub](https://github.com/autonomousvision/gaussian-opacity-fields)
- [3DGS PLY Format Reference](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/)
- [Original 3DGS (graphdeco-inria)](https://github.com/graphdeco-inria/gaussian-splatting)
