"""
3D Gaussian Splatting - Forward Rendering Pipeline

Mathematical Foundation:
Each 3D Gaussian G_i is defined by parameters (μ_i, Σ_i, α_i, c_i):
- μ_i ∈ ℝ³: 3D position (mean of Gaussian distribution)
- Σ_i ∈ ℝ³ˣ³: 3D covariance matrix (defines ellipsoid shape)
- α_i ∈ [0,1]: opacity (controls transparency)
- c_i: view-dependent color from spherical harmonics

The 3D Gaussian function: G_i(x) = exp(-½(x-μ_i)ᵀΣ_i⁻¹(x-μ_i))

Rendering Pipeline:
1. PARAMETERIZATION: Represent each Gaussian with learnable parameters
   - Position μ directly optimizable
   - Covariance Σ = RSS^TR^T (rotation R + scale S for stability) → compute_cov3d()
   - Colors from spherical harmonics c(d) = Σ_l c_l Y_l(d)
   
2. CULLING & PROJECTION: Transform to screen space → wp_preprocess()
   - Frustum culling: remove Gaussians outside view
   - Project μ_i to 2D: p_i = π(Tμ_i) where T is view transform
   - EWA splatting: project 3D covariance Σ_i to 2D covariance Σ'_i → compute_cov2d()
   
3. TILE-BASED RASTERIZATION: Efficient GPU rendering
   - Divide screen into 16×16 tiles for parallel processing
   - Sort Gaussians by depth per tile (painter's algorithm) → wp_duplicate_with_keys(), wp_identify_tile_ranges()
   - Alpha compositing: C = Σ_i α_i c_i ∏_{j<i}(1-α_j) → wp_render_gaussians()
   
4. DIFFERENTIABLE RENDERING: Enable gradient-based optimization
   - All operations differentiable w.r.t. Gaussian parameters
   - Gradients flow from pixel loss back to μ, Σ, α, c parameters

Key Innovation: Replace slow volumetric rendering with fast rasterization
while maintaining differentiability for end-to-end neural optimization.
"""

import warp as wp
from utils.wp_utils import to_warp_array, wp_prefix_sum, wp_copy_int64, wp_copy_int
from config import *
# Initialize Warp
wp.init()

# Define spherical harmonics constants
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


import warp as wp

# Define the CUDA code snippets for bit reinterpretation
float_to_uint32_snippet = """
    return reinterpret_cast<uint32_t&>(x);
"""

@wp.func_native(float_to_uint32_snippet)
def float_bits_to_uint32(x: float) -> wp.uint32:
    ...

@wp.func
def ndc2pix(x: float, size: float) -> float:
    return ((x + 1.0) * size - 1.0) * 0.5

@wp.func
def get_rect(p: wp.vec2, max_radius: float, tile_grid: wp.vec3):
    # Extract grid dimensions
    grid_size_x = tile_grid[0]
    grid_size_y = tile_grid[1]
    
    rect_min_x = wp.min(wp.int32(grid_size_x), wp.int32(wp.max(wp.int32(0), wp.int32((p[0] - max_radius) / float(TILE_M)))))
    rect_min_y = wp.min(wp.int32(grid_size_y), wp.int32(wp.max(wp.int32(0), wp.int32((p[1] - max_radius) / float(TILE_N)))))
    

    rect_max_x = wp.min(wp.int32(grid_size_x), wp.int32(wp.max(wp.int32(0), wp.int32((p[0] + max_radius + float(TILE_M) - 1.0) / float(TILE_M)))))
    rect_max_y = wp.min(wp.int32(grid_size_y), wp.int32(wp.max(wp.int32(0), wp.int32((p[1] + max_radius + float(TILE_N) - 1.0) / float(TILE_N)))))
    
    return rect_min_x, rect_min_y, rect_max_x, rect_max_y


@wp.func
def compute_cov2d(p_orig: wp.vec3, cov3d: VEC6, view_matrix: wp.mat44, 
                 tan_fovx: float, tan_fovy: float, width: float, height: float) -> wp.vec3:
    """
    PIPELINE STEP 2: EWA SPLATTING
    Project 3D Gaussian to 2D screen space using EWA (Elliptical Weighted Average) splatting.
    
    Key insight: When we view a 3D Gaussian from a camera, it appears as a 2D Gaussian
    on the image plane. We need to compute this 2D covariance for rendering.
    
    The projection formula: Σ_2D = J * W * Σ_3D * W^T * J^T
    
    Breaking it down:
    1. W transforms from world to camera space (viewing transformation)
    2. Σ_3D is our 3D Gaussian's covariance in world space
    3. J is the Jacobian of the perspective projection (handles depth scaling)
    
    The Jacobian J captures how perspective projection distorts the Gaussian:
    - Objects farther away appear smaller (1/z scaling)
    - Off-center objects get stretched (perspective distortion)
    
    The result is a 2x2 covariance matrix that tells us:
    - The size of the Gaussian splat on screen
    - Its orientation (elongation direction)
    - How to properly blend it with other Gaussians
    """
    
    t = wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0) * view_matrix
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    # Clamp X/Y to stay inside frustum
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]
    
    focal_x = width / (2.0 * tan_fovx)
    focal_y = height / (2.0 * tan_fovy)
    # compute Jacobian
    J = wp.mat33(
        focal_x / t[2], 0.0, -(focal_x * t[0]) / (t[2] * t[2]),
        0.0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2]),
        0.0, 0.0, 0.0
    )
    
    W = wp.mat33(
        view_matrix[0, 0], view_matrix[0, 1], view_matrix[0, 2],
        view_matrix[1, 0], view_matrix[1, 1], view_matrix[1, 2],
        view_matrix[2, 0], view_matrix[2, 1], view_matrix[2, 2]
    )
    
    T = J * W
    
    Vrk = wp.mat33(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5]
    )
    
    # CORE 3DGS MATH: Project 3D covariance to 2D using transformation matrix T
    # Formula: Σ_2D = T * Σ_3D * T^T  
    # Your implementation correctly transposes Vrk first, then applies the transforms
    cov = T * wp.transpose(Vrk) * wp.transpose(T)
    
    # Return 2D covariance as (σ_xx, σ_xy, σ_yy) - enough to define 2D ellipse
    return wp.vec3(cov[0, 0], cov[0, 1], cov[1, 1])

@wp.func
def compute_cov3d(scale: wp.vec3, scale_mod: float, rot: wp.vec4) -> VEC6:
    """
    PIPELINE STEP 1: PARAMETERIZATION
    Build 3D covariance matrix that defines the shape of our Gaussian ellipsoid.
    
    A 3D Gaussian has the form: G(x) = exp(-0.5 * (x-μ)^T * Σ^(-1) * (x-μ))
    where Σ is the covariance matrix that controls the Gaussian's shape.
    
    Instead of directly optimizing Σ (which could become invalid), we decompose it as:
    Σ = R * S * S^T * R^T
    
    Where:
    - S = diagonal matrix with our scale parameters (ellipsoid radii)  
    - R = rotation matrix from quaternion (ellipsoid orientation)
    
    Why this works:
    1. Guarantees Σ is positive semi-definite (valid probability distribution)
    2. Scales can be constrained > 0 during optimization (no invalid states)
    3. Quaternions naturally parameterize rotations (no gimbal lock)
    4. Only need 7 parameters (3 scales + 4 quaternion) vs 6 for raw covariance
    
    The final covariance determines how the Gaussian "spreads" in 3D space.
    """
    # Create diagonal scaling matrix with modifier applied for adaptive densification
    S = wp.mat33(
        scale_mod * scale[0], 0.0, 0.0,
        0.0, scale_mod * scale[1], 0.0,
        0.0, 0.0, scale_mod * scale[2]
    )
    # Convert quaternion to rotation matrix
    R = wp.quat_to_matrix(wp.quaternion(rot[0], rot[1], rot[2], rot[3]))
    # Combined transformation matrix M = R * S
    M = R * S
    
    # Compute 3D covariance matrix: Σ = M * M^T
    # This gives us the full 3x3 symmetric covariance matrix
    sigma = M * wp.transpose(M)
    
    # Return upper triangular part (6 unique values) since matrix is symmetric
    return VEC6(sigma[0, 0], sigma[0, 1], sigma[0, 2], sigma[1, 1], sigma[1, 2], sigma[2, 2])

# --- Forward Rendering Kernels ---
@wp.kernel
def wp_preprocess(
    # --- Inputs ---
    orig_points: wp.array(dtype=wp.vec3),        # 3D positions in world space (N, 3)
    scales: wp.array(dtype=wp.vec3),             # Ellipsoid scales (N, 3)
    scale_modifier: float,                       # Global scale multiplier for densification
    rotations: wp.array(dtype=wp.vec4),          # Ellipsoid orientations as quaternions (N, 4)
    opacities: wp.array(dtype=float),            # Transparency values (N,)
    shs: wp.array(dtype=wp.vec3),                # Spherical harmonic coefficients (N*16, 3)
    degree: int,                                 # SH degree (0=diffuse, higher=view-dependent)
    clamped: bool,                               # Whether to clamp SH evaluation
    view_matrix: wp.mat44,                       # World to camera transformation
    proj_matrix: wp.mat44,                       # Camera to screen projection
    cam_pos: wp.vec3,                            # Camera position in world space
    W: int,                                      # Screen width in pixels
    H: int,                                      # Screen height in pixels
    tan_fovx: float,                             # tan(fov_x/2) for perspective projection
    tan_fovy: float,                             # tan(fov_y/2) for perspective projection
    focal_x: float,                              # Focal length X (pixels)
    focal_y: float,                              # Focal length Y (pixels)
    
    prefiltered: bool,                           # Whether to prefilter small Gaussians
    antialiasing: bool,                          # Whether to apply antialiasing
    
    # --- Outputs ---
    radii: wp.array(dtype=int),                  # 2D radius of each Gaussian on screen (N,)
    points_xy_image: wp.array(dtype=wp.vec2),    # 2D projected positions (N, 2)
    depths: wp.array(dtype=float),               # Depth values for sorting (N,)
    cov3Ds: wp.array(dtype=VEC6),                # 3D covariance matrices (N, 6)
    rgb: wp.array(dtype=wp.vec3),                # View-dependent colors from SH (N, 3)
    conic_opacity: wp.array(dtype=wp.vec4),      # 2D covariance inverse + opacity (N, 4)
    tile_grid: wp.vec3,                          # Tile grid dimensions for rasterization
    tiles_touched: wp.array(dtype=int),          # Number of tiles each Gaussian touches (N,)
    clamped_state: wp.array(dtype=wp.vec3)       # SH clamping state for gradients (N, 3)
):
    """
    PIPELINE STEP 2: CULLING & PROJECTION
    Transform 3D Gaussians to screen space and prepare for rasterization.
    
    This kernel performs the critical "splatting" operation:
    1. Frustum culling: remove Gaussians outside camera view
    2. Perspective projection: μ_i → p_i = π(T·μ_i) 
    3. EWA splatting: project 3D covariance Σ_3D → Σ_2D
    4. Spherical harmonics evaluation: compute view-dependent colors
    5. Tile determination: which screen tiles does each Gaussian affect?
    
    Mathematical core: For each Gaussian i:
    - Build 3D covariance: Σ₃D = RSS^TR^T
    - Project to 2D: Σ₂D = J·W·Σ₃D·W^T·J^T (EWA transformation)
    - Evaluate SH color: c = Σₗ cₗYₗ(view_direction)
    - Compute 2D extent: radius ≈ 3σ (covers 99.7% of Gaussian mass)
    
    Performance critical: Determines what gets rendered and how.
    """
    # Get thread indices
    i = wp.tid()
    
    # For each Gaussian
    p_orig = orig_points[i]
    p_view = wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0) * view_matrix

    if p_view[2] < 0.2:
        return

    p_hom = wp.vec4(p_orig[0], p_orig[1], p_orig[2], 1.0) * proj_matrix
    
    p_w = 1.0 / (p_hom[3] + 0.0000001)
    p_proj = wp.vec3(p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w)

    cov3d = compute_cov3d(scales[i], scale_modifier, rotations[i])

    cov3Ds[i] = cov3d
    # Compute 2D covariance matrix
    cov2d = compute_cov2d(p_orig, cov3d, view_matrix, tan_fovx, tan_fovy, float(W), float(H))

    # Constants
    h_var = 0.3
    W_float = float(W)
    H_float = float(H)
    C = 3  # RGB channels
    
    # Add low-pass filter for antialiasing (prevents aliasing of small Gaussians)
    det_cov = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1]
    cov_with_blur = wp.vec3(cov2d[0] + h_var, cov2d[1], cov2d[2] + h_var)
    det_cov_plus_h_cov = cov_with_blur[0] * cov_with_blur[2] - cov_with_blur[1] * cov_with_blur[1]
    
    # Invert 2D covariance matrix to get conic form for efficient evaluation
    # Quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ) where Σ⁻¹ = [[c,-b],[-b,a]]/det
    det = det_cov_plus_h_cov
    if det == 0.0:  # Degenerate case: Gaussian has zero area
        return
        
    det_inv = 1.0 / det
    conic = wp.vec3(
        cov_with_blur[2] * det_inv, 
        -cov_with_blur[1] * det_inv, 
        cov_with_blur[0] * det_inv
    )
    # Compute eigenvalues of 2D covariance matrix to determine screen-space extent
    # For 2x2 matrix [[a,b],[b,c]], eigenvalues are: λ = (a+c)/2 ± √((a+c)²/4 - ac + b²)
    mid = 0.5 * (cov_with_blur[0] + cov_with_blur[2])
    lambda1 = mid + wp.sqrt(wp.max(0.1, mid * mid - det))  # Larger eigenvalue
    lambda2 = mid - wp.sqrt(wp.max(0.1, mid * mid - det))  # Smaller eigenvalue
    # 3-sigma rule: 99.7% of Gaussian mass lies within 3σ from center
    my_radius = wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2)))  # Screen radius in pixels
    # Convert to pixel coordinates
    point_image = wp.vec2(ndc2pix(p_proj[0], W_float), ndc2pix(p_proj[1], H_float))
    
    # Get rectangle of affected tiles
    rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_rect(point_image, my_radius, tile_grid)
    
    # Skip if rectangle has 0 area
    if (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y) == 0:
        return
    # Compute color from spherical harmonics
    pos = p_orig
    dir_orig = pos - cam_pos
    dir = wp.normalize(dir_orig)
    x, y, z = dir[0], dir[1], dir[2]
    
    # Base offset for this Gaussian's SH coefficients
    base_idx = i * 16  # assuming degree 3 (16 coefficients)
    
    # Start with the DC component (degree 0)
    result = SH_C0 * shs[base_idx]
    
    # Add higher degree terms if requested
    if degree > 0:
        # Degree 1 terms
        result = result - SH_C1 * y * shs[base_idx + 1] + SH_C1 * z * shs[base_idx + 2] - SH_C1 * x * shs[base_idx + 3]
        
        if degree > 1:
            # Degree 2 terms
            xx = x*x
            yy = y*y
            zz = z*z
            xy = x*y
            yz = y*z
            xz = x*z
            
            # Degree 2 terms with hardcoded constants
            result = result + 1.0925484305920792 * xy * shs[base_idx + 4] 
            result = result + (-1.0925484305920792) * yz * shs[base_idx + 5]
            result = result + 0.31539156525252005 * (2.0 * zz - xx - yy) * shs[base_idx + 6]
            result = result + (-1.0925484305920792) * xz * shs[base_idx + 7]
            result = result + 0.5462742152960396 * (xx - yy) * shs[base_idx + 8]
                   
            if degree > 2:
                # Degree 3 terms with hardcoded constants
                result = result + (-0.5900435899266435) * y * (3.0 * xx - yy) * shs[base_idx + 9]
                result = result + 2.890611442640554 * xy * z * shs[base_idx + 10]
                result = result + (-0.4570457994644658) * y * (4.0 * zz - xx - yy) * shs[base_idx + 11]
                result = result + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shs[base_idx + 12]
                result = result + (-0.4570457994644658) * x * (4.0 * zz - xx - yy) * shs[base_idx + 13]
                result = result + 1.445305721320277 * z * (xx - yy) * shs[base_idx + 14]
                result = result + (-0.5900435899266435) * x * (xx - 3.0 * yy) * shs[base_idx + 15]
    
    result = result + wp.vec3(0.5, 0.5, 0.5)  # Add 0.5 offset for color range [0,1]
    
    # Track which color channels are clamped (using wp.vec3 instead of separate uint32 values)
    # Store 1.0 if clamped, 0.0 if not clamped
    # Use separate assignments instead of conditional expressions
    r_clamped = 0.0
    g_clamped = 0.0
    b_clamped = 0.0
    
    if result[0] < 0.0:
        r_clamped = 1.0
    if result[1] < 0.0:
        g_clamped = 1.0
    if result[2] < 0.0:
        b_clamped = 1.0
        
    clamped_state[i] = wp.vec3(r_clamped, g_clamped, b_clamped)
    
    if clamped:
        # RGB colors are clamped to positive values
        result = wp.vec3(
            wp.max(result[0], 0.0),
            wp.max(result[1], 0.0),
            wp.max(result[2], 0.0)
        )

    rgb[i] = result
    
    # Store computed data
    depths[i] = p_view[2]
    radii[i] = int(my_radius)
    points_xy_image[i] = point_image
    
    # Pack conic and opacity into single vec4
    conic_opacity[i] = wp.vec4(conic[0], conic[1], conic[2], opacities[i])
    # Store tile information
    tiles_touched[i] = (rect_max_y - rect_min_y) * (rect_max_x - rect_min_x)

@wp.kernel
def wp_render_gaussians(
    # --- Outputs ---
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    
    # --- Inputs ---
    ranges: wp.array(dtype=wp.vec2i),
    point_list: wp.array(dtype=int),
    W: int,
    H: int,
    points_xy_image: wp.array(dtype=wp.vec2),
    colors: wp.array(dtype=wp.vec3),
    conic_opacity: wp.array(dtype=wp.vec4),
    depths: wp.array(dtype=float),
    background: wp.vec3,
    tile_grid: wp.vec3,
    final_Ts: wp.array2d(dtype=float),
    n_contrib: wp.array2d(dtype=int),
):
    tile_x, tile_y, tid_x, tid_y = wp.tid()
    
    # Calculate tile index
    if tile_y >= (H + TILE_N - 1) // TILE_N:
        return
    
    # Calculate pixel boundaries for this tile
    pix_min_x = tile_x * TILE_M
    pix_min_y = tile_y * TILE_N
    pix_max_x = wp.min(pix_min_x + TILE_M, W)
    pix_max_y = wp.min(pix_min_y + TILE_N, H)
    
    # Calculate pixel position for this thread
    pix_x = pix_min_x + tid_x
    pix_y = pix_min_y + tid_y
    
    # Check if this thread processes a valid pixel
    inside = (pix_x < W) and (pix_y < H)
    if not inside:
        return
    
    pixf_x = float(pix_x)
    pixf_y = float(pix_y)
    
    # Get start/end range of IDs to process for this tile
    tile_id = tile_y * int(tile_grid[0]) + tile_x
    range_start = ranges[tile_id][0]
    range_end = ranges[tile_id][1]
    
    # ALPHA COMPOSITING: Core rendering equation for transparency
    # Mathematical formulation: C = Σᵢ αᵢcᵢ ∏ⱼ<ᵢ(1-αⱼ) + T_final * background
    # Where:
    # - αᵢ = opacity of Gaussian i at this pixel
    # - cᵢ = color of Gaussian i (from spherical harmonics)
    # - T = transmittance = ∏ⱼ≤ᵢ(1-αⱼ) (fraction of light transmitted through layers 0...i)
    #
    # Physical interpretation: Light travels from background through transparent
    # layers, getting attenuated and colored by each Gaussian it encounters
    #
    # Implementation note: We render front-to-back (far to near) with early 
    # termination when T becomes negligible (pixel effectively opaque)
    
    T = float(1.0)  # (starts fully transparent)
    r, g, b = float(0.0), float(0.0), float(0.0)
    expected_inv_depth = float(0.0)
    
    contributor_count = int(0)
    last_contributor = int(0)
    
    # Process all depth-sorted Gaussians affecting this tile (front-to-back)
    for i in range(range_start, range_end):
        # Get Gaussian ID
        gaussian_id = point_list[i]
        
        # Get Gaussian data
        xy = points_xy_image[gaussian_id]
        con_o = conic_opacity[gaussian_id]
        color = colors[gaussian_id]
        
        # Compute distance to Gaussian center
        d_x = xy[0] - pixf_x
        d_y = xy[1] - pixf_y
        
        # Increment contributor count for this pixel
        contributor_count += 1
        
        # Compute Gaussian power (exponent)
        power = -0.5 * (con_o[0] * d_x * d_x + con_o[2] * d_y * d_y) - con_o[1] * d_x * d_y
        
        # Skip if power is positive (too far away)
        if power > 0.0:
            continue
        
        # Compute alpha from power and opacity
        alpha = wp.min(0.99, con_o[3] * wp.exp(power))
        
        # Skip if alpha is too small
        if alpha < (1.0 / 255.0):
            continue
        
        
        # Test if we're close to fully opaque
        test_T = T * (1.0 - alpha)
        if test_T < 0.0001:
            break  # Early termination if pixel is almost opaque
        
        # Accumulate color contribution
        r += color[0] * alpha * T
        g += color[1] * alpha * T
        b += color[2] * alpha * T
        
        # Accumulate inverse depth
        expected_inv_depth += (1.0 / depths[gaussian_id]) * alpha * T
        
        # Update transmittance
        T = test_T
        
        last_contributor = contributor_count
    
    # Store final transmittance (T) and contributor count
    final_Ts[pix_y, pix_x] = T
    n_contrib[pix_y, pix_x] = last_contributor
    
    # Write final color to output buffer (color + background)
    rendered_image[pix_y, pix_x] = wp.vec3(
        r + T * background[0],
        g + T * background[1],
        b + T * background[2]
    )
    
    # Write depth to output buffer
    depth_image[pix_y, pix_x] = expected_inv_depth

@wp.kernel
def wp_duplicate_with_keys(
    points_xy_image: wp.array(dtype=wp.vec2),
    depths: wp.array(dtype=float),
    point_offsets: wp.array(dtype=int),
    point_list_keys_unsorted: wp.array(dtype=wp.int64),
    point_list_unsorted: wp.array(dtype=int),
    radii: wp.array(dtype=int),
    tile_grid: wp.vec3
):
    tid = wp.tid()

    if tid >= points_xy_image.shape[0]:
        return

    r = radii[tid]
    if r <= 0:
        return

    # Find the global offset into key/value buffers
    offset = 0
    if tid > 0:
        offset = point_offsets[tid - 1]

    pos = points_xy_image[tid]
    depth_val = depths[tid]

    rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_rect(pos, float(r), tile_grid)
    
    for y in range(rect_min_y, rect_max_y):
        for x in range(rect_min_x, rect_max_x):
            tile_id = y * int(tile_grid[0]) + x
            # Convert to int64 to avoid overflow during bit shift
            tile_id_64 = wp.int64(tile_id)
            shifted = tile_id_64 << wp.int64(32)
            depth_bits = wp.int64(float_bits_to_uint32(depth_val))
            # Combine tile ID and depth into single key
            key = wp.int64(shifted) | depth_bits

            point_list_keys_unsorted[offset] = key
            point_list_unsorted[offset] = tid
            offset += 1
            
@wp.kernel
def wp_identify_tile_ranges(
    num_rendered: int,
    point_list_keys: wp.array(dtype=wp.int64),
    ranges: wp.array(dtype=wp.vec2i)  # Each range is (start, end)
):
    idx = wp.tid()

    if idx >= num_rendered:
        return

    key = point_list_keys[idx]
    curr_tile = int(key >> wp.int64(32))

    # Set start of range if first element or tile changed
    if idx == 0:
        ranges[curr_tile][0] = 0
    else:
        prev_key = point_list_keys[idx - 1]
        prev_tile = int(prev_key >> wp.int64(32))
        if curr_tile != prev_tile:
            ranges[prev_tile][1] = idx
            ranges[curr_tile][0] = idx

    # Set end of range if last element
    if idx == num_rendered - 1:
        ranges[curr_tile][1] = num_rendered


@wp.kernel
def track_pixel_stats(
    rendered_image: wp.array2d(dtype=wp.vec3),
    depth_image: wp.array2d(dtype=float),
    background: wp.vec3,
    final_Ts: wp.array2d(dtype=float),
    n_contrib: wp.array2d(dtype=int),
    W: int,
    H: int
):
    """Kernel to track final transparency values and contributor counts for each pixel."""
    x, y = wp.tid()
    
    if x >= W or y >= H:
        return
    
    # Get the rendered pixel
    pixel = rendered_image[y, x]
    
    # Calculate approximate alpha transparency by checking for background contribution
    # If the pixel has no contribution from background, final_T should be close to 0
    # If it's mostly background, final_T will be close to 1
    diff_r = abs(pixel[0] - background[0])
    diff_g = abs(pixel[1] - background[1]) 
    diff_b = abs(pixel[2] - background[2])
    has_content = (diff_r > 0.01) or (diff_g > 0.01) or (diff_b > 0.01)
    
    if has_content:
        # Approximate final_T - in a real scenario this should already be tracked during rendering
        # We're just making sure it's populated for existing renderings
        if final_Ts[y, x] == 0.0:
            # If final_Ts hasn't been set during rendering, approximate it
            # Higher difference from background means lower T
            max_diff = max(diff_r, max(diff_g, diff_b))
            final_Ts[y, x] = 1.0 - min(0.99, max_diff)
        
        # Set n_contrib to 1 if we know the pixel has content but no contributor count
        if n_contrib[y, x] == 0:
            n_contrib[y, x] = 1

def render_gaussians(
    background,
    means3D,
    colors=None,
    opacity=None,
    scales=None,
    rotations=None,
    scale_modifier=1.0,
    viewmatrix=None,
    projmatrix=None,
    tan_fovx=0.5, 
    tan_fovy=0.5,
    image_height=256,
    image_width=256,
    sh=None,
    degree=3,
    campos=None,
    prefiltered=False,
    antialiasing=False,
    clamped=True,
    debug=False,
):
    """Render 3D Gaussians using Warp.
    
    Args:
        background: Background color tensor of shape (3,)
        means3D: 3D positions tensor of shape (N, 3)
        colors: Optional RGB colors tensor of shape (N, 3)
        opacity: Opacity values tensor of shape (N, 1) or (N,)
        scales: Scales tensor of shape (N, 3)
        rotations: Rotation quaternions of shape (N, 4)
        scale_modifier: Global scale modifier (float)
        viewmatrix: View matrix tensor of shape (4, 4)
        projmatrix: Projection matrix tensor of shape (4, 4)
        tan_fovx: Tangent of the horizontal field of view
        tan_fovy: Tangent of the vertical field of view
        image_height: Height of the output image
        image_width: Width of the output image
        sh: Spherical harmonics coefficients tensor of shape (N, D, 3)
        degree: Degree of spherical harmonics
        campos: Camera position tensor of shape (3,)
        prefiltered: Whether input Gaussians are prefiltered
        antialiasing: Whether to apply antialiasing
        clamped: Whether to clamp the colors
        debug: Whether to print debug information
        
    Returns:
        Tuple of (rendered_image, depth_image, intermediate_buffers)
    """
    # === PIPELINE STEP 1: SETUP & CONVERSION ===
    rendered_image = wp.zeros((image_height, image_width), dtype=wp.vec3, device=DEVICE)
    depth_image = wp.zeros((image_height, image_width), dtype=float, device=DEVICE)
    final_Ts = wp.zeros((image_height, image_width), dtype=float, device=DEVICE)
    n_contrib = wp.zeros((image_height, image_width), dtype=int, device=DEVICE)

    background_warp = wp.vec3(background[0], background[1], background[2])
    points_warp = to_warp_array(means3D, wp.vec3)
    
    sh_data = sh.reshape(-1, 3) if hasattr(sh, 'reshape') else sh
    shs_warp = to_warp_array(sh_data, wp.vec3)
    
    opacities_warp = to_warp_array(opacity, float, flatten=True)
    scales_warp = to_warp_array(scales, wp.vec3)
    rotations_warp = to_warp_array(rotations, wp.vec4)

    view_matrix_warp = wp.mat44(viewmatrix.flatten()) if not isinstance(viewmatrix, wp.mat44) else viewmatrix
    proj_matrix_warp = wp.mat44(projmatrix.flatten()) if not isinstance(projmatrix, wp.mat44) else projmatrix
    campos_warp = wp.vec3(campos[0], campos[1], campos[2]) if not isinstance(campos, wp.vec3) else campos
    
    tile_grid = wp.vec3((image_width + TILE_M - 1) // TILE_M, 
                        (image_height + TILE_N - 1) // TILE_N, 
                        1)
    
    num_points = points_warp.shape[0]
    radii = wp.zeros(num_points, dtype=int, device=DEVICE)
    points_xy_image = wp.zeros(num_points, dtype=wp.vec2, device=DEVICE)
    depths = wp.zeros(num_points, dtype=float, device=DEVICE)
    cov3Ds = wp.zeros(num_points, dtype=VEC6, device=DEVICE)
    rgb = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    conic_opacity = wp.zeros(num_points, dtype=wp.vec4, device=DEVICE)
    tiles_touched = wp.zeros(num_points, dtype=int, device=DEVICE)
    clamped_state = wp.zeros(num_points, dtype=wp.vec3, device=DEVICE)
    
    if debug:
        print(f"\nWARP RENDERING: {image_width}x{image_height} image, {num_points} gaussians")
        print(f"Colors: {'from SH' if colors is None else 'provided'}, SH degree: {degree}")
        print(f"Antialiasing: {antialiasing}, Prefiltered: {prefiltered}")

    # === PIPELINE STEP 2: PREPROCESSING ===
    # Transform 3D Gaussians to screen space, project covariances, evaluate SH colors
    wp.launch(
        kernel=wp_preprocess,
        dim=(num_points,),
        inputs=[
            points_warp,               # orig_points
            scales_warp,               # scales
            scale_modifier,            # scale_modifier
            rotations_warp,            # rotations
            opacities_warp,            # opacities
            shs_warp,                  # shs
            degree,
            clamped,                   # clamped
            view_matrix_warp,          # view_matrix
            proj_matrix_warp,          # proj_matrix
            campos_warp,               # cam_pos
            image_width,               # W
            image_height,              # H
            tan_fovx,                  # tan_fovx
            tan_fovy,                  # tan_fovy
            image_width / (2.0 * tan_fovx),  # focal_x
            image_height / (2.0 * tan_fovy),  # focal_y
            prefiltered,               # prefiltered
            antialiasing,              # antialiasing
            radii,                     # radii
            points_xy_image,           # points_xy_image
            depths,                    # depths
            cov3Ds,                    # cov3Ds
            rgb,                       # rgb
            conic_opacity,             # conic_opacity
            tile_grid,                 # tile_grid
            tiles_touched,             # tiles_touched
            clamped_state,             # clamped_state
        ],
    )
    # === PIPELINE STEP 3: SORTING & TILING ===
    # Compute prefix sums to determine how many tile-Gaussian pairs each point generates
    point_offsets = wp.zeros(num_points, dtype=int, device=DEVICE)
    wp.launch(
        kernel=wp_prefix_sum,
        dim=1,
        inputs=[
            tiles_touched,
            point_offsets
        ]
    )
    num_rendered = int(wp.to_torch(point_offsets)[-1].item())  # total number of duplicated entries
    if num_rendered > (1 << 30):
        # radix sort needs 2x memory
        raise ValueError("Number of rendered points exceeds the maximum supported by Warp.")

    # Create unsorted lists for duplicated points and their sorting keys (tile_id << 32 | depth)
    point_list_keys_unsorted = wp.zeros(num_rendered, dtype=wp.int64, device=DEVICE)
    point_list_unsorted = wp.zeros(num_rendered, dtype=int, device=DEVICE)
    point_list_keys = wp.zeros(num_rendered, dtype=wp.int64, device=DEVICE)
    point_list = wp.zeros(num_rendered, dtype=int, device=DEVICE)
    
    # Duplicate points for each tile they touch, creating sort keys for depth ordering
    wp.launch(
        kernel=wp_duplicate_with_keys,
        dim=num_points,
        inputs=[
            points_xy_image,
            depths,
            point_offsets,
            point_list_keys_unsorted,
            point_list_unsorted,
            radii,
            tile_grid
        ]
    )
    
    # Sort points by combined tile_id and depth for efficient tile-based rasterization
    point_list_keys_unsorted_padded = wp.zeros(num_rendered * 2, dtype=wp.int64, device=DEVICE) 
    point_list_unsorted_padded = wp.zeros(num_rendered * 2, dtype=int, device=DEVICE)
    
    # Copy data to padded arrays (radix sort requires extra space)
    wp.copy(point_list_keys_unsorted_padded, point_list_keys_unsorted)
    wp.copy(point_list_unsorted_padded, point_list_unsorted)
    
    # Perform radix sort to organize points by tile then depth (front-to-back)
    wp.utils.radix_sort_pairs(
        point_list_keys_unsorted_padded,  # keys to sort
        point_list_unsorted_padded,       # values to sort along with keys
        num_rendered                      # number of elements to sort
    )

    # Copy sorted results back to working arrays
    wp.launch(
        kernel=wp_copy_int64,
        dim=num_rendered,
        inputs=[
            point_list_keys_unsorted_padded,
            point_list_keys,
            num_rendered
        ]
    )
    
    wp.launch(
        kernel=wp_copy_int,
        dim=num_rendered, 
        inputs=[
            point_list_unsorted_padded,
            point_list,
            num_rendered
        ]
    )
    
    # Build tile ranges for efficient GPU access patterns
    tile_count = int(tile_grid[0] * tile_grid[1])
    ranges = wp.zeros(tile_count, dtype=wp.vec2i, device=DEVICE)  # each is (start, end)

    if num_rendered > 0:
        # Identify start/end indices for each tile in the sorted point list
        wp.launch(
            kernel=wp_identify_tile_ranges,
            dim=num_rendered,
            inputs=[
                num_rendered,
                point_list_keys,
                ranges
            ]
        )
        
        # === PIPELINE STEP 4: RASTERIZATION ===
        # Render tiles in parallel, each thread handling one pixel with alpha compositing
        wp.launch(
            kernel=wp_render_gaussians,
            dim=(int(tile_grid[0]), int(tile_grid[1]), TILE_M, TILE_N),
            inputs=[
                rendered_image,        # Output color image
                depth_image,           # Output depth image
                ranges,                # Tile ranges
                point_list,            # Sorted point indices
                image_width,           # Image width
                image_height,          # Image height
                points_xy_image,       # 2D points
                rgb,                   # Precomputed colors
                conic_opacity,         # Conic matrices and opacities
                depths,                # Depth values
                background_warp,       # Background color
                tile_grid,             # Tile grid configuration
                final_Ts,              # Final transparency values
                n_contrib,             # Number of contributors per pixel
            ]
        )
        
        # === PIPELINE STEP 5: POST-PROCESSING ===
        # Track pixel statistics (transmittance and contributor counts) for gradient computation
        wp.launch(
            kernel=track_pixel_stats,
            dim=(image_width, image_height),
            inputs=[
                rendered_image,
                depth_image,
                background_warp,
                final_Ts,
                n_contrib,
                image_width,
                image_height
            ]
        )

    return rendered_image, depth_image, {
        "radii": radii,
        "point_offsets": point_offsets,
        "points_xy_image": points_xy_image,
        "depths": depths,
        "colors": rgb,
        "cov3Ds": cov3Ds,
        "conic_opacity": conic_opacity,
        "point_list": point_list,
        "ranges": ranges,
        "final_Ts": final_Ts,  # Add final_Ts to intermediate buffers
        "n_contrib": n_contrib,  # Add contributor count to intermediate buffers
        "clamped_state": clamped_state  # Add clamped state to intermediate buffers
    }