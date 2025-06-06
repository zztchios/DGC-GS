#pragma once

int select(
    const int P, const int width, int height,
    const float* means3D,
    const float* opacities,
    const float* gt_depth,
    const float depth_opa_thresh, 
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    // float* means2D,
    // float* p_views,
    // float* cov3Ds,
    // float* conic_opacity,
    int* out_idx,
    float* out_xyz,
    // float* radii,
    bool debug);

int preComputeDepthScale(
    const int P, const int width, int height,
    const float* means3D,
    const float* viewmatrix,
    const float* projmatrix,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    int* out_mask,
    float* out_depth,
    bool debug);

int alphaSelect(
    const torch::Tensor& pixels,
    const uint32_t pix_count,
	int W, int H, int P, int R, float thresh, 
    char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
    torch::Tensor& out_range,
    torch::Tensor& out_idxs,
    const bool debug);

int select_tiles(
    std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
    const int P, const int width, int height,
    const float* means3D,
    const float* opacities,
    const int* tiles,
    const int K,
    const int h,
    const int w,
    const int T, 
    const float alpha_thresh, 
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    int* out_idxs,
    float* out_alphas, 
    int* radii, 
    bool debug);