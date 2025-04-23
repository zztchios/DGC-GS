#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "rasterizer_impl.h"
using namespace CudaRasterizer;

#include "auxiliary.h"
#include "depth_select.h"

#define CudaMalloc(p, c) cudaMalloc(&p, c); cudaMemset(p,0,c);

// #include <inttypes.h>

__forceinline__ __device__ float3 transformPoint4x3Inv(
    const float3& p,     // 存储的高斯球的位置信息
    const float* matrix  // w2c矩阵
    ) 
    {
        // size_t length = sizeof(matrix) / sizeof(matrix[0]);
        // printf("matrix: %d\n", length);
        // printf("matrix: %f %f %f %f\n", matrix[0], matrix[1], matrix[2], matrix[3]);
        float3 pt = {
            p.x - matrix[12],
            p.y - matrix[13],
            p.z - matrix[14]
        };      // 去除平移分量
    // w2c矩阵 -> c2w矩阵
    float3 transformedInv = {
        matrix[0] * pt.x + matrix[1] * pt.y + matrix[2] * pt.z,
        matrix[4] * pt.x + matrix[5] * pt.y + matrix[6] * pt.z,
        matrix[8] * pt.x + matrix[9] * pt.y + matrix[10] * pt.z
    };
    return transformedInv;
}

// Forward version of 2D covariance matrix computation
__device__ float3 mycomputeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
    // The following models the steps outlined by equations 29
    // and 31 in "EWA Splatting" (Zwicker et al., 2002). 
    // Additionally considers aspect / scaling of viewport.
    // Transposes used to account for row-/column-major conventions.
    float3 t = transformPoint4x3(mean, viewmatrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
        0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0, 0, 0);

    glm::mat3 W = glm::mat3(
        viewmatrix[0], viewmatrix[4], viewmatrix[8],
        viewmatrix[1], viewmatrix[5], viewmatrix[9],
        viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    glm::mat3 T = W * J;

    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]);

    glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void mycomputeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
    // Create scaling matrix
    glm::mat3 S = glm::mat3(1.0f);
    S[0][0] = mod * scale.x;
    S[1][1] = mod * scale.y;
    S[2][2] = mod * scale.z;

    // Normalize quaternion to get valid rotation
    glm::vec4 q = rot;// / glm::length(rot);
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    glm::mat3 M = S * R;

    // Compute 3D world covariance matrix Sigma
    glm::mat3 Sigma = glm::transpose(M) * M;

    // Covariance is symmetric, only store upper right
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][1];
    cov3D[4] = Sigma[1][2];
    cov3D[5] = Sigma[2][2];
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t mygetHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

__global__ void identifyPixRanges(
    int L, // 排序列表中的元素个数 
    uint64_t* point_list_keys, // 排过序的keys
    uint32_t* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
    // printf("pix_64: %lu\n", key);
	uint32_t currpix = key >> 32;       // 当前tile
    // printf("pix: %u\n", currpix);
	if (idx == 0)
		ranges[currpix*2] = 0;          // 边界条件：tile 0的起始位置
	else
	{
		uint32_t prevpix = point_list_keys[idx - 1] >> 32;
		if (currpix != prevpix)
        // 上一个元素和我处于不同的tile，
		// 那我是上一个tile的终止位置和我所在tile的起始位置
		{
			ranges[prevpix*2+1] = idx;
			ranges[currpix*2] = idx;
		}
	}
	if (idx == L - 1)
		ranges[currpix*2+1] = L;        // 边界条件：最后一个tile的终止位置
}

__global__ void projectCUDA(int P, 
    const float* orig_points,
    const glm::vec3* scales,
    const float scale_modifier,
    const glm::vec4* rotations,
    const float* opacities,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const glm::vec3* cam_pos,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    float* points_xy_image,     //means2D
    float* p_views,
    float* cov3Ds,
    float* conic_opacity,       // 椭圆对应二次型的矩阵和不透明度的打包存储
    bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank(); // 该函数预处理第idx个Gaussian
    if (idx >= P)
        return;

    // Perform near culling, quit if outside.
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // If 3D covariance matrix is precomputed, use it, otherwise compute
    // from scaling and rotation parameters. 
    const float* cov3D;
    if (cov3D_precomp != nullptr)
    {
        cov3D = cov3D_precomp + idx * 6;
    }
    else
    {
        mycomputeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        cov3D = cov3Ds + idx * 6;
    }

    // Compute 2D screen-space covariance matrix
    float3 cov = mycomputeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // Invert covariance (EWA algorithm)
    float det = (cov.x * cov.z - cov.y * cov.y);        // 二维协方差矩阵的行列式
    if (det == 0.0f)
        return;
    float det_inv = 1.f / det;                          // 行列式的逆
    float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
    // conic是cone的形容词，意为“圆锥的”。猜测这里是指圆锥曲线（椭圆）。
	// 二阶矩阵求逆口诀：“主对调，副相反”。
    
    // 这里就是截取Gaussian的中心部位（3σ原则），只取像平面上半径为my_radius的部分
    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    // Store some useful helper data for the next steps.
    (p_views+idx*3)[0] = p_view.x;  //p_views+idx*3表示在数组p_views中的位置为idx*3的元素。
    (p_views+idx*3)[1] = p_view.y;
    (p_views+idx*3)[2] = p_view.z;

    (points_xy_image+idx*2)[0] = point_image.x;
    (points_xy_image+idx*2)[1] = point_image.y;

    (conic_opacity+idx*4)[0] = conic.x;
    (conic_opacity+idx*4)[1] = conic.y;
    (conic_opacity+idx*4)[2] = conic.z;
    (conic_opacity+idx*4)[3] = opacities[idx];
    //等同于 conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] }; 
}

__global__ void pointProjectCUDA(int P, 
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    const int W, int H,
    const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    float* points_xy_image,
    float* p_views,
    bool prefiltered)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Perform near culling, quit if outside.
    float3 p_view;
    if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
        return;

    // Transform point by projecting
    float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
    float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    float p_w = 1.0f / (p_hom.w + 0.0000001f);
    float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

    // Store some useful helper data for the next steps.
    (p_views+idx*3)[0] = p_view.x;
    (p_views+idx*3)[1] = p_view.y;
    (p_views+idx*3)[2] = p_view.z;

    (points_xy_image+idx*2)[0] = point_image.x;
    (points_xy_image+idx*2)[1] = point_image.y;

}

__global__ void myduplicateWithKeys(
    int P, int W, int H,
    const float* points_xy,
    float* p_views,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    // printf("idx: %d\n", idx);
    uint2 pix = {static_cast<uint32_t>(points_xy[idx*2]), static_cast<uint32_t>(points_xy[idx*2+1])};
    // printf("pixel: (%d, %d)\n", pix.x, pix.y);
    // Generate no key/value pair for invisible Gaussians
    if (p_views[idx*3+2] > 0 && pix.x<W && pix.y<H)
    {
        // uint2 pix = {static_cast<uint32_t>(points_xy[idx*2]), static_cast<uint32_t>(points_xy[idx*2+1])};

        uint64_t key = pix.y * W + pix.x;       // tile的ID
        // printf("key: %lu\n",key);
        key <<= 32;                             // 放在高位
        key |= *((uint32_t*)&p_views[idx*3+2]); // 低位是深度
        // printf("key_64: %lu\n",key);
        gaussian_keys_unsorted[idx] = key;
        gaussian_values_unsorted[idx] = idx;
    } else {
        uint64_t key = (W-1) * (H-1) + 1;
        // printf(" %lu",key);
        key <<= 32;
        gaussian_keys_unsorted[idx] = key;
        gaussian_values_unsorted[idx] = idx;
    }
}

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
selectCUDA(
    const uint32_t* __restrict__ ranges,        // 每个tile对应排过序的数组中的哪一部分
    const uint32_t* __restrict__ point_list,    // 按tile、深度排序后的Gaussian ID列表
    int W, int H,
    const float* __restrict__ points_xy_image,  // 图像上每个Gaussian中心的2D坐标
    const float* __restrict__ conic_opacity,
    const float* gt_depth,
    const float depth_opa_thresh, 
    const float* p_views,                       // 存储的高斯球的位置信息
    const float* viewmatrix,                    // W2C矩阵
    int* out_idx,
    float* out_xyz)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    // uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };   
    // 负责的tile的坐标较小的那个角的坐标
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    // 负责的tile的坐标较大的那个角的坐标
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    // 负责哪个像素
    uint32_t pix_id = W * pix.y + pix.x;
    // 负责的像素在整张图片中排行老几
    float2 pixf = { (float)pix.x+0.5f, (float)pix.y+0.5f }; // pix的浮点数版本, 加入0.5f是为了多推一段，其实就是防止球体表面凸出来

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;    // 看看我负责的像素有没有跑到图像外面去
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = {ranges[pix_id*2], ranges[pix_id*2+1]};
    // const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;   // 我要处理的Gaussian个数

    // Initialize helper variables
    float T = 1.0f;

    bool pushed = false;

    float depth = gt_depth[pix_id];
    // if (toDo > 0) {
    //     printf("pix:(%u,%u), pix_id:%u, toDO:(%u,%u)\n",pix.x, pix.y, pix_id, range.x, range.y);
    // }
    // printf("pix:(%u,%u), pix_id:%u, toDO:(%u,%u)\n",pix.x, pix.y, pix_id, range.x, range.y);
    // Iterate over current batch
    for (int j = 0; !done && j < toDo; j++)
    {
        // // Keep track of current position in range
        // contributor++;   // 多少个Gaussian对该像素的颜色有贡献
        uint32_t idx = point_list[range.x+j];
        float3 p_view = {p_views[idx*3], p_views[idx*3+1], p_views[idx*3+2]};

        // Resample using conic matrix (cf. "Surface 
        // Splatting" by Zwicker et al., 2001)
        float2 xy = {points_xy_image[idx*2], points_xy_image[idx*2+1]};
        float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        float4 con_o = {conic_opacity[idx*4], conic_opacity[idx*4+1], conic_opacity[idx*4+2], conic_opacity[idx*4+3]};
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        // Eq. (2) from 3D Gaussian splatting paper.
        // Obtain alpha by multiplying with Gaussian opacity
        // and its exponential falloff from mean.
        // Avoid numerical instabilities (see paper appendix). 
        float alpha = min(0.99f, con_o.w * exp(power));

        // TODO： if alpha > 阈值:
        //          if p_view.z < gt_depth:
        //              推到gt_depth，反投影，记录
        //              pushed = true
        //          else if: p_view.z > gt_depth:
        //              if !pushed:
        //                  拉到gt_depth反投影，记录
        //              done = true
        if (alpha > depth_opa_thresh) {
            if (p_view.z < depth) {
                //计算
                p_view.z = depth;
                float3 xyz = transformPoint4x3Inv(p_view, viewmatrix); // cam2world
                out_idx[idx] = 1;
                out_xyz[idx*3] = xyz.x;
                out_xyz[idx*3+1] = xyz.y;
                out_xyz[idx*3+2] = xyz.z;
                pushed = true;
            } 
            else if (p_view.z > depth) {
                // if (!pushed) {
                //     //计算
                //     p_view.z = depth;
                //     float3 xyz = transformPoint4x3Inv(p_view, viewmatrix);
                //     out_idx[idx] = 1;
                //     out_xyz[idx*3] = xyz.x;
                //     out_xyz[idx*3+1] = xyz.y;
                //     out_xyz[idx*3+2] = xyz.z;
                //     // printf(" idx: %d", idx);
                // }
                done = true;
            }
        }

    }

}


__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
preDepthCUDA(
    const uint32_t* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float* p_views,
    int* out_mask,
    float* out_depth)
{
    // Identify current tile and associated min/max pixel range.
    auto block = cg::this_thread_block();
    // uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    uint32_t pix_id = W * pix.y + pix.x;

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = {ranges[pix_id*2], ranges[pix_id*2+1]};
    // const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    for (int j = 0; !done && j < toDo; j++)
    {
        uint32_t idx = point_list[range.x+j];
        float3 p_view = {p_views[idx*3], p_views[idx*3+1], p_views[idx*3+2]};

        out_mask[pix_id] = 1;
        out_depth[pix_id] = p_view.z;
        done = true;

    }

}

int select(
    const int P,                    // 需要检查的点的个数
    const int width, int height,
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
    bool debug)
{
    float *means2D, *p_views, *cov3Ds, *conic_opacity;
    // cudaMalloc(&means2D, P*2*sizeof(float));
    // cudaMalloc(&p_views, P*3*sizeof(float));
    // cudaMalloc(&cov3Ds, P*6*sizeof(float));
    // cudaMalloc(&conic_opacity, P*4*sizeof(float));

    CudaMalloc(means2D, P*2*sizeof(float))   //内存分配 P=means3D.size(0)也就是有多少个高斯球
    CudaMalloc(p_views, P*3*sizeof(float))
    CudaMalloc(cov3Ds, P*6*sizeof(float))
    CudaMalloc(conic_opacity, P*4*sizeof(float))

    const float focal_y = height / (2.0f * tan_fovy);   // y方向的焦距
    const float focal_x = width / (2.0f * tan_fovx);    // x方向的焦距

    projectCUDA << <(P + 255) / 256, 256 >> > (
        P,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        cov3D_precomp,
        viewmatrix, 
        projmatrix,
        (glm::vec3*)cam_pos,
        width, height,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        means2D,
        p_views,
        cov3Ds,
        conic_opacity,
        prefiltered
    )CHECK_CUDA(, debug)

    // torch::TensorOptions option = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

    // uint64_t* point_list_keys_unsorted = (uint64_t*)torch::full({P*8}, 0, option).contiguous().data_ptr();
    // uint32_t* point_list_unsorted = (uint32_t*)torch::full({P*4}, 0, option).contiguous().data_ptr();
    // uint64_t* point_list_keys = (uint64_t*)torch::full({P*8}, 0, option).contiguous().data_ptr();
    // uint32_t* point_list = (uint32_t*)torch::full({P*4}, 0, option).contiguous().data_ptr();

    // uint32_t* ranges = (uint32_t*)torch::full({width*height*4*2+1}, 0, option).contiguous().data_ptr();
    uint64_t *point_list_keys_unsorted, *point_list_keys;
    uint32_t *point_list_unsorted, *point_list, *ranges;
    // cudaMalloc(&point_list_keys_unsorted, P*sizeof(uint64_t));
    // cudaMalloc(&point_list_keys, P*sizeof(uint64_t));
    // cudaMalloc(&point_list_unsorted, P*sizeof(uint32_t));
    // cudaMalloc(&point_list, P*sizeof(uint32_t));
    // cudaMalloc(&ranges, (width*height+1)*2*sizeof(uint32_t));
    CudaMalloc(point_list_keys_unsorted, P*sizeof(uint64_t))
    CudaMalloc(point_list_keys, P*sizeof(uint64_t))
    CudaMalloc(point_list_unsorted, P*sizeof(uint32_t))
    CudaMalloc(point_list, P*sizeof(uint32_t))
    CudaMalloc(ranges, (width*height+1)*2*sizeof(uint32_t))

    size_t sorting_size = 0;
    void* tmp_sorting_space = nullptr;

    cub::DeviceRadixSort::SortPairs(
        tmp_sorting_space, sorting_size,
        point_list_keys_unsorted, point_list_keys,
        point_list_unsorted, point_list, P);

    CudaMalloc(tmp_sorting_space, sorting_size)


    myduplicateWithKeys << <(P + 255) / 256, 256 >> > (
        P, width, height,
        means2D,
        p_views,
        point_list_keys_unsorted,
        point_list_unsorted // 生成排序所用的keys和values
    )CHECK_CUDA(, debug)


    // uint64_t* point_list_keys_unsort_cpu = (uint64_t*)torch::full({P*8}, 0, torch::TensorOptions().dtype(torch::kUInt8)).contiguous().data_ptr();
    // cudaMemcpy(point_list_keys_unsort_cpu, point_list_keys_unsorted, P*8, cudaMemcpyDeviceToHost);
    // for (int i=0;i<P;++i) {
    //     // uint32_t pix = point_list_keys_unsort_cpu[i]>>32;
    //     uint64_t pix = point_list_keys_unsort_cpu[i];
    //     printf(" %lu", pix);
    // }

    // uint32_t* point_list_unsort_cpu = (uint32_t*)torch::full({P*4}, 0, torch::TensorOptions().dtype(torch::kUInt8)).contiguous().data_ptr();
    // cudaMemcpy(point_list_unsort_cpu, point_list_unsorted, P*4, cudaMemcpyDeviceToHost);
    // for (int i=0;i<P;++i) {
    //     // uint32_t pix = point_list_keys_unsort_cpu[i]>>32;
    //     uint32_t pix = point_list_unsort_cpu[i];
    //     printf(" %u", pix);
    // }

    int bit = mygetHigherMsb(width * height);

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        tmp_sorting_space, sorting_size,
        point_list_keys_unsorted, point_list_keys,
        point_list_unsorted, point_list,
        P), debug)
    // 进行排序，按keys排序：每个tile对应的Gaussians按深度放在一起；value是Gaussian的ID
    
    cudaFree(tmp_sorting_space);


    // uint64_t* point_list_keys_cpu = (uint64_t*)torch::full({P*8}, 0, torch::TensorOptions().dtype(torch::kUInt8)).contiguous().data_ptr();
    // cudaMemcpy(point_list_keys_cpu, point_list_keys, P*8, cudaMemcpyDeviceToHost);
    // for (int i=0;i<P;++i) {
    //     // uint32_t pix = point_list_keys_cpu[i]>>32;
    //     uint64_t pix = point_list_keys_cpu[i];
    //     printf(" %lu", pix);
    // }


    // Identify start and end of per-pixel workloads in sorted list
    if (P > 0)
        identifyPixRanges << <(P + 255) / 256, 256 >> > (
            P,
            point_list_keys,
            ranges)CHECK_CUDA(, debug)

    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
    
    selectCUDA << <grid, block >> > (
        ranges,
        point_list,
        width, height,
        means2D,
        conic_opacity,
        gt_depth,
        depth_opa_thresh,
        p_views,     // 存储的高斯球的位置信息
        viewmatrix,  // W2C矩阵
        out_idx,
        out_xyz)CHECK_CUDA(, debug)


    cudaFree(means2D);
    cudaFree(p_views);
    cudaFree(cov3Ds);
    cudaFree(conic_opacity);

    cudaFree(point_list_keys_unsorted);
    cudaFree(point_list_keys);
    cudaFree(point_list_unsorted);
    cudaFree(point_list);
    cudaFree(ranges);

    return 0;
}

int preComputeDepthScale(
    const int P, const int width, int height,
    const float* means3D,
    const float* viewmatrix,
    const float* projmatrix,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    int* out_mask,
    float* out_depth,
    bool debug)
{
    float *means2D, *p_views;

    CudaMalloc(means2D, P*2*sizeof(float))
    CudaMalloc(p_views, P*3*sizeof(float))

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    pointProjectCUDA << <(P + 255) / 256, 256 >> > (
        P,
        means3D,
        viewmatrix, 
        projmatrix,
        width, height,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        means2D,
        p_views,
        prefiltered
    )CHECK_CUDA(, debug)

    uint64_t *point_list_keys_unsorted, *point_list_keys;
    uint32_t *point_list_unsorted, *point_list, *ranges;

    CudaMalloc(point_list_keys_unsorted, P*sizeof(uint64_t))
    CudaMalloc(point_list_keys, P*sizeof(uint64_t))
    CudaMalloc(point_list_unsorted, P*sizeof(uint32_t))
    CudaMalloc(point_list, P*sizeof(uint32_t))
    CudaMalloc(ranges, (width*height+1)*2*sizeof(uint32_t))

    size_t sorting_size = 0;
    void* tmp_sorting_space = nullptr;

    cub::DeviceRadixSort::SortPairs(
        tmp_sorting_space, sorting_size,
        point_list_keys_unsorted, point_list_keys,
        point_list_unsorted, point_list, P);

    CudaMalloc(tmp_sorting_space, sorting_size)


    myduplicateWithKeys << <(P + 255) / 256, 256 >> > (
        P, width, height,
        means2D,
        p_views,
        point_list_keys_unsorted,
        point_list_unsorted
    )CHECK_CUDA(, debug)

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
        tmp_sorting_space, sorting_size,
        point_list_keys_unsorted, point_list_keys,
        point_list_unsorted, point_list,
        P), debug)
    
    cudaFree(tmp_sorting_space);

    // Identify start and end of per-pixel workloads in sorted list
    if (P > 0)
        identifyPixRanges << <(P + 255) / 256, 256 >> > (
            P,
            point_list_keys,
            ranges)CHECK_CUDA(, debug)

    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
    
    preDepthCUDA << <grid, block >> > (
        ranges,
        point_list,
        width, height,
        p_views,
        out_mask,
        out_depth)CHECK_CUDA(, debug)


    cudaFree(means2D);
    cudaFree(p_views);

    cudaFree(point_list_keys_unsorted);
    cudaFree(point_list_keys);
    cudaFree(point_list_unsorted);
    cudaFree(point_list);
    cudaFree(ranges);

    return 0;
}



__global__ void preCountCUDA(
    const int* __restrict__ pixels,
    int pix_count, float thresh, 
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	int* __restrict__ out_count)
{
    //TODO：根据thread_id，取出要处理的像素点
    //      计算像素点所在tile
    //      根据tile，取出要处理的gaussian范围
    //      遍历GS,计算 alpha 大于阈值的数量 
    //      根据数量结果，分配显存，填写range
    //      重新遍历GS，将大于阈值的GS id 写入结果

    //根据thread_id，取出要处理的像素点
    auto block = cg::this_thread_block();
    int deal_id = block.group_index().x * block.group_dim().x + block.thread_index().x;
    if (deal_id >= pix_count)
        return;

    uint2 pix = {pixels[deal_id * 2], pixels[deal_id * 2 + 1]};
    uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

    //计算像素点所在tile
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint32_t tile = pix.y / BLOCK_Y * horizontal_blocks + pix.x / BLOCK_X;

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[tile];
    // const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Initialize helper variables
    float T = 1.0f;
    // printf("deal_id: %d, pix: (%d, %d), tile: %d", deal_id, pix.x, pix.y, tile);
    int count = 0;
    for (int j = 0; !done && j < toDo; j++)
    {
        // // Keep track of current position in range
        // contributor++;
        uint32_t idx = point_list[range.x+j];

        // Resample using conic matrix (cf. "Surface 
        // Splatting" by Zwicker et al., 2001)
        float2 xy = points_xy_image[idx];
        float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        float4 con_o = conic_opacity[idx];
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        float alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;

        float last_alpha =  alpha * T;
        if (last_alpha > thresh) {
            count += 1;
        }


        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f)
        {
            done = true;
            continue;
        }


        T = test_T;

    }
    out_count[deal_id] = count;
}


__global__ void alphaCUDA(
    const int* __restrict__ pixels,
    uint32_t pix_count, float thresh, 
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	int* __restrict__ out_idxs,
    const int* __restrict__ idxs_range)
{
    //TODO：根据thread_id，取出要处理的像素点
    //      计算像素点所在tile
    //      根据tile，取出要处理的gaussian范围
    //      遍历GS,计算 alpha 大于阈值的数量 
    //      根据数量结果，分配显存，填写range
    //      重新遍历GS，将大于阈值的GS id 写入结果

    //根据thread_id，取出要处理的像素点
    auto block = cg::this_thread_block();
    int deal_id = block.group_index().x * block.group_dim().x + block.thread_index().x;
    if (deal_id >= pix_count)
        return;

    uint2 pix = {pixels[deal_id * 2], pixels[deal_id * 2 + 1]};
    uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

    //计算像素点所在tile
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint32_t tile = pix.y / BLOCK_Y * horizontal_blocks + pix.x / BLOCK_X;

    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[tile];
    // const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Initialize helper variables
    float T = 1.0f;

    uint32_t index = idxs_range[deal_id*2];
    uint32_t end_index = idxs_range[deal_id*2+1];
    
    for (int j = 0; !done && j < toDo; j++)
    {
        // // Keep track of current position in range
        // contributor++;
        uint32_t idx = point_list[range.x+j];

        // Resample using conic matrix (cf. "Surface 
        // Splatting" by Zwicker et al., 2001)
        float2 xy = points_xy_image[idx];
        float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        float4 con_o = conic_opacity[idx];
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        float alpha = min(0.99f, con_o.w * exp(power));
        if (alpha < 1.0f / 255.0f)
            continue;

        float last_alpha =  alpha * T;
        if (last_alpha > thresh) {
            // out_count[deal_id] += 1;
            
            out_idxs[index] = idx;
            index++;
        }


        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f)
        {
            done = true;
            assert(index == end_index);
            continue;
        }


        T = test_T;

    }
}

__global__ void getRange(
    const uint32_t pix_count,
    const int* __restrict__ acc_count,
    int* __restrict__ out_range)
{
    auto block = cg::this_thread_block();
    int deal_id = block.group_index().x * block.group_dim().x + block.thread_index().x;
    if (deal_id >= pix_count)
        return;

    if (deal_id == 0) {
        out_range[deal_id*2] = 0;
    } else {
        out_range[deal_id*2] = acc_count[deal_id-1];
    }
    out_range[deal_id*2 + 1] = acc_count[deal_id];
}


int alphaSelect(
    const torch::Tensor& pixels,
    const uint32_t pix_count,
	int W, int H, int P, int R, float thresh, 
    char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
    torch::Tensor& out_range,
    torch::Tensor& out_idxs,
    const bool debug)
{

    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, W * H);

    int *out_count, *acc_count;
    CudaMalloc(out_count, pix_count*sizeof(int))
    CudaMalloc(acc_count, pix_count*sizeof(int))
    // printf("222222222222222222222");
    preCountCUDA << <(pix_count + 255) / 256, 256 >> > (
        pixels.contiguous().data_ptr<int>() ,
        pix_count, thresh, 
        imgState.ranges,
		binningState.point_list,
        W, H,
        geomState.means2D,
		geomState.conic_opacity,
        out_count
        )CHECK_CUDA(, debug)
    // printf("25252525252525252525");
    // printf("=============%d", out_count[0]);
    //累加
    void* temp_space;
    size_t temp_storage_bytes = 0;
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, out_count, acc_count, pix_count), debug)
    CudaMalloc(temp_space, temp_storage_bytes)
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(temp_space, temp_storage_bytes, out_count, acc_count, pix_count), debug)
    cudaFree(temp_space);
    // printf("262626262626262626");
    // printf("=============%d", out_count[0]);
    // printf("=============%d", acc_count[0]);
    uint32_t out_num = 0;
    cudaMemcpy(&out_num, &acc_count[pix_count - 1], sizeof(int), cudaMemcpyDeviceToHost);
    // uint32_t out_num = acc_count[pix_count - 1];
    // printf("2727272727272");
    out_idxs.resize_({out_num});
    // printf("33333333333333333333333333");
    getRange<< <(pix_count + 255) / 256, 256 >> > (
        pix_count,
        acc_count,
        out_range.contiguous().data_ptr<int>()
        )CHECK_CUDA(, debug)

    // printf("4444444444444444444444444");
    alphaCUDA<< <(pix_count + 255) / 256, 256 >> >(
        pixels.contiguous().data_ptr<int>(),
        pix_count, thresh, 
        imgState.ranges,
		binningState.point_list,
        W, H,
        geomState.means2D,
		geomState.conic_opacity,
        out_idxs.contiguous().data_ptr<int>(),
        out_range.contiguous().data_ptr<int>())CHECK_CUDA(, debug)

    cudaFree(out_count);
    cudaFree(acc_count);

    return out_num;

}


__global__ void duplicateWithKeysWithTile(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid, int w, int h)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRectWithTile(points_xy[idx], radii[idx], rect_min, rect_max, grid, w, h);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}


__global__ void alphaTileCUDA(
    const int* __restrict__ tiles,
    int w, int h, 
    int K, float alpha_thresh, 
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ alpha_list)
{
    //TODO：根据thread_id，取出要处理的像素点
    //      计算像素点所在tile
    //      根据tile，取出要处理的gaussian范围
    //      遍历GS,计算 alpha 大于阈值的数量 
    //      根据数量结果，分配显存，填写range
    //      重新遍历GS，将大于阈值的GS id 写入结果

    //根据thread_id，取出要处理的像素点
    auto block = cg::this_thread_block();
    int tile_id = tiles[block.group_index().x];
    // printf("tile_id: %d\n", tile_id);

    uint32_t horizontal_blocks = (W + w - 1) / w;
    // printf("horizontal_blocks: %d\n", horizontal_blocks);
    uint2 pix_min = { tile_id % horizontal_blocks * w, tile_id / horizontal_blocks * h };
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };


    // Check if this thread is associated with a valid pixel or outside.
    bool inside = pix.x < W&& pix.y < H;
    // Done threads can help with fetching, but don't rasterize
    bool done = !inside;

    // Load start/end range of IDs to process in bit sorted list.
    uint2 range = ranges[tile_id];
    // const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int toDo = range.y - range.x;

    // Initialize helper variables
    float T = 1.0f;
    
    for (int j = 0; !done && j < toDo; j++)
    {
        // // Keep track of current position in range
        // contributor++;
        uint32_t idx = point_list[range.x+j];

        // Resample using conic matrix (cf. "Surface 
        // Splatting" by Zwicker et al., 2001)
        float2 xy = points_xy_image[idx];
        float2 d = { xy.x - pixf.x, xy.y - pixf.y };
        // printf("pix:(%f, %f), xy:(%f, %f), d:(%f, %f", pixf.x, pixf.y, xy.x, xy.y, d.x, d.y);
        float4 con_o = conic_opacity[idx];
        float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
        if (power > 0.0f)
            continue;

        float alpha = min(0.99f, con_o.w * exp(power));
        // printf("gs:%d, tile:%d, alpha: %f\n", idx, tile_id, alpha);
        if (alpha < 1.0f / 255.0f)
            continue;

        float last_alpha =  alpha * T;
        // printf("tile:%d, alpha: %f, last_alpha: %f\n", tile_id, alpha, last_alpha);
        if (last_alpha > alpha_thresh) {
            // alpha_list[range.x+j] = last_alpha;
            atomicAdd(&alpha_list[range.x+j], last_alpha);
            // printf("gs:%d, tile:%d, last_alpha: %f, alpha_list: %f\n", idx, tile_id, last_alpha, alpha_list[range.x+j]);
        }


        float test_T = T * (1 - alpha);
        if (test_T < 0.0001f)
        {
            done = true;
            continue;
        }


        T = test_T;

    }
}


__global__ void selectTilesPreprocessCUDA(int P, int w, int h, 
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	bool* clamped,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		mycomputeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = mycomputeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRectWithTile(point_image, my_radius, rect_min, rect_max, grid, w, h);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

__global__ void combineTileAlpha(
    int R, 
    float* alpha_list, 
    uint64_t* point_list_keys, 
    uint64_t* tile_alpha_keys)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= R)
		return;

    int32_t tile = point_list_keys[idx] >> 32;
    float alpha = alpha_list[idx];
    
    uint64_t key = tile;
    key <<= 32;
    key |= *((uint32_t*)&alpha);
    tile_alpha_keys[idx] = key;

}

__global__ void topKAlphaIdx(
    const int *tiles, int K, 
    uint2* tile_alpha_ranges, 
    uint64_t* tile_alpha_sorted, 
    uint32_t* point_list_alpha_sorted, 
    int* out_idxs, float* out_alphas)
{
    auto block = cg::this_thread_block();
    int tile_id = tiles[block.group_index().x];

    uint2 range = tile_alpha_ranges[tile_id];
    int start = range.x;
    int end = range.y;

    // int idx = block.thread_index().x + start;
    int idx = end - 1 - block.thread_index().x;
    if (idx < start)
        return;

    uint64_t key = tile_alpha_sorted[idx];
    uint32_t alp = static_cast<uint32_t>(key & 0xFFFFFFFF);

    uint32_t tile = key >> 32;
    assert(tile == tile_id);

    uint32_t point_idx = point_list_alpha_sorted[idx];
    out_idxs[block.group_index().x*K + block.thread_index().x] = point_idx;
    out_alphas[block.group_index().x*K + block.thread_index().x] = *((float*)&alp);
}


__global__ void myIdentifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}


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
    bool debug)
{
    const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + w - 1) / w, (height + h - 1) / h, 1);
	dim3 block(w, h, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	selectTilesPreprocessCUDA<< <(P + 255) / 256, 256 >> >(
		P, w, h,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		geomState.clamped,
		cov3D_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	)CHECK_CUDA(, debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeysWithTile << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid, w, h)
	CHECK_CUDA(, debug)

	int bit = mygetHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		myIdentifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges)CHECK_CUDA(, debug);

    float* alpha_list;
    CudaMalloc(alpha_list, num_rendered*sizeof(float))

    alphaTileCUDA<< <T, (w, h) >> >(
        tiles,
        w, h, 
        K, alpha_thresh, 
        imgState.ranges,
		binningState.point_list,
        width, height,
        geomState.means2D,
		geomState.conic_opacity,
        alpha_list)CHECK_CUDA(, debug)

    uint64_t* tile_alpha_keys;
    CudaMalloc(tile_alpha_keys, num_rendered*sizeof(uint64_t))

    combineTileAlpha<< <(num_rendered + 255) / 256, 256 >> >(
        num_rendered, 
        alpha_list, 
        binningState.point_list_keys, 
        tile_alpha_keys)CHECK_CUDA(, debug)

    uint64_t *tile_alpha_sorted;
    uint32_t *point_list_alpha_sorted;
    CudaMalloc(tile_alpha_sorted, num_rendered*sizeof(uint64_t))
    CudaMalloc(point_list_alpha_sorted, num_rendered*sizeof(uint32_t))


    size_t sorting_size = 0;
    void* tmp_sorting_space = nullptr;

    cub::DeviceRadixSort::SortPairs(
        tmp_sorting_space, sorting_size,
        tile_alpha_keys, tile_alpha_sorted,
        binningState.point_list, point_list_alpha_sorted,
        num_rendered)CHECK_CUDA(, debug)

    CudaMalloc(tmp_sorting_space, sorting_size);

    // Sort complete list of (duplicated) Gaussian indices by keys
    cub::DeviceRadixSort::SortPairs(
        tmp_sorting_space, sorting_size,
        tile_alpha_keys, tile_alpha_sorted,
        binningState.point_list, point_list_alpha_sorted, 
        num_rendered)CHECK_CUDA(, debug)

    cudaFree(tmp_sorting_space);

    uint2 *tile_alpha_ranges;
    CudaMalloc(tile_alpha_ranges, (width+w-1)*(height+h-1)/(w*h) * sizeof(uint2))

    if (num_rendered > 0)
        myIdentifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            tile_alpha_sorted,
            tile_alpha_ranges)CHECK_CUDA(, debug);
    
    topKAlphaIdx << < T, K >> >(
        tiles, K, 
        tile_alpha_ranges, 
        tile_alpha_sorted, 
        point_list_alpha_sorted, 
        out_idxs,out_alphas)CHECK_CUDA(, debug)

    cudaFree(alpha_list);
    cudaFree(tile_alpha_keys);
    cudaFree(tile_alpha_sorted);
    cudaFree(point_list_alpha_sorted);
    cudaFree(tile_alpha_ranges);
    return 0;
}