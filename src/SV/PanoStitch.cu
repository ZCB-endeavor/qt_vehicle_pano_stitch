#include "PanoStitch.cuh"

__global__ void init_pano_depth_kernel(PanoDepthType *pano_depth) {
    int tid = get_tid();

    if (tid < PANO_W * PANO_H)
        pano_depth[tid] = 0;
}

extern "C" void init_pano_depth(PanoDepthType *d_pano_depth) {
    int bs = 32;
    int sz = ceil(sqrt((PANO_W * PANO_H + bs - 1) / bs));
    dim3 grid = dim3(sz, sz);

    init_pano_depth_kernel<<<grid, bs>>>(d_pano_depth);
    cudaDeviceSynchronize(); // 等待gpu完成
}

__global__ void init_pano_color_kernel(uchar3 *pano_rgb) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < PANO_W && y < PANO_H) {
        pano_rgb[y * PANO_W + x].x = 200;
        pano_rgb[y * PANO_W + x].y = 200;
        pano_rgb[y * PANO_W + x].z = 200;
    }
}

extern "C" void init_pano_color(uchar3 *pano_rgb) {
    dim3 threads(32, 32);
    unsigned int blockX = (threads.x + PANO_W - 1) / threads.x;
    unsigned int blockY = (threads.y + PANO_H - 1) / threads.y;
    dim3 blocks(blockX, blockY);
    init_pano_color_kernel<<<blocks, threads>>>(pano_rgb);
    cudaDeviceSynchronize(); // 等待gpu完成
}

__global__ void forward_project_fish_kernel(uchar3 *rgb4, InputDepthType *depth4,
                                            double *T_camera_2virt4, int camNum, double fov,
                                            PanoDepthType *pano_depth, uchar3 *pano_rgb) {
    int tid = get_tid();

    if (tid < RGB_W * RGB_H) {

        int v = tid / RGB_W;
        int u = tid % RGB_W;

        double r_max = min(RGB_W / 2.0, RGB_H / 2.0);
        double theta_max = fov * PI / 180 / 2;
        double f = r_max / theta_max;

        double rd = sqrt((u - RGB_W / 2.0) * (u - RGB_W / 2.0) + (v - RGB_H / 2.0) * (v - RGB_H / 2.0));

        if (rd < r_max) {

            double theta = rd / f;
            for (int i = 0; i < camNum; ++i) {
                double pt_c_z = depth4[i * RGB_W * RGB_H + v * RGB_W + u] / img_depth_scale * cos(theta);
                double pt_c_x = depth4[i * RGB_W * RGB_H + v * RGB_W + u] / img_depth_scale * sin(theta) *
                                sin(atan2(u - RGB_W / 2.0, v - RGB_H / 2.0));
                double pt_c_y = depth4[i * RGB_W * RGB_H + v * RGB_W + u] / img_depth_scale * sin(theta) *
                                cos(atan2(u - RGB_W / 2.0, v - RGB_H / 2.0));

                double pt_virt_x = T_camera_2virt4[i * 16 + 0] * pt_c_x + T_camera_2virt4[i * 16 + 1] * pt_c_y +
                                   T_camera_2virt4[i * 16 + 2] * pt_c_z + T_camera_2virt4[i * 16 + 3];
                double pt_virt_y = T_camera_2virt4[i * 16 + 4] * pt_c_x + T_camera_2virt4[i * 16 + 5] * pt_c_y +
                                   T_camera_2virt4[i * 16 + 6] * pt_c_z + T_camera_2virt4[i * 16 + 7];
                double pt_virt_z = T_camera_2virt4[i * 16 + 8] * pt_c_x + T_camera_2virt4[i * 16 + 9] * pt_c_y +
                                   T_camera_2virt4[i * 16 + 10] * pt_c_z + T_camera_2virt4[i * 16 + 11];

                double alpha = atan2(pt_virt_y, pt_virt_x); // 旋转角
                double beta = asin(
                        pt_virt_z / sqrt(pt_virt_x * pt_virt_x + pt_virt_y * pt_virt_y + pt_virt_z * pt_virt_z)); // 俯仰角

                int u_pano = int((alpha / (2 * PI) + 1 / 2.0) * PANO_W);
                int v_pano = int((1 / 2.0 - beta / PI) * PANO_H);

                if (u_pano >= 0 && u_pano < PANO_W && v_pano >= 0 && v_pano < PANO_H) {
                    double cur_dist = sqrt(pt_virt_x * pt_virt_x + pt_virt_y * pt_virt_y + pt_virt_z * pt_virt_z);

                    if (pano_depth[v_pano * PANO_W + u_pano] < eps || pano_depth[v_pano * PANO_W + u_pano] > cur_dist) {
                        pano_depth[v_pano * PANO_W + u_pano] = cur_dist;
                        pano_rgb[v_pano * PANO_W + u_pano] = rgb4[i * RGB_W * RGB_H + tid];
                    }

                    if (u_pano >= 1 && u_pano <= PANO_W - 2 && v_pano >= 1 && v_pano <= PANO_H - 2) {
                        if (pano_depth[(v_pano - 1) * PANO_W + (u_pano)] < eps ||
                            cur_dist < pano_depth[(v_pano - 1) * PANO_W + (u_pano)]) {
                            pano_depth[(v_pano - 1) * PANO_W + (u_pano)] = cur_dist;
                            pano_rgb[(v_pano - 1) * PANO_W + (u_pano)] = rgb4[i * RGB_W * RGB_H + tid];
                        }
                        if (pano_depth[(v_pano) * PANO_W + (u_pano - 1)] < eps ||
                            cur_dist < pano_depth[(v_pano) * PANO_W + (u_pano - 1)]) {
                            pano_depth[(v_pano) * PANO_W + (u_pano - 1)] = cur_dist;
                            pano_rgb[(v_pano) * PANO_W + (u_pano - 1)] = rgb4[i * RGB_W * RGB_H + tid];
                        }
                        if (pano_depth[(v_pano) * PANO_W + (u_pano + 1)] < eps ||
                            cur_dist < pano_depth[(v_pano) * PANO_W + (u_pano + 1)]) {
                            pano_depth[(v_pano) * PANO_W + (u_pano + 1)] = cur_dist;
                            pano_rgb[(v_pano) * PANO_W + (u_pano + 1)] = rgb4[i * RGB_W * RGB_H + tid];
                        }
                        if (pano_depth[(v_pano + 1) * PANO_W + (u_pano)] < eps ||
                            cur_dist < pano_depth[(v_pano + 1) * PANO_W + (u_pano)]) {
                            pano_depth[(v_pano + 1) * PANO_W + (u_pano)] = cur_dist;
                            pano_rgb[(v_pano + 1) * PANO_W + (u_pano)] = rgb4[i * RGB_W * RGB_H + tid];
                        }
                    }
                }
            }
        }
    }
}

extern "C" void forward_project_fish(uchar3 *d_rgb4, InputDepthType *d_depth4,
                                     double *d_T_camera_2virt4, PanoDepthType *d_pano_depth,
                                     int camNum, double fov, uchar3 *pano_rgb) {
    int bs = 32;
    int sz = ceil(sqrt((RGB_W * RGB_H + bs - 1) / bs));
    dim3 grid = dim3(sz, sz);
    forward_project_fish_kernel<<<grid, bs>>>(d_rgb4, d_depth4, d_T_camera_2virt4, camNum, fov, d_pano_depth, pano_rgb);
    cudaDeviceSynchronize(); // 等待gpu完成
}

// 膨胀处理
__global__ void dilate_kernel(uchar3 *src, uchar3 *dst,
                              int kernelWidth, int kernelHeight,
                              int imgWidth, int imgHeight) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_kernel_width = kernelWidth / 2;
    int half_kernel_height = kernelHeight / 2;

    dst[y * imgWidth + x] = src[y * imgWidth + x];

    int r = dst[y * imgWidth + x].x;
    int g = dst[y * imgWidth + x].y;
    int b = dst[y * imgWidth + x].z;

    if (x >= half_kernel_width && x < imgWidth - half_kernel_width &&
        y >= half_kernel_height && y < imgHeight - half_kernel_height &&
        r == 0 && g == 0 && b == 0) {
        for (int v = -half_kernel_height; v < half_kernel_height + 1; v++) {
            for (int u = -half_kernel_width; u < half_kernel_width + 1; u++) {
                int src_x = src[(y + v) * imgWidth + (x + u)].x;
                int src_y = src[(y + v) * imgWidth + (x + u)].y;
                int src_z = src[(y + v) * imgWidth + (x + u)].z;

                int dst_x = dst[y * imgWidth + x].x;
                int dst_y = dst[y * imgWidth + x].y;
                int dst_z = dst[y * imgWidth + x].z;

                if (src_x > dst_x && src_y > dst_y && src_z > dst_z) {
                    dst[y * imgWidth + x] = src[(y + v) * imgWidth + (x + u)];
                }
            }
        }
    }
}

extern "C"
void dilateProcess(uchar3 *src, uchar3 *dst, int kernelWidth, int kernelHeight, int imgWidth, int imgHeight) {
    dim3 threads(32, 32);
    unsigned int blockX = (threads.x + PANO_W - 1) / threads.x;
    unsigned int blockY = (threads.y + PANO_H - 1) / threads.y;
    dim3 blocks(blockX, blockY);
    dilate_kernel<<<blocks, threads>>>(src, dst, kernelWidth, kernelHeight, imgWidth, imgHeight);
    cudaDeviceSynchronize(); // 等待gpu完成
}

// 腐蚀处理
__global__ void erode_kernel(uchar3 *src, uchar3 *dst,
                             int kernelWidth, int kernelHeight,
                             int imgWidth, int imgHeight) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_kernel_width = kernelWidth / 2;
    int half_kernel_height = kernelHeight / 2;

    dst[y * imgWidth + x] = src[y * imgWidth + x];

    if (x >= half_kernel_width && x < imgWidth - half_kernel_width &&
        y >= half_kernel_height && y < imgHeight - half_kernel_height) {
        for (int v = -half_kernel_height; v < half_kernel_height + 1; v++) {
            for (int u = -half_kernel_width; u < half_kernel_width + 1; u++) {
                int src_x = src[(y + v) * imgWidth + (x + u)].x;
                int src_y = src[(y + v) * imgWidth + (x + u)].y;
                int src_z = src[(y + v) * imgWidth + (x + u)].z;

                int dst_x = dst[y * imgWidth + x].x;
                int dst_y = dst[y * imgWidth + x].y;
                int dst_z = dst[y * imgWidth + x].z;

                if (src_x < dst_x && src_y < dst_y && src_z < dst_z) {
                    dst[y * imgWidth + x] = src[(y + v) * imgWidth + (x + u)];
                }
            }
        }
    }
}

extern "C"
void erodeProcess(uchar3 *src, uchar3 *dst, int kernelWidth, int kernelHeight, int imgWidth, int imgHeight) {
    dim3 threads(32, 32);
    unsigned int blockX = (threads.x + PANO_W - 1) / threads.x;
    unsigned int blockY = (threads.y + PANO_H - 1) / threads.y;
    dim3 blocks(blockX, blockY);
    erode_kernel<<<blocks, threads>>>(src, dst, kernelWidth, kernelHeight, imgWidth, imgHeight);
    cudaDeviceSynchronize(); // 等待gpu完成
}

// 均值处理
__global__ void average_kernel(uchar3 *src, uchar3 *dst, const PanoDepthType *d_pano_depth,
                               int kernelWidth, int kernelHeight, int imgWidth, int imgHeight) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_kernel_width = kernelWidth / 2;
    int half_kernel_height = kernelHeight / 2;

    dst[y * imgWidth + x] = src[y * imgWidth + x];

    if (x >= half_kernel_width && x < imgWidth - half_kernel_width &&
        y >= half_kernel_height && y < imgHeight - half_kernel_height &&
        d_pano_depth[y * imgWidth + x] == 0) {
        int mean_x = 0;
        int mean_y = 0;
        int mean_z = 0;
        int count = 0;
        for (int v = -half_kernel_height; v < half_kernel_height + 1; v++) {
            for (int u = -half_kernel_width; u < half_kernel_width + 1; u++) {
                int src_x = (int) src[(y + v) * imgWidth + (x + u)].x;
                int src_y = (int) src[(y + v) * imgWidth + (x + u)].y;
                int src_z = (int) src[(y + v) * imgWidth + (x + u)].z;
                if (src_x != 0 || src_y != 0 || src_z != 0) {
                    mean_x += src_x;
                    mean_y += src_y;
                    mean_z += src_z;
                    count++;
                }
            }
        }
        if (count != 0) {
            mean_x = mean_x / count;
            mean_y = mean_y / count;
            mean_z = mean_z / count;
            if (mean_x != 0 || mean_y != 0 || mean_z != 0) {
                dst[y * imgWidth + x] = make_uchar3(mean_x, mean_y, mean_z);
            }
        }
    }
}

extern "C"
void averageProcess(uchar3 *src, uchar3 *dst, PanoDepthType *d_pano_depth,
                    int kernelWidth, int kernelHeight, int imgWidth, int imgHeight) {
    dim3 threads(32, 32);
    unsigned int blockX = (threads.x + PANO_W - 1) / threads.x;
    unsigned int blockY = (threads.y + PANO_H - 1) / threads.y;
    dim3 blocks(blockX, blockY);
    average_kernel<<<blocks, threads>>>(src, dst, d_pano_depth, kernelWidth, kernelHeight, imgWidth, imgHeight);
    cudaDeviceSynchronize(); // 等待gpu完成
}