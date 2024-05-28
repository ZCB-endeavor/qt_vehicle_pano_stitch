#ifndef PANOSTITCH_CUH
#define PANOSTITCH_CUH

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <cuda.h>

#include <sys/time.h>
#include <time.h>

#define PI 3.141592654
#define PANO_W 3840
#define PANO_H 2160

#define RGB_W 1344
#define RGB_H 1344
#define img_depth_scale (float)1000
#define eps 1e-8

using InputDepthType = uint16_t; // 输入的那4张深度图的数据类型，因为是tiff，所以是double。如果是16位png，那么就是uint16_t。
using PanoDepthType = double;  // 中间全景深度图的数据类型

/* get thread id: 2D grid and 1D block */
#define get_tid() ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x)

extern "C" void init_pano_depth(PanoDepthType *d_pano_depth);

extern "C" void init_pano_color(uchar3 *pano_rgb);

extern "C" void forward_project_fish(uchar3 *d_rgb4, InputDepthType *d_depth4, double *d_T_camera_2virt4,
                                     PanoDepthType *d_pano_depth, int camNum, double fov, uchar3 *pano_rgb);

extern "C" void dilateProcess(uchar3 *src, uchar3 *dst, int kernelWidth, int kernelHeight, int imgWidth, int imgHeight);

extern "C" void erodeProcess(uchar3 *src, uchar3 *dst, int kernelWidth, int kernelHeight, int imgWidth, int imgHeight);

extern "C" void averageProcess(uchar3 *src, uchar3 *dst, PanoDepthType *d_pano_depth,
                               int kernelWidth, int kernelHeight, int imgWidth, int imgHeight);

#endif //PANOSTITCH_CUH