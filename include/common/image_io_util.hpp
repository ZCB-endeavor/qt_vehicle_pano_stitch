/*
* Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#ifndef IMAGE_UTIL_HPP_
#define IMAGE_UTIL_HPP_

#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

using namespace std;
using namespace cv;

bool showImageRgb(int width, int height, unsigned char* imgRgb);

std::unique_ptr<unsigned char[]>  getRgbImage(const string& imagePath, int& imgWidth, int& imgHeight)
{
    Mat inImgMat = imread(imagePath, IMREAD_COLOR);

    if (!inImgMat.data)
    {
        return nullptr;
    }

    Mat outImgRgb;
    cvtColor(inImgMat, outImgRgb, COLOR_BGR2RGB);

    if (outImgRgb.data == nullptr)
    {
        return nullptr;
    }

    auto retval = std::make_unique<unsigned char[]>(outImgRgb.rows * outImgRgb.cols * 3);
    memcpy(retval.get(), outImgRgb.data, outImgRgb.rows * outImgRgb.cols * 3);
    imgWidth = outImgRgb.cols;
    imgHeight = outImgRgb.rows;

    return retval;
}

std::unique_ptr<unsigned char[]> getRgbaImage(const string& imagePath, int& imgWidth, int& imgHeight)
{
    Mat inImgMat = imread(imagePath, IMREAD_COLOR);
    if (!inImgMat.data)
    {
        return nullptr;
    }

    Mat outImgRgb;
    cvtColor(inImgMat, outImgRgb, COLOR_BGRA2RGBA);
    if (outImgRgb.data == nullptr)
    {
        return nullptr;
    }

    auto retval = std::make_unique<unsigned char[]>(outImgRgb.rows * outImgRgb.cols * 4);
    memcpy(retval.get(), outImgRgb.data, outImgRgb.rows * outImgRgb.cols * 4);
    imgWidth = outImgRgb.cols;
    imgHeight = outImgRgb.rows;

    return retval;
}

bool putRgbaImage(const string& imagePath, unsigned char* imgRgba, int imgWidth, int imgHeight)
{
    Mat img = cv::Mat(imgHeight, imgWidth, CV_8UC4, (void*)imgRgba);

    Mat imgBgra;
    cvtColor(img, imgBgra, COLOR_RGBA2BGRA);

    imwrite(imagePath, imgBgra);

    return true;
}

bool showImageRgb(int width, int height, unsigned char* imgRgb)
{
    Mat img = cv::Mat(height, width, CV_8UC3, (void*)imgRgb);

    Mat imgBgr;
    cvtColor(img, imgBgr, COLOR_RGB2BGR);

    imshow("Imshow", imgBgr);
    waitKey();

    return true;
}

// Use OpenCV to check if image path is valid and get image width. height
bool getImageWidthHeightFromFilePath(const std::string& imgFilePath, int& width, int& height)
{
    cv::Mat img = cv::imread(imgFilePath);

    if (img.size().width == 0 || img.size().height == 0)
    {
        std::cout << "Invalid image:" << imgFilePath;
        return false;
    }
    width = img.size().width;
    height = img.size().height;
    return true;
}

#endif