#ifndef RENDERVIEW_H
#define RENDERVIEW_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QTime>
#include <QtMath>
#include <QKeyEvent>
#include <QPainter>
#include <QWheelEvent>

#include <string>
#include <ctime>
#include <thread>
#include <omp.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iomanip>

#include "CudaTexture.h"
#include "Model.h"
#include "ShapeSphere.h"
#include "PanoStitch.cuh"
#include "NvCodecUtils.h"
#include "CircleQueue.h"

class RenderView : public QOpenGLWidget, protected QOpenGLFunctions {
Q_OBJECT
public:
    explicit RenderView(QWidget *parent = nullptr);

    ~RenderView();

protected:
    void initializeGL() override;

    void paintGL() override;

    void resizeGL(int w, int h) override;

    void timerEvent(QTimerEvent *) override;

    void keyPressEvent(QKeyEvent *event) override;

    void keyReleaseEvent(QKeyEvent *event) override;

    void mousePressEvent(QMouseEvent *event) override;

    void mouseMoveEvent(QMouseEvent *event) override;

    void wheelEvent(QWheelEvent *event) override;

    void checkKey();

    void calcFPS();

    void updateFPS(qreal);

private:
    void initViewer();

    void initSphere();

    void initRect();

    void addModel();

    void recoveryViewer();

    static void threadImageCapture(const std::string &rootPath, int camId, int max_frame,
                                   const std::string &color_ext, const std::string &depth_ext,
                                   CircleQueue &imgQueue);

    void initSkyBox();

    void initPointCloud();

private:
    // 交互相关矩阵向量
    QVector3D cameraPos, cameraFront, cameraUp;
    QMatrix4x4 modelMat, viewMat, projectMat;
    // 键盘输入
    bool keys[1024];
    // 放缩系数
    qreal aspect;
    // 鼠标上一个位置
    QPoint last;
    // 偏航角 俯仰角
    qreal yaw, pitch;
    // 帧率间隔
    qreal elapsed;
    // 帧率
    qreal fps;
    // 时间对象
    QTime m_time;

    // 自定义着色器编译对象
    QOpenGLShaderProgram *bowlProgram;
    QOpenGLShaderProgram *rectProgram;
    QOpenGLShaderProgram *modelProgram;
    QOpenGLShaderProgram *sphereProgram;
    QOpenGLShaderProgram *skyboxProgram;
    QOpenGLShaderProgram *pointProgram;
    // 碗状模型
    unsigned int bowlVAO;
    unsigned int bowlVBO;
    unsigned int bowlEBO;
    uint bowlIndexBuffer;
    // 球模型
    ShapeSphere::Ptr pShapeSphere;
    unsigned int textureId;
    // 矩形阴影
    unsigned int rectVAO;
    unsigned int rectVBO;
    unsigned int rectEBO;
    // 汽车模型
    Model *carModel;

public:
    // 保存彩色与深度图
    CircleQueue imageQue[8];
    std::string colorExt, depthExt;
    std::string dataPath;
    std::string paraPath;
    int camNum, maxFrame;
    std::vector<NvThread> capThreadVec;
    static bool isStop, dataIsOk;

    // 存储输入的4张rgb图像
    uchar3 *d_rgb4;
    // 存储输入的4张深度图
    InputDepthType *d_depth4;
    // 待输出的全景图
    uchar3 *d_pano_rgb, *pano_rgb_dilate;
    // 全景深度图（中间输出，这里的深度是相对于虚拟球心的距离）
    PanoDepthType *d_pano_depth;
    std::vector<Eigen::Matrix4d> T_cameraNToWorldVec;
    // 世界到1号相机的T
    Eigen::Matrix4d T_worldToCamera1;
    // 虚拟球心在1号相机坐标系下的坐标
    Eigen::Vector3d sphere_center_c1, last_center_c1, recovery_center_c1;
    // 虚拟球心到1号相机 T
    Eigen::Matrix4d T_sphereToCamera1;
    // 1号相机转虚拟球心 T
    Eigen::Matrix4d T_camera1ToSphere;
    // 4路相机所在坐标系到虚拟球心坐标系的变换矩阵T
    double *d_T_cameraNToSphere4;
    bool circleViewMode = false;
    float circleViewTheta = 90.f;
    double fov;
    double *intrinsic_cam4;

    // 当前帧id
    int frameId = 0;
    // 显示图像
    cv::Mat panoImage;

signals:

    void outputTimer(std::string duration);

    void changeImg();
};

#endif // RENDERVIEW_H
