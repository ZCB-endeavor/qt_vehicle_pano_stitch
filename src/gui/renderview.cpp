#include "renderview.h"

bool RenderView::isStop = false;

bool RenderView::dataIsOk = false;

RenderView::RenderView(QWidget *parent) : QOpenGLWidget(parent) {
    startTimer(1);
    qsrand(time(0));
    m_time.start();
    fps = 60;
    elapsed = 1;
    yaw = -90;
    pitch = -39;
    aspect = 75;
    fov = 220;
}

RenderView::~RenderView() {
    delete rectProgram;
    rectProgram = nullptr;
    delete modelProgram;
    modelProgram = nullptr;
    delete sphereProgram;
    sphereProgram = nullptr;
    delete carModel;
    carModel = nullptr;
}

void RenderView::initializeGL() {
    setFocusPolicy(Qt::StrongFocus);
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    cameraPos = QVector3D(0.035f, 5.088f, 0.598f);
    QVector3D front;
    front.setX(cos(qDegreesToRadians(pitch)) * cos(qDegreesToRadians(yaw)));
    front.setY(sin(qDegreesToRadians(pitch)));
    front.setZ(cos(qDegreesToRadians(pitch)) * sin(qDegreesToRadians(yaw)));
    cameraFront = front.normalized();
    cameraUp = QVector3D(0.0f, 1.0f, 0.0f);

    for (auto &i: keys) {
        i = false;
    }
    modelMat.setToIdentity();
    viewMat.setToIdentity();
    projectMat.setToIdentity();

    initViewer();
    initSphere();
    initRect();
    addModel();
    for (int i = 0; i < camNum; i++) {
        capThreadVec.push_back(NvThread(std::thread(threadImageCapture,
                                                    dataPath,
                                                    i,
                                                    maxFrame,
                                                    colorExt,
                                                    depthExt,
                                                    std::ref(imageQue[i]))));
    }
}

void RenderView::paintGL() {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    checkKey();

//    qDebug() << "===============================" << endl;
//    qDebug() << "cameraPos: " << cameraPos << endl;
//    qDebug() << "cameraFront: " << cameraFront << endl;
//    qDebug() << "cameraUp: " << cameraUp << endl;
//    qDebug() << "yaw: " << yaw << endl;
//    qDebug() << "pitch: " << pitch << endl;
//    qDebug() << "aspect: " << aspect << endl;
//    qDebug() << "===============================" << endl;

    //! surrounding texture
    auto lastTick = std::chrono::high_resolution_clock::now();

    if (!isStop) {
        for (int i = 0; i < camNum; i++) {
            if (imageQue[i].isQueueEmpty()) {
                dataIsOk = false;
                break;
            }
            dataIsOk = true;
        }
        //! 队列中有数据
        if (dataIsOk) {
            FrameData imgFrame[camNum];
            //! 为拼接模块提供数据
            for (int i = 0; i < camNum; i++) {
                imageQue[i].Get(imgFrame[i]);
                cudaMemcpy(d_rgb4 + i * RGB_W * RGB_H, imgFrame[i].color_data.data(),
                           RGB_W * RGB_H * sizeof(uchar3),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(d_depth4 + i * RGB_W * RGB_H, imgFrame[i].depth_data.data(),
                           RGB_W * RGB_H * sizeof(InputDepthType),
                           cudaMemcpyHostToDevice);

                cudaDeviceSynchronize();
            }
        }
    }

    //! 数据准备完毕就可以初始化数组元素为0
    init_pano_depth(d_pano_depth);
    init_pano_color(d_pano_rgb);
    init_pano_color(pano_rgb_dilate);

    //! 直接正向投影，生成全景图，进行均值滤波填补空洞
    forward_project_fish(d_rgb4, d_depth4, d_T_cameraNToSphere4, d_pano_depth, camNum, fov, d_pano_rgb);
    averageProcess(d_pano_rgb, pano_rgb_dilate, d_pano_depth, 3, 3, PANO_W, PANO_H);

    //! 显示全景图
    cv::Mat pano_rgb(PANO_H, PANO_W, CV_8UC3);
    cudaMemcpy(pano_rgb.data, pano_rgb_dilate, PANO_W * PANO_H * sizeof(uchar3), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    pano_rgb.copyTo(panoImage);
    //! 触发信号
    emit changeImg();

    // 球状模型
    sphereProgram->bind();
    viewMat.setToIdentity();
    if (!circleViewMode) {
        viewMat.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    } else {
        // 控制相机位置
        float circleViewR = 5.0f;
        float circleViewH = 3.5f;
        // 相机绕该点进行圆周运动
        QVector3D circleViewCenter(0.0f, 3.5f, 0.0f);
        circleViewTheta += 0.5f;

        float circleViewX = circleViewR * cos(qDegreesToRadians(circleViewTheta)) + circleViewCenter[0];
        float circleViewZ = circleViewR * sin(qDegreesToRadians(circleViewTheta)) + circleViewCenter[2];
        float circleViewY = 0.0f + circleViewH;

        cameraPos = QVector3D(circleViewX, circleViewY, circleViewZ);
        viewMat.lookAt(cameraPos, circleViewCenter, cameraUp);
    }

    projectMat.setToIdentity();
    projectMat.perspective(aspect, width() / height(), 0.1f, 100.0f);
    modelMat.setToIdentity();

    sphereProgram->setUniformValue("projection", projectMat);
    sphereProgram->setUniformValue("view", viewMat);
    sphereProgram->setUniformValue("model", modelMat);

    //! 球状模型纹理显示
    cv::Mat texImg;
    cv::cvtColor(pano_rgb, texImg, cv::COLOR_BGR2RGBA);
    cv::flip(texImg, texImg, 1);
    unsigned char *data = texImg.data;
    int width = texImg.cols;
    int height = texImg.rows;
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    } else {
        std::cout << "Failed to load texture" << std::endl;
    }
    texImg.release();
    // 激活纹理
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureId);

    // 根据预先生成的顶点缓存绘制球模型
    glBindVertexArray(this->pShapeSphere->VAO);
    glBindBuffer(GL_ARRAY_BUFFER, this->pShapeSphere->VBO[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, this->pShapeSphere->VBO[1]);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, this->pShapeSphere->VBO[2]);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->pShapeSphere->EBO);
    glDrawElements(GL_TRIANGLES, this->pShapeSphere->m_nIndices, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // 计时器
    const auto now = std::chrono::high_resolution_clock::now();
    const auto dt = now - lastTick;
    lastTick = now;
    const int dtMs = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
    std::stringstream tmpDuration;
    if (frameId < maxFrame) {
        tmpDuration << "# frame ... " << frameId++ << " ... " << "dt = " << dtMs << " ms";
    } else {
        tmpDuration << "# frame ... " << maxFrame << " ... " << "dt = " << dtMs << " ms";
    }
    emit outputTimer(tmpDuration.str());
    //! end surrounding texture

    // 矩形阴影
    rectProgram->bind();
    modelMat.translate(QVector3D(0.f, 0.05f, 0.f));
    modelMat.scale(10.0f);
    rectProgram->setUniformValue("projection", projectMat);
    rectProgram->setUniformValue("view", viewMat);
    rectProgram->setUniformValue("model", modelMat);
    glBindVertexArray(this->rectVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    // 汽车模型
    QMatrix4x4 transform_car;
    transform_car.setToIdentity();
    transform_car.translate(QVector3D(0.f, 0.05f, 0.f));
    transform_car.rotate(-90.f, QVector3D(0.f, 0.f, 1.f));
    transform_car.rotate(-180.f, QVector3D(0.f, 1.f, 0.f));
    transform_car.rotate(-90.f, QVector3D(0.f, 0.f, 1.f));
    transform_car.rotate(-90.f, QVector3D(1.f, 0.f, 0.f));
    transform_car.scale(0.018f, 0.014f, 0.015f);

    modelProgram->bind();
    modelProgram->setUniformValue("projection", projectMat);
    modelProgram->setUniformValue("view", viewMat);
    modelProgram->setUniformValue("model", transform_car);
    carModel->Render(*modelProgram);

    calcFPS();
}

void RenderView::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    float ratio = (float) w / h;

    projectMat.setToIdentity();
    projectMat.perspective(aspect, ratio, 0.1f, 100.0f);
}

// 交互功能相关方法
void RenderView::timerEvent(QTimerEvent *) {
    update();
}

void RenderView::keyPressEvent(QKeyEvent *event) {
    if (0 <= event->key() && event->key() < (int) (sizeof(keys) / sizeof(keys[0]))) {
        keys[event->key()] = true;
    }
    if (event->key() == Qt::Key_F5) {
        circleViewMode = false;
        circleViewTheta = 90.f;

        modelMat.setToIdentity();
        viewMat.setToIdentity();
        projectMat.setToIdentity();

        elapsed = 1;
        yaw = -90;
        pitch = -39;
        aspect = 75;

        cameraPos = QVector3D(0.035f, 5.088f, 0.598f);
        QVector3D front;
        front.setX(cos(qDegreesToRadians(pitch)) * cos(qDegreesToRadians(yaw)));
        front.setY(sin(qDegreesToRadians(pitch)));
        front.setZ(cos(qDegreesToRadians(pitch)) * sin(qDegreesToRadians(yaw)));
        cameraFront = front.normalized();
        cameraUp = QVector3D(0.0f, 1.0f, 0.0f);

        recoveryViewer();

        update();
    }

    event->accept();
}

void RenderView::keyReleaseEvent(QKeyEvent *event) {
    if (0 <= event->key() && event->key() < (int) (sizeof(keys) / sizeof(keys[0]))) {
        keys[event->key()] = false;
    }
    event->accept();
}

void RenderView::mousePressEvent(QMouseEvent *event) {
    last = event->pos();
}

void RenderView::mouseMoveEvent(QMouseEvent *event) {
    QPointF diff = QPointF(0, 0);

    diff = event->pos() - last;
    last = event->pos();

    qreal sensitivity = 0.05;
    qreal xoffset = diff.x() * sensitivity;
    qreal yoffset = -diff.y() * sensitivity;
    yaw -= xoffset;
    pitch -= yoffset;

    if (pitch > 89.0f) {
        pitch = 89.0f;
    } else if (pitch < -89.0f) {
        pitch = -89.0f;
    }
    QVector3D front;
    front.setX(cos(qDegreesToRadians(pitch)) * cos(qDegreesToRadians(yaw)));
    front.setY(sin(qDegreesToRadians(pitch)));
    front.setZ(cos(qDegreesToRadians(pitch)) * sin(qDegreesToRadians(yaw)));
    cameraFront = front.normalized();

    event->accept();
}

void RenderView::wheelEvent(QWheelEvent *event) {
    int offset = event->angleDelta().y() < 0 ? -1 : 1;
    qreal speed = 10;
    if (1 <= aspect + offset * speed && aspect + offset * speed <= 45) {
        aspect = aspect + offset * speed;
    }
    event->accept();
}

void RenderView::checkKey() {
    GLfloat cameraSpeed = 0.05 * elapsed / 1000.0;
    if (keys[Qt::Key_A]) {
        cameraPos -= QVector3D::crossProduct(cameraFront, cameraUp).normalized() * cameraSpeed;
        update();
    }
    if (keys[Qt::Key_D]) {
        cameraPos += QVector3D::crossProduct(cameraFront, cameraUp).normalized() * cameraSpeed;
        update();
    }
    if (keys[Qt::Key_W]) {
        cameraPos += cameraSpeed * cameraFront;
        update();
    }
    if (keys[Qt::Key_S]) {
        cameraPos -= cameraSpeed * cameraFront;
        update();
    }
    if (keys[Qt::Key_C]) {
        circleViewMode = true;
    }
}

void RenderView::calcFPS() {
    static QTime time;
    static int once = [=]() {
        time.start();
        return 0;
    }();
    Q_UNUSED(once)
    static int frame = 0;
    if (frame++ > 100) {
        elapsed = time.elapsed();
        updateFPS(frame / elapsed * 1000);
        time.restart();
        frame = 0;
    }
}

void RenderView::updateFPS(qreal v) {
    fps = v;
}

void RenderView::initViewer() {
    //! 存储各相机从相机坐标到世界坐标的T
    T_cameraNToWorldVec.resize(camNum);
    sphere_center_c1 = Eigen::Vector3d::Zero();
    //! 初始化显存空间
    cudaMalloc((void **)&d_rgb4, camNum * RGB_W * RGB_H * sizeof(uchar3));
    cudaMalloc((void **)&d_depth4, camNum * RGB_W * RGB_H * sizeof(InputDepthType));

    cudaMalloc((void **) &d_pano_rgb, PANO_W * PANO_H * sizeof(uchar3));
    cudaMalloc((void **) &pano_rgb_dilate, PANO_W * PANO_H * sizeof(uchar3));
    cudaMalloc((void **) &d_pano_depth, PANO_W * PANO_H * sizeof(PanoDepthType));

    cudaMalloc((void **) &d_T_cameraNToSphere4, camNum * 16 * sizeof(double));

    cudaMalloc((void **) &intrinsic_cam4, camNum * 9 * sizeof(double));

    //! 世界到1号相机的T
    for (int i = 0; i < camNum; ++i) {
        std::stringstream ss;
        ss << paraPath << i << ".xml";

        Eigen::Matrix4d T_temp;
        cv::Mat T_cv_temp, I_cv_temp;
        cv::FileStorage xml(ss.str(), cv::FileStorage::READ);

        xml["RT"] >> T_cv_temp;
            T_cv_temp = T_cv_temp.inv();
            cv::cv2eigen(T_cv_temp, T_temp);
            T_cameraNToWorldVec[i] = T_temp;
            xml.release();

        // 记录1号相机信息
        if (!i) {
            Eigen::Matrix4d T_temp_inv;
            T_temp_inv = T_temp.inverse();
            T_worldToCamera1 = T_temp_inv;
        }

        // 其余相机转到1号相机坐标系
        Eigen::Vector4d camera_c1 = T_worldToCamera1 * T_cameraNToWorldVec[i] * Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    }
    last_center_c1 = sphere_center_c1;
    // 用于F5复原视角
    recovery_center_c1 = sphere_center_c1;

    /****************************************
     * 虚拟球心坐标系和1号相机坐标系的转换：
     * 虚拟球心坐标系（左手系） ：x-前，y-右，z-上
     * 1号相机的坐标系（右手系）：x-右，y-下，z-前
    *****************************************/
    // 虚拟球心到1号相机 T
    T_sphereToCamera1 << 0.0, 1.0, 0.0, sphere_center_c1[0],
            0.0, 0.0, -1.0, sphere_center_c1[1],
            1.0, 0.0, 0.0, sphere_center_c1[2],
            0.0, 0.0, 0.0, 1.0;
    // 1号相机到虚拟球心 T
    T_camera1ToSphere = T_sphereToCamera1.inverse();

    for (int i = 0; i < camNum; ++i) {
        // 各相机从相机坐标系到虚拟球心坐标系的T
        Eigen::Matrix4d T_cameraNToSphere;
        T_cameraNToSphere = T_camera1ToSphere * T_worldToCamera1 * T_cameraNToWorldVec[i];
        cv::Mat T_cameraNToSphere_Mat;
        cv::eigen2cv(T_cameraNToSphere, T_cameraNToSphere_Mat);
        // 存储各相机从相机坐标系到虚拟球心坐标系的T
        cudaMemcpy(d_T_cameraNToSphere4 + i * 16, T_cameraNToSphere_Mat.data, 16 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
}

void RenderView::initSphere() {
    pShapeSphere = std::make_shared<ShapeSphere>();

    // 创建纹理缓存
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 创建缓冲对象
    glGenBuffers(3, this->pShapeSphere->VBO);
    glBindBuffer(GL_ARRAY_BUFFER, this->pShapeSphere->VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * this->pShapeSphere->vertices.size(),
                 this->pShapeSphere->vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, this->pShapeSphere->VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * this->pShapeSphere->normals.size(),
                 this->pShapeSphere->normals.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, this->pShapeSphere->VBO[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * this->pShapeSphere->texCoords.size(),
                 this->pShapeSphere->texCoords.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &this->pShapeSphere->EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->pShapeSphere->EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * this->pShapeSphere->indices.size(),
                 this->pShapeSphere->indices.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &this->pShapeSphere->VAO);
    glBindVertexArray(this->pShapeSphere->VAO);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    sphereProgram = new QOpenGLShaderProgram;
    if (!sphereProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shaders/sphere_vert.glsl")) {
        qDebug() << __FILE__ << __FUNCTION__ << " add vertex shader file failed.";
        return;
    }
    if (!sphereProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shaders/sphere_frag.glsl")) {
        qDebug() << __FILE__ << __FUNCTION__ << " add fragment shader file failed.";
        return;
    }
    sphereProgram->link();
}

void RenderView::initRect() {
    const float rectvert[] = {
            0.4f, 0.0f, 0.525f,
            -0.4f, 0.0f, 0.525f,
            -0.4f, 0.0f, -0.525f,

            0.4f, 0.0f, 0.525f,
            -0.4f, 0.0f, -0.525f,
            0.4f, 0.0f, -0.525f
    };
    rectProgram = new QOpenGLShaderProgram;
    if (!rectProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shaders/blackrectshadervert.glsl")) {
        qDebug() << __FILE__ << __FUNCTION__ << " add vertex shader file failed.";
        return;
    }
    if (!rectProgram->addShaderFromSourceFile(QOpenGLShader::Fragment,
                                              "../shaders/blackrectshaderfrag.glsl")) {
        qDebug() << __FILE__ << __FUNCTION__ << " add fragment shader file failed.";
        return;
    }
    rectProgram->link();

    glGenVertexArrays(1, &this->rectVAO);
    glGenBuffers(1, &this->rectVBO);
    glGenBuffers(1, &this->rectEBO);

    glBindVertexArray(this->rectVAO);
    glBindBuffer(GL_ARRAY_BUFFER, this->rectVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rectvert), &rectvert, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
}

void RenderView::addModel() {
    modelProgram = new QOpenGLShaderProgram;
    if (!modelProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shaders/modelshadervert.glsl")) {
        qDebug() << __FILE__ << __FUNCTION__ << " add vertex shader file failed.";
        return;
    }
    if (!modelProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shaders/modelshaderfrag.glsl")) {
        qDebug() << __FILE__ << __FUNCTION__ << " add fragment shader file failed.";
        return;
    }
    modelProgram->link();
    carModel = new Model("../models/car/Dodge Challenger SRT Hellcat 2015.obj");
}

void RenderView::threadImageCapture(const std::string &rootPath, int camId, int max_frame,
                                    const std::string &color_ext, const std::string &depth_ext,
                                    CircleQueue &imgQueue) {
    int count = 0;
    while (!isStop) {
        //! 队列满后就停止Put数据, 直到队列不满
        if (imgQueue.isQueueFull()) {
            //! 防止while循环太快
            std::this_thread::sleep_for(1ms);
            continue;
        }
        std::stringstream ss1, ss2;
        ss1 << rootPath << count % max_frame << "/color/" << camId << color_ext;
        ss2 << rootPath << count % max_frame << "/depth/" << camId << depth_ext;

        cv::Mat color = cv::imread(ss1.str(), cv::IMREAD_UNCHANGED);
        cv::Mat depth = cv::imread(ss2.str(), cv::IMREAD_UNCHANGED);
        FrameData imgData;
        imgData.color_data.insert(imgData.color_data.begin(),
                                  (uchar *) color.data,
                                  (uchar *) color.data +
                                  (color.rows * color.cols * color.channels() * sizeof(uchar)));
        imgData.depth_data.insert(imgData.depth_data.begin(),
                                  (uint16_t *) depth.data,
                                  (uint16_t *) depth.data +
                                  (depth.rows * depth.cols * depth.channels() * sizeof(uint16_t)));

        imgQueue.Put(imgData);
        count++;
        if (count == max_frame) {
            break;
        }
    }
}

void RenderView::recoveryViewer() {
    // 复原生成全景图的虚拟视点视角
    T_sphereToCamera1 << 0.0, 1.0, 0.0, recovery_center_c1[0],
            0.0, 0.0, -1.0, recovery_center_c1[1],
            1.0, 0.0, 0.0, recovery_center_c1[2],
            0.0, 0.0, 0.0, 1.0;
    // 重置虚拟视点坐标
    sphere_center_c1 = recovery_center_c1;
    last_center_c1 = recovery_center_c1;

    T_camera1ToSphere = T_sphereToCamera1.inverse();
    for (int i = 0; i < camNum; ++i) {
        // 各相机从相机坐标系到虚拟球心坐标系的T
        Eigen::Matrix4d T_cameraNToSphere;
        T_cameraNToSphere = T_camera1ToSphere * T_worldToCamera1 * T_cameraNToWorldVec[i];
        cv::Mat T_cameraNToSphere_Mat;
        cv::eigen2cv(T_cameraNToSphere, T_cameraNToSphere_Mat);
        // 存储各相机从相机坐标系到虚拟球心坐标系的T
        cudaMemcpy(d_T_cameraNToSphere4 + i * 16, T_cameraNToSphere_Mat.data, 16 * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
}
