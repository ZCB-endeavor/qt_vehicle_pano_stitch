#include "widget.h"
#include "./ui_widget.h"

Widget::Widget(QWidget *parent)
        : QWidget(parent), ui(new Ui::Widget) {
    ui->setupUi(this);
    this->setWindowIcon(QIcon("../resources/logo.jpg"));

    //! 图像输入模块事件绑定
    connect(ui->dataPathBtn, &QPushButton::clicked, this, &Widget::onDataPathBtnClick);
    connect(ui->paraBtn, &QPushButton::clicked, this, &Widget::onParamBtnClick);
    connect(ui->maxFrame, &QLineEdit::textChanged, this, &Widget::onMaxFrameChanged);
    connect(ui->colorExt, &QComboBox::currentTextChanged, this, &Widget::onColorExtChanged);
    connect(ui->depthExt, &QComboBox::currentTextChanged, this, &Widget::onDepthExtChanged);
    connect(ui->camNum, &QComboBox::currentTextChanged, this, &Widget::onCamNumChanged);
    connect(ui->startBtn, &QPushButton::clicked, this, &Widget::onStartBtnClick);
    connect(ui->stopBtn, &QPushButton::clicked, this, &Widget::onStopBtnClick);

    //! 给label安装事件过滤器
    ui->stitchView->installEventFilter(this);

    this->colorExt = ui->colorExt->currentText();
    this->depthExt = ui->depthExt->currentText();
    this->camNum = ui->camNum->currentText();

    isRunning = false;
    codecRunning = false;
    hBoxLayout = new QHBoxLayout();

    ui->dataPath->setText("../datasets/images");
    this->dataPath = "../datasets/images";
    ui->paraPath->setText("../datasets/param");
    this->paraPath = "../datasets/param";
    ui->maxFrame->setText("55");
    this->maxFrame = "55";
}

Widget::~Widget() {
    if (codecRunning) {
        codecRunning = false;
    }
    delete ui;
}

void Widget::onDataPathBtnClick() {
    QString filePath = QFileDialog::getExistingDirectory(this, tr("选择数据所在路径"), "/home");
    if (filePath != "") {
        ui->dataPath->setText(filePath);
        this->dataPath = ui->dataPath->text();
        qDebug() << "dataPath: " << this->dataPath << endl;
    }
}

void Widget::onParamBtnClick() {
    QString filePath = QFileDialog::getExistingDirectory(this, tr("选择标定参数文件所在路径"), "/home");
    if (filePath != "") {
        ui->paraPath->setText(filePath);
        this->paraPath = ui->paraPath->text();
        qDebug() << "paraPath: " << this->paraPath << endl;
    }
}

void Widget::onMaxFrameChanged(const QString &arg) {
    qDebug() << "onMaxFrameChanged(): " << arg << endl;
    this->maxFrame = arg;
}

void Widget::onColorExtChanged(const QString &arg) {
    qDebug() << "onColorExtChanged(): " << arg << endl;
    this->colorExt = arg;
}

void Widget::onDepthExtChanged(const QString &arg) {
    qDebug() << "onDepthExtChanged(): " << arg << endl;
    this->depthExt = arg;
}

void Widget::onCamNumChanged(const QString &arg) {
    qDebug() << "onCamNumChanged(): " << arg << endl;
    this->camNum = arg;
}

void Widget::onStartBtnClick() {
    if (this->dataPath != "" && this->paraPath != "" && this->maxFrame != "") {
        //! 避免重复启动
        ui->dataPathBtn->setEnabled(false);
        ui->paraBtn->setEnabled(false);
        ui->maxFrame->setEnabled(false);
        ui->startBtn->setEnabled(false);
        ui->colorExt->setEnabled(false);
        ui->depthExt->setEnabled(false);
        ui->camNum->setEnabled(false);

        std::string info = "###### Start System ######";
        ui->cliResult->append(info.c_str());
        renderView = new RenderView(ui->renderView);
        RenderView::isStop = false;
        connect(renderView, &RenderView::outputTimer, this, &Widget::getDuration);
        connect(renderView, &RenderView::changeImg, this, &Widget::disImage);
        hBoxLayout->addWidget(renderView);
        ui->renderView->setLayout(hBoxLayout);

        //! 保证变量值为当前页面设置的值
        this->camNum = ui->camNum->currentText();
        renderView->camNum = this->camNum.toInt();
        this->maxFrame = ui->maxFrame->text();
        renderView->maxFrame = this->maxFrame.toInt();
        //! 保证路径结尾为'/'
        this->dataPath = this->dataPath.endsWith('/') ? this->dataPath : this->dataPath + "/";
        this->paraPath = this->paraPath.endsWith('/') ? this->paraPath : this->paraPath + "/";
        renderView->dataPath = this->dataPath.toStdString();
        renderView->paraPath = this->paraPath.toStdString();

        renderView->colorExt = this->colorExt.toStdString();
        renderView->depthExt = this->depthExt.toStdString();

        isRunning = true;
    }
}

void Widget::onStopBtnClick() {
    if (isRunning) {
        std::string info = "****** Stop System ******";
        ui->cliResult->append(info.c_str());

        disconnect(renderView, &RenderView::outputTimer, this, &Widget::getDuration);
        disconnect(renderView, &RenderView::changeImg, this, &Widget::disImage);

        RenderView::isStop = true;

        //! 清空环视系统
        ui->stitchView->clear();
        ui->stitchView->setText("全景拼接图");

        renderView->close();
        //! 释放显存空间
        cudaFree(renderView->d_rgb4);
        cudaFree(renderView->d_depth4);
        cudaFree(renderView->d_pano_rgb);
        cudaFree(renderView->pano_rgb_dilate);
        cudaFree(renderView->d_pano_depth);
        cudaFree(renderView->d_T_cameraNToSphere4);
        cudaFree(renderView->intrinsic_cam4);

        isRunning = false;

        ui->dataPathBtn->setEnabled(true);
        ui->paraBtn->setEnabled(true);
        ui->maxFrame->setEnabled(true);
        ui->startBtn->setEnabled(true);
        ui->colorExt->setEnabled(true);
        ui->depthExt->setEnabled(true);
        ui->camNum->setEnabled(true);

        delete renderView;
    }
}

void Widget::getDuration(std::string duration) {
    ui->cliResult->append(duration.c_str());
}

void Widget::disImage() {
    //! 展示全景图像
    cv::Mat mat = renderView->panoImage;
    cv::Mat rgb;
    QImage img;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        img = QImage((const uchar *) (rgb.data),
                     rgb.cols, rgb.rows,
                     rgb.cols * rgb.channels(),
                     QImage::Format_RGB888);
    } else {
        img = QImage((const uchar *) (mat.data),
                     mat.cols, mat.rows,
                     mat.cols * mat.channels(),
                     QImage::Format_Indexed8);
    }
    QPixmap pixmap = QPixmap::fromImage(img);
    if (ui->stitchView->isFullScreen()) {
        QScreen *screen = QApplication::primaryScreen();
        QSize size = screen->size();
        const QPixmap newPixmap = pixmap.scaled(size, Qt::KeepAspectRatio);
        ui->stitchView->setPixmap(newPixmap);
        ui->stitchView->setScaledContents(false);
    } else {
        ui->stitchView->setPixmap(pixmap);
        ui->stitchView->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        ui->stitchView->setScaledContents(true);
    }
    ui->stitchView->show();
}

bool Widget::eventFilter(QObject *watched, QEvent *event) {
    if (watched == ui->stitchView) {
        if (event->type() == QEvent::MouseButtonDblClick) {
            if (!ui->stitchView->isFullScreen()) {
                ui->stitchView->setWindowFlags(Qt::Dialog);
                ui->stitchView->showFullScreen();
            } else {
                ui->stitchView->setWindowFlags(Qt::SubWindow);
                ui->stitchView->showNormal();
            }
        }
        return QObject::eventFilter(watched, event);
    }
}

void Widget::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape) {
        this->close();
    }
}
