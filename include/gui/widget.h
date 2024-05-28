#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QFileDialog>
#include <QPushButton>
#include <QKeyEvent>
#include <QDebug>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QMessageBox>
#include <QScreen>
#include "renderview.h"

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget {
Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);

    ~Widget();

private:
    Ui::Widget *ui;

public slots:

    void onDataPathBtnClick();

    void onParamBtnClick();

    void onMaxFrameChanged(const QString &arg);

    void onColorExtChanged(const QString &arg);

    void onDepthExtChanged(const QString &arg);

    void onCamNumChanged(const QString &arg);

    void onStartBtnClick();

    void onStopBtnClick();

    void getDuration(std::string duration);

    void disImage();

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

    void keyPressEvent(QKeyEvent *event) override;

public:
    RenderView *renderView;
    QHBoxLayout *hBoxLayout;
    QString colorExt, depthExt;
    QString camNum;
    QString dataPath;
    QString paraPath;
    QString maxFrame;
    bool isRunning, codecRunning;
};

#endif // WIDGET_H
