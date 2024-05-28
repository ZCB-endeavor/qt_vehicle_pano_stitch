#include <QApplication>
#include "widget.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Widget w;
    w.show();
    QFile styleFile("../resources/darkstyle.qss");
    if (styleFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Open Success!" << endl;
        QString setStyleSheet(styleFile.readAll());
        a.setStyleSheet(setStyleSheet);
        styleFile.close();
    } else {
        qDebug() << "Open Failed!" << endl;
    }
    return a.exec();
}