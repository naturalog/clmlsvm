#-------------------------------------------------
#
# Project created by QtCreator 2014-02-07T04:31:55
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

TARGET = clmlcl-qt
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    matrixdialog.cpp \
    qcustomplot.cpp \
    classifier.cpp

HEADERS  += mainwindow.h \
    defs.h \
    matrixdialog.h \
    qcustomplot.h \
    classifier.h

FORMS    += mainwindow.ui \
    matrixdialog.ui

CONFIG += c++11

LIBS += -lOpenCL -lboost_system

OTHER_FILES += \
    cls.cl
