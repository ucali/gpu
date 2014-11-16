TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

TARGET = gpu_test

QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -msse -msse2 -msse3

SOURCES += main.cpp \
    cl.cpp

INCLUDEPATH += $$_PRO_FILE_PWD_/../
INCLUDEPATH += /opt/AMDAPPSDK-2.9-1/include
LIBS += -lOpenCL -lgtest -lgtest_main -lpthread
