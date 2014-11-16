TEMPLATE = lib
CONFIG += staticlib
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -msse -msse2 -msse3

OTHER_FILES += \
    cl.pro.user

HEADERS += \
    BufferCL.h \
    CommonCL.h \
    ContextCL.h \
    DeviceCL.h \
    KernelCL.h \
    ProgramCL.h \
    QueueCL.h \
    Storage.h


