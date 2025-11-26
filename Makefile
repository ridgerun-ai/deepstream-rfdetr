# Copyright (C) 2025 RidgeRun, LLC <support@ridgerun.ai>
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to
# RidgeRun, LLC. No part of this program may be photocopied,
# reproduced or translated into another programming language without
# prior written consent of RidgeRun, LLC. The user is free to modify
# the source code after obtaining a software license from
# RidgeRun. All source code changes must be provided back to RidgeRun
# without any encumbrance.

DS_HOME ?= /opt/nvidia/deepstream/deepstream/
CUDA_HOME ?= /usr/local/cuda/
DEV ?= 0

CXXFLAGS := -Wall -std=c++20 -fPIC -O3

CXXFLAGS +=                     \
  -I$(DS_HOME)/sources/includes \
  -I$(CUDA_HOME)/include

ifeq ($(DEV),1)
CXXFLAGS += -O0 -ggdb3 -Werror
endif

LIBS := -lnvinfer
LDFLAGS := -shared

TARGET := libdeepstream-rfdetr.so

SRCS := deepstream_rfdetr_bbox.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all clean lint format

all: $(TARGET)

$(TARGET): $(OBJS) Makefile
	$(CXX) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

%.o: %.cpp Makefile
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.so *~ \#*

lint:
	clang-tidy --extra-arg=-isystem/usr/include/c++/13 --extra-arg=-isystem/usr/include/aarch64-linux-gnu/c++/13/  $(SRCS)

format:
	clang-format --style=file -i $(SRCS)
