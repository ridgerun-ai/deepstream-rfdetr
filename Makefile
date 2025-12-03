# SPDX-License-Identifier: MIT
# Copyright (c) 2025 RidgeRun, LLC <support@ridgerun.ai>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

VERSION := 0.1.0

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

.PHONY: all clean lint format version

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

version:
	@echo "DeepStream RF-DETR $(VERSION)"
