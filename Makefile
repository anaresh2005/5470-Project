# Build on Perlmutter: source scripts/env.sh && make
ifndef NCCL_HOME
$(error NCCL_HOME is not set. Run: source scripts/env.sh)
endif

NVCC     ?= nvcc
CXXFLAGS := -std=c++17 -O3 -diag-suppress=177 -I$(NCCL_HOME)/include
LDFLAGS  := -L$(NCCL_HOME)/lib
LDLIBS   := -lnccl
SRC_DIR  := stragglar/stragglar

all: build/allreduce4 build/allreduce8

build/allreduce4: $(SRC_DIR)/allreduce_4GPU.cu
build/allreduce8: $(SRC_DIR)/allreduce_8GPU.cu

build/allreduce4 build/allreduce8:
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf build

.PHONY: all clean
