# Build on Perlmutter: source scripts/env.sh && make
ifndef NCCL_HOME
$(error NCCL_HOME is not set. Run: source scripts/env.sh)
endif

NVCC     ?= nvcc
CXXFLAGS := -std=c++17 -O3 -diag-suppress=177 -I$(NCCL_HOME)/include
LDFLAGS  := -L$(NCCL_HOME)/lib
LDLIBS   := -lnccl
SRC_DIR  := stragglar/stragglar

all: build/allreduce4 build/allreduce8 build/mp_sendrecv

build/allreduce4: $(SRC_DIR)/allreduce_4GPU.cu
build/allreduce8: $(SRC_DIR)/allreduce_8GPU.cu

build/allreduce4 build/allreduce8:
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

build/mp_sendrecv: tests/mp_sendrecv.cu
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) -I$(MPICH_DIR)/include -o $@ $< $(LDFLAGS) -L$(MPICH_DIR)/lib -lmpi_gnu $(LDLIBS)

clean:
	rm -rf build

.PHONY: all clean

# Aneesh's one-process / four-GPU correctness and timing harness.
build/harness4: harness/allreduce.cu harness/handwritten.cu harness/generated.cu harness/common.h harness/validation.h gen/gen_n4.cu gen/gen_stragglar.h $(SRC_DIR)/allreduce_4GPU.cu
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) -o $@ harness/allreduce.cu harness/handwritten.cu harness/generated.cu $(LDFLAGS) $(LDLIBS)

harness: build/harness4
.PHONY: harness
