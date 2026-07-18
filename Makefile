CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
IMGVIEWER = gwenview

DIR := $(CURDIR)
IMG = $(DIR)/out.ppm
# select one of these for Debug vs. Release
NVCC_DBG       = -g -G
#NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64

all: out.ppm
	$(IMGVIEWER) $(IMG)

cudart.out: main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o cudart.out main.cu

out.ppm: cudart.out
	rm -f $(IMG)
	./cudart.out > $(IMG)

profile_basic: cudart.out
	nvprof ./cudart.out > $(IMG)

# use nvprof --query-metrics
profile_metrics: cudart.out
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart.out > out.ppm

clean:
	rm -f cudart.out out.ppm out.jpg
