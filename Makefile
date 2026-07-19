CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER := g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

IMGVIEWER := gwenview

SRC := $(wildcard *.cu)
OBJ := $(SRC:.cu=.o)
HDR := $(wildcard *.cuh *.h *.hpp)

TARGET := cudart.out
IMG := out.ppm

NVCCFLAGS := -g -G -m64 -rdc=true

all: $(IMG)

$(IMG): $(TARGET)
	rm -f $@
	./$(TARGET)
	$(IMGVIEWER) $@

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJ)

%.o: %.cu $(HDR)
	$(NVCC) $(NVCCFLAGS) -dc -c $< -o $@

view: $(IMG)
	nohup $(IMGVIEWER) $(IMG) >/dev/null 2>&1 &

profile: $(TARGET)
	rm -f profile.ncu-rep
	sudo ncu --export profile ./$(TARGET)
	ncu-ui profile.ncu-rep

clean:
	rm -f $(OBJ) $(TARGET) $(IMG)

.PHONY: all clean
