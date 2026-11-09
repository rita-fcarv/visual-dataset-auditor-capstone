NVCC = nvcc
NVCC_FLAGS = -std=c++14 -arch=sm_86 -O2
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

TARGET = visual_auditor
SOURCES = src/main.cu src/image_processor.cu src/image_metrics.cu src/image_hash.cu src/similarity.cu

all: $(TARGET)

$(TARGET): $(SOURCES) src/image_processor.cuh src/image_metrics.cuh src/image_hash.cuh src/similarity.cuh
	$(NVCC) $(NVCC_FLAGS) -Isrc $(SOURCES) -o $@ $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
