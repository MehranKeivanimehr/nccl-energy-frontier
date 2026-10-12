# Build: make NCCL_SRC=<path to NCCL 2.27.5 source>   (headers from ext-profiler/example/nccl)
NCCL_SRC ?= ../../third_party/nccl-src
CFLAGS ?= -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
build/libnccl-profiler-nccle.so: plugin.c
	mkdir -p build
	$(CC) $(CFLAGS) -I$(NCCL_SRC)/ext-profiler/example/nccl -shared -o $@ $< -lpthread
clean:
	rm -rf build
