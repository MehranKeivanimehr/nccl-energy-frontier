# nccl-energy-frontier v2.2 guarded tuner plugin (see plugin.c and guard.c headers).
# Reads NCCL_TUNER_CONFIG_FILE: "#@scope key=value" header lines + example-tuner CSV rules.
NCCL_SRC ?= ../../third_party/nccl-src
build/libnccl-tuner-guarded.so: plugin.c guard.c guard.h
	mkdir -p build
	$(CC) -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -I$(NCCL_SRC)/ext-tuner/example/nccl -I. -shared -o $@ plugin.c guard.c -ldl
clean:
	rm -rf build
