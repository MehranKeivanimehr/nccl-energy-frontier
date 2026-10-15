# NCCL 2.27.5 example tuner plugin: plugin.orig.c = unmodified NCCL ext-tuner/example source;
# plugin.c = same + two fixes (default channels 0, cost-table cast): patches/nccl-2.27.5-example-tuner-fixes.patch.
# Reads NCCL_TUNER_CONFIG_FILE (CSV: collective,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff).
NCCL_SRC ?= ../../third_party/nccl-src
build/libnccl-tuner-example.so: plugin.c
	mkdir -p build
	$(CC) -O2 -fPIC -Wall -I$(NCCL_SRC)/ext-tuner/example/nccl -shared -o $@ $<
clean:
	rm -rf build
