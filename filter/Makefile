DATA_PATH=../data
COMMON_PATH=../common
FUTHARK_PATH=./futhark
FUTHARK_PROGRAM=futhark_filter
CUDA_PATH=./cuda
CUDA_PROGRAM=cuda_filter
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14

default: bench

.PHONY: clean bench

randomints_sparse_100MiB.in: 
	(cd $(DATA_PATH) && make)

randomints_dense_100MiB.in:
	(cd $(DATA_PATH) && make)

$(FUTHARK_PROGRAM): $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut
	futhark cuda $< -o $(FUTHARK_PROGRAM)

randomints_sparse_100MiB.out: $(DATA_PATH)/randomints_sparse_100MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_dense_100MiB.out: $(DATA_PATH)/randomints_dense_100MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

bench: $(FUTHARK_PROGRAM) randomints_dense_100MiB.out randomints_sparse_100MiB.out
	futhark bench $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut

$(CUDA_PROGRAM): $(CUDA_PATH)/$(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.out $(FUTHARK_PATH)/*.c $(FUTHARK_PATH)/$(FUTHARK_PROGRAM)