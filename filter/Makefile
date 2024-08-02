DATA_PATH=../data
COMMON_PATH=../common
FUTHARK_PATH=./futhark
FUTHARK_PROGRAM=futhark_filter
CUDA_PATH=./cuda
CUDA_PROGRAM=cuda_filter
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550

default: bench

.PHONY: clean bench

$(DATA_PATH)/randomints_sparse_500MiB.in: 
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_dense_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_moderate_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_empty_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/randomints_full_500MiB.in:
	(cd $(DATA_PATH) && make)

$(FUTHARK_PROGRAM): $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut
	futhark cuda $< -o $(FUTHARK_PROGRAM)

randomints_sparse_500MiB.out: $(DATA_PATH)/randomints_sparse_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_dense_500MiB.out: $(DATA_PATH)/randomints_dense_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_moderate_500MiB.out: $(DATA_PATH)/randomints_moderate_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_empty_500MiB.out: $(DATA_PATH)/randomints_empty_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_full_500MiB.out: $(DATA_PATH)/randomints_full_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

bench: randomints_dense_500MiB.out \
	   randomints_sparse_500MiB.out \
	   randomints_moderate_500MiB.out \
	   randomints_empty_500MiB.out \
	   randomints_full_500MiB.out \
	   $(DATA_PATH)/randomints_dense_500MiB.in \
	   $(DATA_PATH)/randomints_sparse_500MiB.in \
	   $(DATA_PATH)/randomints_moderate_500MiB.in \
	   $(DATA_PATH)/randomints_empty_500MiB.in \
	   $(DATA_PATH)/randomints_full_500MiB.in \
	   $(CUDA_PROGRAM) \
	   $(FUTHARK_PROGRAM)
	futhark bench $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut --backend=cuda
	./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_dense_500MiB.in ./randomints_dense_500MiB.out
	./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_sparse_500MiB.in ./randomints_sparse_500MiB.out
	./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_moderate_500MiB.in ./randomints_moderate_500MiB.out
	./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_empty_500MiB.in ./randomints_empty_500MiB.out
	./$(CUDA_PROGRAM) $(DATA_PATH)/randomints_full_500MiB.in ./randomints_full_500MiB.out

$(CUDA_PROGRAM): $(CUDA_PATH)/$(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.out $(FUTHARK_PATH)/*.c $(FUTHARK_PATH)/$(FUTHARK_PROGRAM)
