DATA_PATH=../data
COMMON_PATH=../common
FUTHARK_PATH=./futhark
FUTHARK_PROGRAM=futhark_lexer
CUDA_PATH=./cuda
CUDA_PROGRAM=cuda_lexer
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550

default: bench

.PHONY: clean bench

$(DATA_PATH)/tokens_dense_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/tokens_moderate_500MiB.in:
	(cd $(DATA_PATH) && make)

$(DATA_PATH)/tokens_sparse_500MiB.in:
	(cd $(DATA_PATH) && make)

tokens_dense_500MiB.out: $(DATA_PATH)/tokens_dense_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -b > $@

tokens_moderate_500MiB.out: $(DATA_PATH)/tokens_moderate_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -b > $@

tokens_sparse_500MiB.out: $(DATA_PATH)/tokens_sparse_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -b > $@

$(FUTHARK_PROGRAM): $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut
	(cd futhark && futhark pkg sync)
	futhark cuda $< -o $@

$(CUDA_PROGRAM): $(CUDA_PATH)/$(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

bench: $(DATA_PATH)/tokens_dense_500MiB.in \
	   $(DATA_PATH)/tokens_moderate_500MiB.in \
	   $(DATA_PATH)/tokens_sparse_500MiB.in \
	   tokens_dense_500MiB.out \
	   tokens_moderate_500MiB.out \
	   tokens_sparse_500MiB.out \
	   $(FUTHARK_PROGRAM) \
	   $(CUDA_PROGRAM)
	futhark bench $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut --backend=cuda
	./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_dense_500MiB.in tokens_dense_500MiB.out
	./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_moderate_500MiB.in tokens_dense_500MiB.out
	./$(CUDA_PROGRAM) $(DATA_PATH)/tokens_sparse_500MiB.in tokens_dense_500MiB.out

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.in *.out $(FUTHARK_PATH)/*.c $(FUTHARK_PATH)/$(FUTHARK_PROGRAM)
