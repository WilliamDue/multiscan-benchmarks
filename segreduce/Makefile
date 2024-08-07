DATA_PATH=../data
COMMON_PATH=../common
FUTHARK_PATH=./futhark
FUTHARK_PROGRAM=futhark_segreduce
CUDA_PATH=./cuda
CUDA_PROGRAM=cuda_segreduce
COMPILER?=nvcc
FLAGS?=-O3 --std=c++14 -diag-suppress 550 -diag-suppress 20054

default: bench

.PHONY: clean bench

randomints_sparse_500MiB.in: $(FUTHARK_PROGRAM)
	(cd $(DATA_PATH) && make)
	cp $(DATA_PATH)/randomints_sparse_500MiB.in .
	cat $(DATA_PATH)/randomints_sparse_500MiB.in | ./$< -e flags -b >> $@

randomints_dense_500MiB.in: $(FUTHARK_PROGRAM)
	(cd $(DATA_PATH) && make)
	cp $(DATA_PATH)/randomints_dense_500MiB.in .
	cat $(DATA_PATH)/randomints_sparse_500MiB.in | ./$< -e flags -b >> $@

randomints_moderate_500MiB.in: $(FUTHARK_PROGRAM)
	(cd $(DATA_PATH) && make)
	cp $(DATA_PATH)/randomints_moderate_500MiB.in .
	cat $(DATA_PATH)/randomints_sparse_500MiB.in | ./$< -e flags -b >> $@

randomints_empty_500MiB.in: $(FUTHARK_PROGRAM)
	(cd $(DATA_PATH) && make)
	cp $(DATA_PATH)/randomints_empty_500MiB.in .
	cat $(DATA_PATH)/randomints_sparse_500MiB.in | ./$< -e flags -b >> $@

randomints_full_500MiB.in: $(FUTHARK_PROGRAM)
	(cd $(DATA_PATH) && make)
	cp $(DATA_PATH)/randomints_full_500MiB.in .
	cat $(DATA_PATH)/randomints_sparse_500MiB.in | ./$< -e flags -b >> $@

$(FUTHARK_PROGRAM): $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut
	(cd $(FUTHARK_PATH) && futhark pkg sync)
	futhark cuda $< -o $(FUTHARK_PROGRAM)

randomints_sparse_500MiB.out: randomints_sparse_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_dense_500MiB.out: randomints_dense_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_moderate_500MiB.out: randomints_moderate_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_empty_500MiB.out: randomints_empty_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

randomints_full_500MiB.out: randomints_full_500MiB.in $(FUTHARK_PROGRAM)
	cat $< | ./$(FUTHARK_PROGRAM) -e expected -b > $@

data: randomints_dense_500MiB.out \
	  randomints_sparse_500MiB.out \
	  randomints_moderate_500MiB.out \
	  randomints_empty_500MiB.out \
	  randomints_full_500MiB.out \
	  randomints_dense_500MiB.in \
	  randomints_sparse_500MiB.in \
	  randomints_moderate_500MiB.in \
	  randomints_empty_500MiB.in \
	  randomints_full_500MiB.in \

bench: randomints_dense_500MiB.out \
	   randomints_sparse_500MiB.out \
	   randomints_moderate_500MiB.out \
	   randomints_empty_500MiB.out \
	   randomints_full_500MiB.out \
	   randomints_dense_500MiB.in \
	   randomints_sparse_500MiB.in \
	   randomints_moderate_500MiB.in \
	   randomints_empty_500MiB.in \
	   randomints_full_500MiB.in \
	   $(CUDA_PROGRAM) \
	   $(FUTHARK_PROGRAM)
	@echo ""
	@echo "Futhark:"
	futhark bench $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut --backend=cuda
	@echo ""
	@echo "Cuda:"
	./$(CUDA_PROGRAM) ./randomints_full_500MiB.in ./randomints_full_500MiB.out
	@echo ""
	./$(CUDA_PROGRAM) ./randomints_dense_500MiB.in ./randomints_dense_500MiB.out
	@echo ""
	./$(CUDA_PROGRAM) ./randomints_moderate_500MiB.in ./randomints_moderate_500MiB.out
	@echo ""
	./$(CUDA_PROGRAM) ./randomints_sparse_500MiB.in ./randomints_sparse_500MiB.out
	@echo ""
	./$(CUDA_PROGRAM) ./randomints_empty_500MiB.in ./randomints_empty_500MiB.out
	@echo ""

$(CUDA_PROGRAM): $(CUDA_PATH)/$(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.out *.in $(FUTHARK_PATH)/*.c $(FUTHARK_PATH)/$(FUTHARK_PROGRAM)
