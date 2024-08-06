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

tokens_500MiB.in:
	(cd $(DATA_PATH) && make)

$(FUTHARK_PROGRAM): $(FUTHARK_PATH)/$(FUTHARK_PROGRAM).fut
	futhark cuda $< -o $(FUTHARK_PROGRAM)

$(CUDA_PROGRAM): $(CUDA_PATH)/$(CUDA_PROGRAM).cu $(COMMON_PATH)/sps.cu.h $(COMMON_PATH)/util.cu.h $(COMMON_PATH)/data.h
	$(COMPILER) $(FLAGS) -o $@ $<

bench: $(FUTHARK_PROGRAM) $(CUDA_PROGRAM) 

clean:
	rm -rf $(CUDA_PROGRAM) $(FUTHARK_PROGRAM) *.c *.out $(FUTHARK_PATH)/*.c $(FUTHARK_PATH)/$(FUTHARK_PROGRAM)
