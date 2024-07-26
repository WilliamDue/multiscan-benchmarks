COMPILER?=nvcc
FLAGS?=-O0 --std=c++14

default: main run

.PHONY: clean run

main: main.cu sps.cu.h
	$(COMPILER) $(FLAGS) -o $@ $<

run: main
	./$<

clean:
	rm -f main