COMPILER?=nvcc
FLAGS?=-O3 --std=c++14
TEST=test

default: $(TEST) run

.PHONY: clean run

$(TEST): $(TEST).cu sps.cu.h
	$(COMPILER) $(FLAGS) -o $@ $<

run: $(TEST)
	./$<

clean:
	rm -f $(TEST)