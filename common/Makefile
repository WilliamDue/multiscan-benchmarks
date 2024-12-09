COMPILER?=nvcc
FLAGS?=-O3 --std=c++14
TEST=test
RED=\u001b[31m
GREEN=\u001b[32m
DEFAULT=\033[39m

default: $(TEST) run

.PHONY: clean run

$(TEST): $(TEST).cu sps.cu.h
	$(COMPILER) $(FLAGS) -o $@ $<

run: $(TEST)
	@echo -e "$(GREEN)=== CUDA SINGLE PASS SCAN ===$(DEFAULT)"
	@./$<
	@echo -e "$(GREEN)============$(DEFAULT)"

clean:
	rm -f $(TEST)