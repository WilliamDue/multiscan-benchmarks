CC?=cc
CFLAGS?=-O3 -Wall -Werror -Wextra -pedantic -g
RANDOM_INTS=randomints
COMMON=../common

default: $(RANDOM_INTS)_sparse_500MiB.in $(RANDOM_INTS)_dense_500MiB.in

.PHONY: clean

$(RANDOM_INTS): $(RANDOM_INTS).c $(COMMON)/data.h
	$(CC) $(CFLAGS) -o $(RANDOM_INTS) $<

$(RANDOM_INTS)_sparse_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 95 > $@ 

$(RANDOM_INTS)_dense_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 50 > $@

clean:
	rm -f $(RANDOM_INTS) $(RANDOM_INTS)_*.in