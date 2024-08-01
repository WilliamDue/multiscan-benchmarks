CC?=cc
CFLAGS?=-O3 -Wall -Werror -Wextra -pedantic -g
RANDOM_INTS=randomints
COMMON=../common

default: $(RANDOM_INTS)_sparse_100MiB.in $(RANDOM_INTS)_dense_100MiB.in

.PHONY: clean

$(RANDOM_INTS): $(RANDOM_INTS).c $(COMMON)/data.h
	$(CC) $(CFLAGS) -o $(RANDOM_INTS) $<

$(RANDOM_INTS)_sparse_100MiB.in: $(RANDOM_INTS)
	./$< 26214400 95 > $@ 

$(RANDOM_INTS)_dense_100MiB.in: $(RANDOM_INTS)
	./$< 26214400 50 > $@

clean:
	rm -f $(RANDOM_INTS) $(RANDOM_INTS)_*.in