CC?=cc
CFLAGS?=-O3 -Wall -Werror -Wextra -pedantic -g
RANDOM_INTS=randomints
TOKENS=tokens
COMMON=../common

default: $(RANDOM_INTS)_full_500MiB.in \
		 $(RANDOM_INTS)_dense_500MiB.in \
		 $(RANDOM_INTS)_moderate_500MiB.in \
		 $(RANDOM_INTS)_sparse_500MiB.in \
		 $(RANDOM_INTS)_empty_500MiB.in \
		 $(TOKENS)_dense_500MiB.in \
		 $(TOKENS)_moderate_500MiB.in \
		 $(TOKENS)_sparse_500MiB.in

.PHONY: clean

$(RANDOM_INTS): $(RANDOM_INTS).c $(COMMON)/data.h
	$(CC) $(CFLAGS) -o $(RANDOM_INTS) $<

$(RANDOM_INTS)_full_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 0 > $@

$(RANDOM_INTS)_dense_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 5 > $@

$(RANDOM_INTS)_moderate_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 50 > $@ 

$(RANDOM_INTS)_sparse_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 95 > $@

$(RANDOM_INTS)_empty_500MiB.in: $(RANDOM_INTS)
	./$< 131072000 100 > $@ 

$(TOKENS): $(TOKENS).c
	$(CC) $(CFLAGS) -o $(TOKENS) $<

$(TOKENS)_dense_500MiB.in: $(TOKENS)
	./$< 524288000 0:10 > $@ 

$(TOKENS)_moderate_500MiB.in: $(TOKENS)
	./$< 524288000 100:110 > $@ 

$(TOKENS)_sparse_500MiB.in: $(TOKENS)
	./$< 524288000 1000:1010 > $@ 

clean:
	rm -f $(RANDOM_INTS) $(RANDOM_INTS)_*.in $(TOKENS) $(TOKENS)_*.in