CC?=cc
CFLAGS?=-O3 -Wall -Werror -Wextra -pedantic -g
RANDOM_INTS=randomints
COMMON=common

default: $(RANDOM_INTS)

.PHONY: clean

$(RANDOM_INTS): $(COMMON)/$(RANDOM_INTS).c $(COMMON)/data.h
	$(CC) $(CFLAGS) -o $(RANDOM_INTS) $<

$(RANDOM_INTS)_95_100MiB.in: $(RANDOM_INTS)
	./$< 26214400 95 > $@ 

$(RANDOM_INTS)_50_100MiB.in: $(RANDOM_INTS)
	./$< 26214400 50 > $@

$(RANDOM_INTS)_50_10.in: $(RANDOM_INTS)
	./$< 10 50 > $@ 

clean:
	rm -f $(RANDOM_INTS) $(RANDOM_INTS)_*.in