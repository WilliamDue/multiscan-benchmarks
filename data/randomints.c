#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../common/data.h"

void put_random_int(int chance) {
    int32_t result = abs(rand());
    int j = 1 + (rand() % 100);

    if (j < chance) {
        result *= -1;
    }

    puti32(result);
}

int main(int argc, char *argv[]) {
  assert(argc == 3);
  size_t max_size = 0;
  sscanf(argv[1], "%lu", &max_size);
  int chance = 0;
  sscanf(argv[2], "%d", &chance);
  assert(0 <= chance);
  assert(chance <= 100);
  uint8_t header[7] = {'b', 2U, 1U, ' ', 'i', '3', '2'};
  for (size_t i = 0; i < 7; i++) {
    putchar(header[i]);
  }
  
  putu64(max_size);
  for (size_t i = 0; i < max_size; i++) {
    put_random_int(chance);
  }
}