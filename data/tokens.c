#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/data.h"
#include <stdint.h>

void random_letter() {
  int i = rand() % 60;
  if (i < 25) {
    putchar('a' + i);
  } else if (i < 50) {
    putchar('A' + i - 25);
  } else {
    putchar('0' + i - 50);
  }
}

void random_whitespace() {
  const char whitespace[4] = {'\t', '\n', ' ', '\r'};
  putchar(whitespace[rand()%sizeof(whitespace)]);
}

int main(int argc, char *argv[]) {
  assert(argc == 3);
  unsigned long max_size = 0;
  sscanf(argv[1], "%lu", &max_size);
  unsigned short lowerbound = 0;
  unsigned short upperbound = 0;
  sscanf(argv[2], "%hu:%hu", &lowerbound, &upperbound);
  unsigned long curr_size = 0;

  uint8_t header[7] = {'b', 2U, 1U, ' ', ' ', 'u', '8'};
  for (size_t i = 0; i < 7; i++) {
    putchar(header[i]);
  }

  putu64(max_size);

  while (curr_size < max_size) {
    int i = rand() % 4;
    switch (i) {
    case 0:
      {
        int count = lowerbound + rand() % upperbound;
        for (int j = 0; j < count && curr_size < max_size; j++) {
          random_letter();
          curr_size += 1;
        }
      }
      break;
    case 1:
      {
        int count = 3 + rand() % 6;
        for (int j = 0; j < count && curr_size < max_size; j++) {
          random_whitespace();
          curr_size += 1;
        }
        break;
      }
    case 2:
      putchar('(');
      curr_size += 1;
      break;
    case 3:
      putchar(')');
      curr_size += 1;
      break;
    }
  }
}
