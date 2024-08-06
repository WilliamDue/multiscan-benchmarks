#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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

int random_token( unsigned short lowerbound,  unsigned short upperbound) {
  int i = rand() % 4;
  switch (i) {
  case 0:
    {
      int count = lowerbound + rand() % upperbound;
      for (int j = 0; j < count; j++) { random_letter(); }
      return count;
    }
    break;
  case 1:
    {
      int count = 3 + rand() % 6;
      for (int j = 0; j < count; j++) { random_whitespace(); }
      return count;
    }
  case 2:
    putchar('(');
    break;
  case 3:
    putchar(')');
    break;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  assert(argc == 3);
  unsigned long max_size = 0;
  sscanf(argv[1], "%lu", &max_size);
  unsigned short lowerbound = 0;
  unsigned short upperbound = 0;
  sscanf(argv[2], "%hu:%hu", &lowerbound, &upperbound);
  unsigned long curr_size = 0;
  while (curr_size < max_size) {
    curr_size += random_token(lowerbound, upperbound);
  }
}
