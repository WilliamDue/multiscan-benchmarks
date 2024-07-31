#include <stdint.h>
#include <stdio.h>

union i32 {
  int32_t i;
  char str[sizeof(int32_t)];
};

union u64 {
  uint64_t i;
  char str[sizeof(uint64_t)];
};

void puti32(int32_t n) {
  union i32 result;
  result.i = n;

  for (int8_t i = sizeof(uint32_t) - 1; i >= 0; i--) {
    putchar(result.str[i]);
  }
}

void putu64(uint64_t n) {
  union u64 result;
  result.i = n;

  for (int8_t i = sizeof(uint64_t) - 1; i >= 0; i--) {
    putchar(result.str[i]);
  }
}