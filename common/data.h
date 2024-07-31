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

  for (uint8_t i = 0; i < sizeof(uint32_t); i++) {
    putchar(result.str[i]);
  }
}

void putu64(uint64_t n) {
  union u64 result;
  result.i = n;

  for (uint8_t i = 0; i < sizeof(uint64_t); i++) {
    putchar(result.str[i]);
  }
}

/*
void read_int32(uint8_t * input, int32_t * output) {

}
*/