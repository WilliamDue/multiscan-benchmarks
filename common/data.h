#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

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

uint8_t* read_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint8_t* buffer = (uint8_t*) malloc(*size);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    size_t bytes = fread(buffer, 1, *size, file);
    if (bytes != *size) {
        fprintf(stderr, "Error reading file %s\n", filename);
        free(buffer);
        fclose(file);
        return NULL;
    }

    fclose(file);
    return buffer;
}

int32_t* read_i32_array(const char* filename, size_t* size) {
  size_t file_size;
  uint8_t* buffer = read_file(filename, &file_size);
  assert(buffer != NULL);
  uint8_t* buffer_ptr = buffer;
  uint8_t header[7] = {'b', 2U, 1U, ' ', 'i', '3', '2'};

  for (size_t i = 0; i < sizeof(header); i++) {
    assert(buffer_ptr[i] == header[i]);
  }
  buffer_ptr += sizeof(header);
  
  union u64 array_size;

  for (uint8_t i = 0; i < sizeof(uint64_t); i++) {
    array_size.str[i] = buffer_ptr[i];
  }

  buffer_ptr += sizeof(uint64_t);
  *size = array_size.i;
  size_t offset = sizeof(header) + sizeof(uint64_t);
  size_t bytes = file_size - offset;
  assert(array_size.i == bytes / sizeof(int32_t));
  
  int32_t* new_buffer = (int32_t*) malloc(bytes);

  assert(new_buffer != NULL);
  
  memcpy(new_buffer, buffer_ptr, bytes);
  free(buffer);

  return new_buffer;
}
