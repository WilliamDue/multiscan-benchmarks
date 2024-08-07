#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

union i32 {
  int32_t i;
  char str[sizeof(int32_t)];
};

union u32 {
  uint32_t i;
  char str[sizeof(uint32_t)];
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

void read_i32_bool_array(const char* filename,
                         int32_t** vals,
                         size_t* vals_size,
                         bool** flags,
                         size_t* flags_size) {
  assert(*vals == NULL);
  assert(*flags == NULL);
  size_t file_size;
  uint8_t* buffer = read_file(filename, &file_size);
  assert(buffer != NULL);
  uint8_t* buffer_ptr = buffer;
  char fst_header[7] = {'b', 2, 1, ' ', 'i', '3', '2'};

  for (size_t i = 0; i < sizeof(fst_header); i++) {
    assert(buffer_ptr[i] == fst_header[i]);
  }
  buffer_ptr += sizeof(fst_header);
  
  union u64 union_vals_size;

  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    union_vals_size.str[i] = buffer_ptr[i];
  }

  buffer_ptr += sizeof(uint64_t);
  *vals_size = union_vals_size.i;
  size_t vals_bytes = union_vals_size.i * sizeof(int32_t);
  *vals = (int32_t*) malloc(vals_bytes);

  assert(*vals != NULL);
  memcpy(*vals, buffer_ptr, vals_bytes);
  buffer_ptr += vals_bytes;

  char snd_header[7] = {'b', 2, 1, 'b', 'o', 'o', 'l'};

  for (size_t i = 0; i < sizeof(snd_header); i++) {
    assert(buffer_ptr[i] == snd_header[i]);
  }

  buffer_ptr += sizeof(snd_header);
  
  union u64 union_flags_size;

  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    union_flags_size.str[i] = buffer_ptr[i];
  }

  buffer_ptr += sizeof(uint64_t);
  *flags_size = union_flags_size.i;
  size_t flags_bytes = union_flags_size.i * sizeof(bool);
  *flags = (bool*) malloc(flags_bytes);
  assert(*flags != NULL);
  memcpy(*flags, buffer_ptr, flags_bytes);

  free(buffer);
}

void read_tuple_u32_u8_array(const char* filename,
                             uint32_t** indices,
                             size_t* indices_size,
                             uint8_t** tokens,
                             size_t* tokens_size) {
  assert(*indices == NULL);
  assert(*tokens == NULL);
  size_t file_size;
  uint8_t* buffer = read_file(filename, &file_size);
  assert(buffer != NULL);
  uint8_t* buffer_ptr = buffer;
  uint8_t fst_header[7] = {'b', 2, 1, ' ', 'u', '3', '2'};

  for (size_t i = 0; i < sizeof(fst_header); i++) {
    assert(buffer_ptr[i] == fst_header[i]);
  }
  buffer_ptr += sizeof(fst_header);
  
  union u64 union_indices_size;

  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    union_indices_size.str[i] = buffer_ptr[i];
  }


  buffer_ptr += sizeof(uint64_t);
  *indices_size = union_indices_size.i;
  size_t indices_bytes = union_indices_size.i * sizeof(uint32_t);
  *indices = (uint32_t*) malloc(indices_bytes);

  assert(*indices != NULL);
  memcpy(*indices, buffer_ptr, indices_bytes);
  buffer_ptr += indices_bytes;

  assert(*buffer_ptr == '\n');
  buffer_ptr++;
  uint8_t snd_header[7] = {'b', 2, 1, ' ', ' ', 'u', '8'};

  for (size_t i = 0; i < sizeof(snd_header); i++) {
    assert(buffer_ptr[i] == snd_header[i]); // printf("%lu\n", buffer_ptr[i]);
  }

  buffer_ptr += sizeof(snd_header);
  
  union u64 union_tokens_size;

  for (size_t i = 0; i < sizeof(uint64_t); i++) {
    union_tokens_size.str[i] = buffer_ptr[i];
  }

  buffer_ptr += sizeof(uint64_t);
  *tokens_size = union_tokens_size.i;
  size_t tokens_bytes = union_tokens_size.i * sizeof(uint8_t);
  *tokens = (uint8_t*) malloc(tokens_bytes);
  assert(*tokens != NULL);
  memcpy(*tokens, buffer_ptr, tokens_bytes);

  free(buffer);
}