#pragma once

#include <stdlib.h>

// return index of max element
int max_index(int8_t* in, int size);

// return indexes of N top elements
void top(int8_t* in, int size, int* out, int N);
