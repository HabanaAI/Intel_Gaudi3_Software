#pragma once

#include <stdlib.h>
#include "types.h"

int getIdx(int b, int c, int y, int x, int cSize, int hSize, int wSize);

bool ReadBMPplain(std::string     fNameStr,
                  int&            width,
                  int&            height,
                  unsigned char** inputBufPtr,
                  bool            bFlipImageArray = false);

void WriteBMP(const char* filename, int width, int height, unsigned char* buf, bool bFlipImageArray = false);

void writeBufferToFile(const char* fileName, unsigned char* buf, int numBytes);

void readBufferFromFile(const char* fileName, unsigned char* buf, int numBytes);

int compareResults(int            batch_size,
                   int            channel_size,
                   int            output_height,
                   int            output_width,
                   unsigned char* rotatorExpectedOutput,
                   unsigned char* rotatorActualOutput);

void copyStripeToFullTensor(int            batch_size,
                            int            channel_size,
                            int            stripeHeight,
                            int            stripeWidth,
                            int            batchIdx,
                            int            verticalOffset,
                            int            horizontalOffset,
                            int            outputHeight,
                            int            outputWidth,
                            unsigned char* stripeData,
                            unsigned char* tensorData);
