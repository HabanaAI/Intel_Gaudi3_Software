#include "rotator_utils.h"
#include "defs.h"

int getIdx(int b, int c, int y, int x, int cSize, int hSize, int wSize)
{
    return b * cSize * hSize * wSize + c * hSize * wSize + y * wSize + x;
}

bool ReadBMPplain(std::string fNameStr, int& width, int& height, unsigned char** inputBufPtr, bool bFlipImageArray)
{
    char filename[200];
    strcpy(filename, fNameStr.c_str());
    int   i;
    FILE* f = fopen(filename, "rb");
    if (f == NULL)
    {
        LOG_ERR(GC, "Error opening input bmp file {}\n", filename);
        return false;
    }

    unsigned char info[54];
    int           bytesRead = fread(info, sizeof(unsigned char), 54, f);  // read the 54-byte header
    if (bytesRead == 0)                                                   // to pass release
    {
        LOG_DEBUG(GC, "Read zero bytes from the bmp header");
    }

    // extract image height and width from header
    width  = *(int*)&info[18];
    height = *(int*)&info[22];

    // Allocate the buffer
    int            planeSize = width * height;
    unsigned char* buf       = new unsigned char[3 * planeSize];

    LOG_DEBUG(SYN_TEST, "Reading input bmp image {} , dimensions are {}x{}\n", filename, width, height);

    int            row_padded = (width * 3 + 3) & (~3);
    unsigned char* data       = new unsigned char[row_padded];

    // image array is swapped in BMP file. We can either write/read it swapped or flip it
    if (bFlipImageArray)
    {
        for (i = height - 1; i >= 0; i--)
        {
            bytesRead = fread(data, sizeof(unsigned char), row_padded, f);
            if (bytesRead == 0)  // to pass release
            {
                LOG_DEBUG(GC, "Read zero bytes from the bmp image");
            }
            for (int j = 0; j < width * 3; j += 3)
            {
                buf[planeSize * 0 + i * width + j / 3] = data[j];
                buf[planeSize * 1 + i * width + j / 3] = data[j + 1];
                buf[planeSize * 2 + i * width + j / 3] = data[j + 2];
            }
        }
    }
    else
    {
        for (i = 0; i < height; i++)
        {
            bytesRead = fread(data, sizeof(unsigned char), row_padded, f);
            if (bytesRead == 0)  // to pass release
            {
                LOG_DEBUG(GC, "Read zero bytes from the bmp image");
            }
            for (int j = 0; j < width * 3; j += 3)
            {
                buf[planeSize * 0 + i * width + j / 3] = data[j];
                buf[planeSize * 1 + i * width + j / 3] = data[j + 1];
                buf[planeSize * 2 + i * width + j / 3] = data[j + 2];
            }
        }
    }

    *inputBufPtr = buf;
    fclose(f);

    delete[] data;

    return true;
}

void WriteBMP(const char* filename, int width, int height, unsigned char* buf, bool bFlipImageArray)
{
    int   i;
    FILE* f = fopen(filename, "wb");

    int filesize = 54 + 3 * width * height;

    unsigned char bmpfileheader[14] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0};
    unsigned char bmpinfoheader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize >> 8);
    bmpfileheader[4] = (unsigned char)(filesize >> 16);
    bmpfileheader[5] = (unsigned char)(filesize >> 24);

    bmpinfoheader[4]  = (unsigned char)(width);
    bmpinfoheader[5]  = (unsigned char)(width >> 8);
    bmpinfoheader[6]  = (unsigned char)(width >> 16);
    bmpinfoheader[7]  = (unsigned char)(width >> 24);
    bmpinfoheader[8]  = (unsigned char)(height);
    bmpinfoheader[9]  = (unsigned char)(height >> 8);
    bmpinfoheader[10] = (unsigned char)(height >> 16);
    bmpinfoheader[11] = (unsigned char)(height >> 24);

    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);

    int            row_padded = (width * 3 + 3) & (~3);
    unsigned char* data       = new unsigned char[row_padded];

    unsigned char* channel0 = buf;
    unsigned char* channel1 = buf + width * height;
    unsigned char* channel2 = buf + width * height * 2;

    // image array is swapped in BMP file. We can either write/read it swapped or flip it
    if (bFlipImageArray)
    {
        for (i = height - 1; i >= 0; i--)
        {
            for (int j = 0; j < width * 3; j += 3)
            {
                data[j]     = channel0[i * width + j / 3];
                data[j + 1] = channel1[i * width + j / 3];
                data[j + 2] = channel2[i * width + j / 3];
            }
            fwrite(data, sizeof(unsigned char), row_padded, f);
        }
    }
    else
    {
        for (i = 0; i < height; i++)
        {
            for (int j = 0; j < width * 3; j += 3)
            {
                data[j]     = channel0[i * width + j / 3];
                data[j + 1] = channel1[i * width + j / 3];
                data[j + 2] = channel2[i * width + j / 3];
            }
            fwrite(data, sizeof(unsigned char), row_padded, f);
        }
    }

    fclose(f);

    delete[] data;
}

void writeBufferToFile(const char* fileName, unsigned char* buf, int numBytes)
{
    FILE* fp = fopen(fileName, "w");
    HB_ASSERT(fp != nullptr, "Error opening {} for writing", fileName);

    int bytesWritten = fwrite(buf, 1, numBytes, fp);
    if (bytesWritten == 0)
    {
        LOG_DEBUG(GC, "Wrote zero bytes to the descriptor");
    }
    fclose(fp);
}

void readBufferFromFile(const char* fileName, unsigned char* buf, int numBytes)
{
    FILE* fp = fopen(fileName, "r");
    HB_ASSERT(fp != nullptr, "Error opening {} for reading\n", fileName);

    int bytesRead = fread(buf, 1, numBytes, fp);
    if (bytesRead == 0)
    {
        LOG_DEBUG(GC, "Read zero bytes from the buffer\n");
    }
    fclose(fp);
}

int compareResults(int            batch_size,
                   int            channel_size,
                   int            output_height,
                   int            output_width,
                   unsigned char* rotatorExpectedOutput,
                   unsigned char* rotatorActualOutput)
{
    int pixelIdx  = 0;
    int numErrors = 0;

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channel_size; c++)
        {
            for (int y = 0; y < output_height; y++)
            {
                for (int x = 0; x < output_width; x++)
                {
                    pixelIdx = getIdx(b, c, y, x, channel_size, output_height, output_width);
                    if (rotatorExpectedOutput[pixelIdx] != rotatorActualOutput[pixelIdx])
                    {
                        if (numErrors < 20)
                        {
                            LOG_DEBUG(
                                GC,
                                "Mistatch in pixel {}, batch {}, channel {}, row {}, col {}. expected {}, Actual {}\n",
                                pixelIdx,
                                b,
                                c,
                                y,
                                x,
                                rotatorExpectedOutput[pixelIdx],
                                rotatorActualOutput[pixelIdx]);
                        }
                        numErrors++;
                    }  // if mismatch
                    // pixelIdx++;
                }  // for x
            }      // for y
        }          // for c
    }              // for b

    return numErrors;
}

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
                            unsigned char* tensorData)
{
    // Copy stripeResult into the proper place in the output
    for (int c = 0; c < channel_size; c++)
    {
        for (int y = 0; y < stripeHeight; y++)
        {
            for (int x = 0; x < stripeWidth; x++)
            {
                int expectedIdx         = getIdx(batchIdx,
                                         c,
                                         y + verticalOffset,
                                         x + horizontalOffset,
                                         channel_size,
                                         outputHeight,
                                         outputWidth);
                int stripeIdx           = getIdx(0, c, y, x, channel_size, stripeHeight, stripeWidth);
                tensorData[expectedIdx] = stripeData[stripeIdx];
            }
        }
    }
}
