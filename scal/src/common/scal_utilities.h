#pragma once
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string>

#include "scal.h"

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/* For simulator, make sure output address is non canonical
 * and easy to interperate and induerstand wer'e on simulator and not real device
 * for example, bits 62 =0, but 63 = 1,..
 */
#define IS_SIM_LBW_MEM(address)  (((((uint64_t)address) & 0xC000000000000000) >> 62) == 2)

static constexpr uint64_t c_lbw_fd_shift  = 40;

/**
 *  returns home dir
 * @param 
 * @return homedir
 */
const char* getHomeFolder();

/**
 * checks if a file exists
 * @param name full path of the file
 * @return true if the file exists, false otherwise
 */
bool fileExists(const std::string& name);

/**
 * Loads a FW binary from the file
 * @param fd device handle
 * @param binFileName full path of the FW binary file
 * @param dccmImageSize size in bytes of the core's dccm
 * @param hbmImageSize size in bytes of the core's hbm image
 * @param dccm - the content of the dccm image
 * @param hbm - the content of the hbm image
 * @return true upon success, false otherwise
 */
bool loadFWImageFromFile_gaudi2(
    int fd,
    const std::string &binFileName,
    unsigned& dccmImageSize,
    unsigned& hbmImageSize,
    uint8_t * dccm,
    uint8_t * hbm,
    struct arc_fw_metadata_t* meta);

bool loadFWImageFromFile_gaudi3(
    int fd,
    const std::string &binFileName,
    unsigned& dccmImageSize,
    unsigned& hbmImageSize,
    uint8_t * dccm,
    uint8_t * hbm,
    struct arc_fw_metadata_t* meta);


bool isSimFD(int fd);

/*
  SCAL utilities provides scal with infrastructure for low level
  functionalities, involving memory operations, read, write, map on device
  and more.
*/

/**
 * Maps LBW memory on the device
 * @param fd device handle
 * @param lbwAddress device address to be mapped
 * @param size size of memory to be mapped
 * @return mapped address
 */
void* mapLbwMemory(int fd, uint64_t lbwAddress, uint32_t size, uint32_t &allocatedSize);

/**
 * unmaps LBW memory on the device
 * @param address device address to be mapped
 * @param size size of memory to be mapped
 * @return 0 for success, -1 otherwise
 */
int unmapLbwMemory(void* address, uint32_t size);

/**
 * Reads from LBW memory on the device
 * @param dstPtr host address, to write the output
 * @param srcPtr device address to read from
 * @param size size of memory to be read
  */
void readLbwMem(void* dstPtr, volatile void* srcPtr, uint32_t size);

inline uint32_t readLbwReg(volatile uint32_t* srcPtr)
{
    if (unlikely((IS_SIM_LBW_MEM(srcPtr))))
    {
        uint32_t ret;
        readLbwMem(&ret, srcPtr, sizeof(uint32_t));
        return ret;
    }
    else
    {
        _mm_mfence();
        return *srcPtr;
    }
}

/**
 * Write to LBW memory on the device
 * @param dstPtr device address, to write todevice handle
 * @param srcPtr host address to read from
 * @param size size of memory to be read
  */
void writeLbwMem(volatile void* dstPtr, void* srcPtr, uint32_t size);

/**
 * Inline version of write_lbw_mem, specifically for reg write
 * @param dstPtr device address, to write todevice handle
 * @param value value to be written
   */
inline void writeLbwReg(uint32_t* dstPtr, uint32_t value)
{

    if (unlikely((IS_SIM_LBW_MEM(dstPtr))))
    {
        writeLbwMem((volatile uint32_t *)dstPtr, &value, sizeof(uint32_t));
    }
    else
    {
        _mm_sfence();
        *dstPtr = value;
    }
}
inline void writeLbwReg(volatile uint32_t* dstPtr, uint32_t value)
{

    if (unlikely((IS_SIM_LBW_MEM(dstPtr))))
    {
        writeLbwMem(dstPtr, &value, sizeof(uint32_t));
    }
    else
    {
        _mm_sfence();
        *dstPtr = value;
    }
}