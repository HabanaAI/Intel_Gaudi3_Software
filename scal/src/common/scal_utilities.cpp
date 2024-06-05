#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <pwd.h>
#include "logger.h"
#include "common/pci_ids.h"
#include "scal_utilities.h"
#include "hlthunk.h"

static constexpr uint64_t c_lbw_addr_mask = 0x3f0000ffffffffffULL;
static constexpr uint64_t c_lbw_fd_mask   = 0x00ffff0000000000ULL;

static void accessMem(uint64_t deviceAddress, uint64_t hostAddress, uint32_t size, uint32_t opType)
{
    hl_debug_args args;
    struct hl_debug_params_mem_access memAccess;

    memset(&args, 0, sizeof(args));
    memset(&memAccess, 0, sizeof(memAccess));

    /* Mapping of the memory is with the following format:
    * 64 bit adderss is devided to multiple parts:
    * {2-bits for non canonical, 6 bits of lbw address, 16 bits fd, 40 bit lbw_address}
    */
    int fd = (deviceAddress & c_lbw_fd_mask) >> c_lbw_fd_shift;
    uint64_t readAddress = (deviceAddress & c_lbw_addr_mask);

    memAccess.cfg_address = readAddress;
    memAccess.user_address = hostAddress;
    memAccess.size = size;

    args.input_ptr = (uint64_t)&memAccess;
    args.op = opType;
    args.input_size = sizeof(struct hl_debug_params_mem_access);

    int ret = hlthunk_debug(fd, &args);

    if (ret == -1)
    {
        // TODO: add trace
        assert(0);
    }

}

const char* getHomeFolder()
{
    char *homedir;
    if ((homedir = getenv("HOME")) == NULL)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    return homedir;
}

bool fileExists(const std::string& name)
{
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

bool isSimFD(int fd)
{
    uint32_t deviceId = hlthunk_get_device_id_from_fd(fd);

    if (deviceId == PCI_IDS_INVALID)
    {
        LOG_ERR(SCAL, "Invalid device ID");
        assert(0);
        return 0;
    }
    switch (deviceId)
    {
        case PCI_IDS_GAUDI2_SIMULATOR :
        case PCI_IDS_GAUDI2B_SIMULATOR :
        case PCI_IDS_GAUDI3_SIMULATOR :
        case PCI_IDS_GAUDI3_SIMULATOR_SINGLE_DIE :
        case PCI_IDS_GAUDI2_ARC_SIMULATOR :
        case PCI_IDS_GAUDI2B_ARC_SIMULATOR :
        case PCI_IDS_GAUDI3_ARC_SIMULATOR :
        case PCI_IDS_GAUDI3_ARC_SIMULATOR_SINGLE_DIE :
        {
            return 1;
        }
        default:
        {
            return 0;
        }
    }
}

void* mapLbwMemory(int fd, uint64_t lbwAddress, uint32_t size, uint32_t &allocatedSize)
{
    allocatedSize = size;

    if (!isSimFD(fd))
    {
        uint64_t deviceAddress;
        int ret = hlthunk_get_hw_block(fd, lbwAddress, &allocatedSize, &deviceAddress);
        LOG_INFO(SCAL, "{}: fd={} hlthunk_get_hw_block: lbwAddress: {:#x} allocatedSize: {} size: {} deviceAddress: {:#x}",  __func__, fd, lbwAddress, allocatedSize, size, deviceAddress);
        if (ret != 0)
        {
            LOG_ERR(SCAL, "{}: fd={} hlthunk_get_hw_block() failed. ret={}", __FUNCTION__, fd, ret);
            assert(0);
            return nullptr;
        }
        if (allocatedSize < size)
        {
            LOG_ERR(SCAL, "{}: fd={} hlthunk_get_hw_block() allocated size {} < requested size {}", __FUNCTION__, fd, allocatedSize, size);
            assert(0);
            return nullptr;
        }
        int   mmapFlags = MAP_SHARED ;
        int   prot      = PROT_READ|PROT_WRITE;
        void *ptr = mmap(0, allocatedSize,  prot, mmapFlags, fd, deviceAddress);
        LOG_INFO(SCAL, "{}: fd={} mmap: deviceAddress: {:#x} ptr: {:#x}",  __FUNCTION__, fd, deviceAddress, (uint64_t)ptr);

        if (ptr == MAP_FAILED || ptr == nullptr)
        {
            LOG_ERR(SCAL, "{}: fd={} mmap failed. size: {} deviceAddress: {:#x} errno: {} {}", __FUNCTION__, fd, allocatedSize, deviceAddress, errno, std::strerror(errno));
            assert(0);
            return nullptr;
        }
        return ptr;
    }
    else  // sim mode
    {
        /* mapping of the memory is with the following format:
         * 64 bit adderss is devided to multiple parts:
         * {2-bits for non canonical, 6 bits of lbw address, 16 bits fd, 40 bit lbw_address}
         */
        uint64_t fd64 = ((uint64_t)fd) << c_lbw_fd_shift;
        if (fd64 & c_lbw_addr_mask)
        {
            assert(0);
            return 0;
        }
        if (lbwAddress & ~c_lbw_addr_mask)
        {
            assert(0);
            return 0;
        }
        uint64_t simMapped =  (1uL<<63) | fd64 | lbwAddress;
        return (void*)simMapped;
    }

}


int unmapLbwMemory(void* address, uint32_t size)
{
    uint64_t deviceAddress = (uint64_t)address;
    int ret = 0;

    if (likely(!(IS_SIM_LBW_MEM(deviceAddress))))
    {
        ret =  munmap(address, (size_t)size);
    }
    return ret;

}

/* canonical address ( always virtual):  from sdm documentation
       "In 64-bit mode, if address bits 63 through to the most-significant
       implemented bit by the microarchitecture are set to either all ones
       or all zeros. " assuming 48-bit, safe to check bits 62& bit 63 are
       different in sign */
void readLbwMem( void* dstPtr, volatile void* srcPtr, uint32_t size)
{
    // validate that size is aligned to word (4 bytes)
    assert ((size & 0x00000003U) == 0);
    assert (((uint64_t)srcPtr & 0x0000000000000003UL) == 0);

    // Check if src_ptr is simulator or not
    uint64_t deviceAddress = (uint64_t)srcPtr;

    if (unlikely((IS_SIM_LBW_MEM(deviceAddress))))
    {
        accessMem(deviceAddress, (uint64_t)dstPtr, size, HL_DEBUG_OP_READMEM);
    }
    else
    {
        // Directly read from device
        _mm_mfence();
        // copy from device
        int loopSize = size / sizeof(uint32_t);
        volatile uint32_t* src = (volatile  uint32_t*) srcPtr;
        uint32_t* dst = (uint32_t*) dstPtr;
        for(int i = 0; i < loopSize; i++)
        {
           *dst = *src;
           dst++; src++;
        }
    }

}

void writeLbwMem(volatile void* dstPtr, void* srcPtr, uint32_t size)
{
    // validate that size is aligned to word (4 bytes)
    assert ((size & 0x00000003U) == 0);
    assert (((uint64_t)dstPtr & 0x0000000000000003UL) == 0);

    // Check if dst_ptr is simulator or not
    uint64_t deviceAddress = (uint64_t)dstPtr;

    if (unlikely((IS_SIM_LBW_MEM(deviceAddress))))
    {
        accessMem(deviceAddress, (uint64_t)srcPtr, size, HL_DEBUG_OP_MEMCPY);
    }
    else
    {
        _mm_sfence();
        // Directly write to device
        int loopSize = size / sizeof(uint32_t);
        uint32_t* src = (uint32_t*) srcPtr;
        volatile uint32_t* dst = (volatile  uint32_t*) dstPtr;
        for(int i = 0; i < loopSize; i++)
        {
           *dst = *src;
           dst++; src++;
        }
    }
}
