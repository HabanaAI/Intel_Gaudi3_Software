#include "utils.h"

#include "data_type_utils.h"
#include "define_synapse_common.hpp"
#include "dma_cost_model.h"
#include "filesystem.h"
#include "graph_editor.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "habana_nodes/node_factory.h"
#include "infra/defs.h"
#include "node.h"
#include "settable.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "tpc_node.h"
#include "type_utils.h"
#include "types.h"

#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <tuple>

#include <sys/stat.h>

// The virtual address is composed like this:
//
//
//          |  dma ctx   |       memory id       |                 OFFSET PART                     |
//          |____________|_______________________|_________________________________________________|
//                8                 16                                40
//
// The DMA context part is fixed on 8 bits and is needed due to DMA hardware work-around that uses the 8 MSB of the DMA
// destination address to store context id for the profiler.

#define VIRTUAL_ADDR_DMA_CTX_RESERVED_PART 8
#define VIRTUAL_ADDR_MEMORY_ID_PART        16
#define VIRTUAL_ADDR_OFFSET_PART           (64 - VIRTUAL_ADDR_DMA_CTX_RESERVED_PART - VIRTUAL_ADDR_MEMORY_ID_PART)

#pragma pack(push, 1)
struct VirtualAddress
{
    union
    {
        struct
        {
            int64_t  offset : VIRTUAL_ADDR_OFFSET_PART;
            unsigned memoryID : VIRTUAL_ADDR_MEMORY_ID_PART;
            unsigned dmaCtx : VIRTUAL_ADDR_DMA_CTX_RESERVED_PART;
        };
        deviceAddrOffset raw = 0;
    };
};
#pragma pack(pop)
static_assert(sizeof(VirtualAddress) == sizeof(deviceAddrOffset), "virtual address struct was padded");

uint64_t getMaxMemorySectionID()
{
    return (1 << VIRTUAL_ADDR_MEMORY_ID_PART) - 1;
}

uint64_t getSramMemoryID()
{
    return getMaxMemorySectionID();
}

deviceAddrOffset getVirtualAddressForMemoryID(uint64_t memId, uint64_t offset)
{
    // Checking the offset doesn't run over the virtual address part
    HB_ASSERT((uint64_t(1) << VIRTUAL_ADDR_OFFSET_PART) > offset, "offset {} for tensor is too big ", offset);

    VirtualAddress va {};
    va.memoryID = memId;
    va.offset   = offset;
    return va.raw;
}

uint64_t getMemoryIDFromVirtualAddress(deviceAddrOffset virtualAddr)
{
    VirtualAddress va {};
    va.raw = virtualAddr;
    // Negative offset causes the memory ID to be off by -1
    if (va.offset < 0) ++va.memoryID;
    return va.memoryID;
}

void getSectionInfoFromVirtualAddress(deviceAddrOffset virtualAddr, uint16_t& memId, uint64_t& offset)
{
    VirtualAddress va {};
    va.raw = virtualAddr;
    memId  = va.memoryID;
    // Negative offset causes the memory ID to be off by -1
    if (va.offset < 0) ++memId;
    offset = va.offset;
}

deviceAddrOffset maskOutMemoryID(deviceAddrOffset virtualAddr)
{
    VirtualAddress va {};
    va.raw = virtualAddr;
    if (va.offset < 0)
    {
        // Offset < 0 only happens in TPC descriptors of slices with adjusted offset. The DMA ctx is irrelevant and sign
        // extent is required.
        return va.offset;
    }
    else
    {
        va.memoryID = 0;
        return va.raw;
    }
}

deviceAddrOffset maskOutMemoryID(uint32_t virtualAddrHigh32bits)
{
    ptrToInt fullAddr;

    fullAddr.u32[0]  = 0;
    fullAddr.u32[1]  = virtualAddrHigh32bits;
    fullAddr.u64     = maskOutMemoryID(fullAddr.u64);

    return fullAddr.u32[1]; // return the high part without the memory ID
}

bool assignVirtualAddressToUserPersistentTensor(pTensor t)
{
    deviceAddrOffset addr = getVirtualAddressForMemoryID(t->getMemorySectionID(), t->getMemorySectionOffset());
    if (addr == 0) return false;
    t->setDramOffset(addr);
    LOG_TRACE(GC, "    Assigning virtual address 0x{:x} to user-persistent tensor {}", addr, t->getName());
    return true;
}

std::string_view getMemorySectionNameForMemoryID(uint64_t memId, const TensorPtr& t /*nullptr*/)
{
    if (memId == MEMORY_ID_RESERVED_FOR_WORKSPACE)
    {
        return WORKSPACE_MEMORY_SECTION_NAME;
    }
    else if (memId == MEMORY_ID_RESERVED_FOR_PROGRAM_DATA)
    {
        return PROGRAM_MEMORY_SECTION_NAME;
    }
    else if (memId == getSramMemoryID())
    {
        LOG_ERR(GC, "SRAM memory section has no name (should not be used for patching)");
        return "sram_memory_section_has_no_name";
    }
    else if (t != nullptr)
    {
        return t->getRealTensor(t)->getName(); // unreserved memory IDs are assigned to user persistent tensors
    }
    else
    {
        return "unknown";
    }
}

// The following can only be used for contiguous fields
// For simplicity, there are no verifications
uint32_t getBitFieldValue(uint32_t registerVal, uint32_t fieldOffset, uint32_t numOfBits)
{
    uint32_t fieldMask = ( ( 1 << numOfBits ) - 1 ) << fieldOffset;
    uint32_t regMasked = registerVal & fieldMask;
    uint32_t retField  = regMasked >> fieldOffset;

    return retField;
}
void setBitFieldValue(uint32_t* pRegisterVal, uint32_t fieldOffset, uint32_t numOfBits, uint32_t newVal)
{
    uint32_t fieldMask   = ( ( 1 << numOfBits ) - 1 ) << fieldOffset;
    uint32_t regMasked   = *pRegisterVal & (~fieldMask);
    uint32_t newValField = ( newVal << fieldOffset ) & fieldMask;

    *pRegisterVal = regMasked | newValField;
}

unsigned next_power_of_two(unsigned a)
{
    if (a > (unsigned)(1 << 31)) return 0;
    unsigned ret = 1;
    while (ret <= a)
    {
        ret <<= 1;
    }
    return ret;
}

const char* getDeviceName(HabanaDeviceType type)
{
    switch (type)
    {
    case DEVICE_MME: return "DEVICE_MME";
    case DEVICE_TPC: return "DEVICE_TPC";
    case DEVICE_ROTATOR: return "DEVICE_ROTATOR";
    case DEVICE_DMA_HOST_DEVICE: return "DEVICE_DMA_HOST_DEVICE";
    case DEVICE_DMA_DEVICE_HOST: return "DEVICE_DMA_DEVICE_HOST";
    case DEVICE_DMA_DRAM_HOST: return "DEVICE_DMA_DRAM_HOST";
    case DEVICE_DMA_SRAM_HOST: return "DEVICE_DMA_SRAM_HOST";
    case DEVICE_DMA_SRAM_DRAM: return "DEVICE_DMA_SRAM_DRAM";
    case DEVICE_DMA_PREFETCH_STATIC_TENSORS: return "DEVICE_DMA_PREFETCH_STATIC_TENSORS";
    case DEVICE_DMA_PREFETCH_ACTIVATIONS: return "DEVICE_DMA_PREFETCH_ACTIVATIONS";
    case DEVICE_DMA_DRAM_SRAM: return "DEVICE_DMA_DRAM_SRAM";
    case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL: return "DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL";
    case DEVICE_COMPLETION_QUEUE: return "DEVICE_COMPLETION_QUEUE";
    case DEVICE_CME: return "DEVICE_CME";
    default: return "UNKNOWN";
    }
}

double calculateExp(double num)
{
    double exp = 0;
    if (num <= 0)
    {
        return 0;
    }

    exp = log2(num);

    double res;
    res = modf(exp, &exp);
    if (res)
    {
        return 0;
    }

    return exp;
}

uint8_t findFirstSetBit(unsigned mask)
{
    uint8_t bitLocation = 0;
    if (mask == 0)
    {
        LOG_ERR(SYN_API,"Invalid mask");
        return 0;
    }
    while ((mask & 0x1) == 0)
    {
        mask >>= 1;
        bitLocation++;
    }
    return bitLocation;
}

uint32_t turnOffMSBs(uint32_t mask, unsigned numBitsToUnset)
{
    // Look for the left-most N bits that are turned on and turn them off (N is numBitsToUnset)
    unsigned unsetCount = numBitsToUnset;
    uint32_t ret        = mask;
    uint32_t checker    = 0x80000000;
    while (ret && unsetCount)
    {
        if (ret & checker)
        {
            ret &= (~checker);
            unsetCount--;
        }
        checker >>= 1;  // unsigned right-shift will zero fill on the left
    }
    return ret;
}

void dumpBufferToFile(const char* fileName, const void* buf, unsigned size)
{
    FILE* h = nullptr;
    if ((h = fopen(fileName, "wb")) != nullptr)
    {
        fwrite(buf, 1, size, h);
        fclose(h);
    }
}

// Bytes look up table for the crc32
static constexpr uint32_t LUT[256] = {
    0x0,        0xad0424f3, 0xf70c6d15, 0x5a0849e6, 0x431cfed9, 0xee18da2a, 0xb41093cc, 0x1914b73f, 0x8639fdb2,
    0x2b3dd941, 0x713590a7, 0xdc31b454, 0xc525036b, 0x68212798, 0x32296e7e, 0x9f2d4a8d, 0xa177df97, 0xc73fb64,
    0x567bb282, 0xfb7f9671, 0xe26b214e, 0x4f6f05bd, 0x15674c5b, 0xb86368a8, 0x274e2225, 0x8a4a06d6, 0xd0424f30,
    0x7d466bc3, 0x6452dcfc, 0xc956f80f, 0x935eb1e9, 0x3e5a951a, 0xefeb9bdd, 0x42efbf2e, 0x18e7f6c8, 0xb5e3d23b,
    0xacf76504, 0x1f341f7,  0x5bfb0811, 0xf6ff2ce2, 0x69d2666f, 0xc4d6429c, 0x9ede0b7a, 0x33da2f89, 0x2ace98b6,
    0x87cabc45, 0xddc2f5a3, 0x70c6d150, 0x4e9c444a, 0xe39860b9, 0xb990295f, 0x14940dac, 0xd80ba93,  0xa0849e60,
    0xfa8cd786, 0x5788f375, 0xc8a5b9f8, 0x65a19d0b, 0x3fa9d4ed, 0x92adf01e, 0x8bb94721, 0x26bd63d2, 0x7cb52a34,
    0xd1b10ec7, 0x72d31349, 0xdfd737ba, 0x85df7e5c, 0x28db5aaf, 0x31cfed90, 0x9ccbc963, 0xc6c38085, 0x6bc7a476,
    0xf4eaeefb, 0x59eeca08, 0x3e683ee,  0xaee2a71d, 0xb7f61022, 0x1af234d1, 0x40fa7d37, 0xedfe59c4, 0xd3a4ccde,
    0x7ea0e82d, 0x24a8a1cb, 0x89ac8538, 0x90b83207, 0x3dbc16f4, 0x67b45f12, 0xcab07be1, 0x559d316c, 0xf899159f,
    0xa2915c79, 0xf95788a,  0x1681cfb5, 0xbb85eb46, 0xe18da2a0, 0x4c898653, 0x9d388894, 0x303cac67, 0x6a34e581,
    0xc730c172, 0xde24764d, 0x732052be, 0x29281b58, 0x842c3fab, 0x1b017526, 0xb60551d5, 0xec0d1833, 0x41093cc0,
    0x581d8bff, 0xf519af0c, 0xaf11e6ea, 0x215c219,  0x3c4f5703, 0x914b73f0, 0xcb433a16, 0x66471ee5, 0x7f53a9da,
    0xd2578d29, 0x885fc4cf, 0x255be03c, 0xba76aab1, 0x17728e42, 0x4d7ac7a4, 0xe07ee357, 0xf96a5468, 0x546e709b,
    0xe66397d,  0xa3621d8e, 0xe5a62692, 0x48a20261, 0x12aa4b87, 0xbfae6f74, 0xa6bad84b, 0xbbefcb8,  0x51b6b55e,
    0xfcb291ad, 0x639fdb20, 0xce9bffd3, 0x9493b635, 0x399792c6, 0x208325f9, 0x8d87010a, 0xd78f48ec, 0x7a8b6c1f,
    0x44d1f905, 0xe9d5ddf6, 0xb3dd9410, 0x1ed9b0e3, 0x7cd07dc,  0xaac9232f, 0xf0c16ac9, 0x5dc54e3a, 0xc2e804b7,
    0x6fec2044, 0x35e469a2, 0x98e04d51, 0x81f4fa6e, 0x2cf0de9d, 0x76f8977b, 0xdbfcb388, 0xa4dbd4f,  0xa74999bc,
    0xfd41d05a, 0x5045f4a9, 0x49514396, 0xe4556765, 0xbe5d2e83, 0x13590a70, 0x8c7440fd, 0x2170640e, 0x7b782de8,
    0xd67c091b, 0xcf68be24, 0x626c9ad7, 0x3864d331, 0x9560f7c2, 0xab3a62d8, 0x63e462b,  0x5c360fcd, 0xf1322b3e,
    0xe8269c01, 0x4522b8f2, 0x1f2af114, 0xb22ed5e7, 0x2d039f6a, 0x8007bb99, 0xda0ff27f, 0x770bd68c, 0x6e1f61b3,
    0xc31b4540, 0x99130ca6, 0x34172855, 0x977535db, 0x3a711128, 0x607958ce, 0xcd7d7c3d, 0xd469cb02, 0x796deff1,
    0x2365a617, 0x8e6182e4, 0x114cc869, 0xbc48ec9a, 0xe640a57c, 0x4b44818f, 0x525036b0, 0xff541243, 0xa55c5ba5,
    0x8587f56,  0x3602ea4c, 0x9b06cebf, 0xc10e8759, 0x6c0aa3aa, 0x751e1495, 0xd81a3066, 0x82127980, 0x2f165d73,
    0xb03b17fe, 0x1d3f330d, 0x47377aeb, 0xea335e18, 0xf327e927, 0x5e23cdd4, 0x42b8432,  0xa92fa0c1, 0x789eae06,
    0xd59a8af5, 0x8f92c313, 0x2296e7e0, 0x3b8250df, 0x9686742c, 0xcc8e3dca, 0x618a1939, 0xfea753b4, 0x53a37747,
    0x9ab3ea1,  0xa4af1a52, 0xbdbbad6d, 0x10bf899e, 0x4ab7c078, 0xe7b3e48b, 0xd9e97191, 0x74ed5562, 0x2ee51c84,
    0x83e13877, 0x9af58f48, 0x37f1abbb, 0x6df9e25d, 0xc0fdc6ae, 0x5fd08c23, 0xf2d4a8d0, 0xa8dce136, 0x5d8c5c5,
    0x1ccc72fa, 0xb1c85609, 0xebc01fef, 0x46c43b1c};

uint32_t crc32(const void* buff, uint64_t size, uint32_t prevCrc /*= 0 */)
{
    HB_ASSERT(size > 0, "non-positive buffer size");
    uint8_t updatedRemainder   = 0;
    uint32_t crc               = prevCrc;
    const char* kerBuff        = reinterpret_cast<const char*>(buff);
    for (uint64_t byte = 0; byte < size; ++byte)
    {
        updatedRemainder       = kerBuff[byte] ^ (crc >> 24);
        crc                    = LUT[updatedRemainder] ^ (crc << 8);
    }
    return crc;
}


#ifdef _WIN32

libHandle LoadSharedObject(const char* name)
{
    return LoadLibrary(name);
}

fnHandle GetFunction(libHandle handle, const char* name)
{
    return GetProcAddress(handle, name);
}

void UnloadSharedObject(libHandle handle)
{
    FreeLibrary(handle);
}

void  SafeMemcpy(void* dst, size_t dstSize, const void* src, size_t srcSize)
{
    memcpy_s(dst, dstSize, src, srcSize);
}

#else

libHandle LoadSharedObject(const char* name)
{
    return dlopen(name, RTLD_LAZY);
}

fnHandle GetFunction(libHandle handle, const char* name)
{
    return dlsym(handle, name);
}

void UnloadSharedObject(libHandle handle)
{
    dlclose(handle);
}

void  SafeMemcpy(void* dst, size_t dstSize, const void* src, size_t srcSize)
{
    //Todo: what is the linux equivalent?
    memcpy(dst, src, std::min(dstSize, srcSize));
}

void formatToFinalPrint(std::string& format, uint32_t* val, std::vector<std::string>& finalPrints )
{
    if (format.empty())
    {
        return;
    }

    unsigned printBuffSize = GCFG_TPC_PRINTF_TENSOR_SIZE.value();
    char *finalPrint = new char[printBuffSize];

    if (format.find("%f") != std::string::npos || format.find("%F") != std::string::npos)
    {
        float* floatVal = reinterpret_cast<float*>(val);
        sprintf(finalPrint, format.c_str(), *floatVal);
        finalPrints.push_back(std::string(finalPrint));
        delete[] finalPrint;
        return;
    }
    sprintf(finalPrint, format.c_str(), *val);
    finalPrints.push_back(std::string(finalPrint));
    delete[] finalPrint;
}

void parsePrintf(uint32_t* buff, std::vector<std::string>& finalPrints)
{
    HB_ASSERT(buff != nullptr, "invalid printf buffer to parse");
    std::string buffOverflowMsg         = "Printf buff overflow, some of the printf commands in the kernel will not appear in output,\n"
                                          "use less or shorter prints in the kernel";
    const uint32_t NEW_PRINTF           = 0xcdcdcdcd;
    const uint32_t END_PRINTF           = 0xffffffff;
    const unsigned BUFF_ENTRY_SIZE      = sizeof(uint32_t);
    // We read till the end of the buffer as we may have multiple sub-buffers representing different TPC activations
    std::string str;
    uint32_t val = 0;
    unsigned count = 0;
    bool bStrStart = false;

    while (count < GCFG_TPC_PRINTF_TENSOR_SIZE.value())
    {
        // 0xffffffff marks the end of string - do not process unneeded chars beyond this point
        if (*buff == END_PRINTF && bStrStart)
        {
            formatToFinalPrint(str, &val, finalPrints);
            str.clear();
            bStrStart = false;
        }
        else if (*buff == NEW_PRINTF)
        {
            bStrStart = true;
            formatToFinalPrint(str, &val, finalPrints);
            str.clear();
            buff++;
            count += BUFF_ENTRY_SIZE;
            val = *buff;
        }
        else if (bStrStart)
        {
            char* nextCharToRead = reinterpret_cast<char*>(buff);
            str.append(nextCharToRead, BUFF_ENTRY_SIZE);
        }
        buff++;
        count += BUFF_ENTRY_SIZE;
    }
    formatToFinalPrint(str, &val, finalPrints);
    if (count > GCFG_TPC_PRINTF_TENSOR_SIZE.value())
    {
        LOG_WARN(SYN_TPC_PRINT, "{}", buffOverflowMsg);
    }
}

#endif

// TODO:  Move spatialIndex to TOffset array [SW-117362]
void findSpatialIndex(const TSize originalSize[Tensor::c_tensorMaxDim],
                      int spatialIndex[Tensor::c_tensorMaxDim - 1],
                      TSize spatialSize)
{
    TOffset spatialIndexTemp[Tensor::c_tensorMaxDim - 1];
    findIndex(originalSize + 1, Tensor::c_tensorMaxDim - 1, spatialSize, spatialIndexTemp);
    castNcopy(spatialIndex, spatialIndexTemp, Tensor::c_tensorMaxDim - 1);
}

void findIndex(const TSize* sizes, uint32_t dimNum, uint64_t sizePos, TOffset* outIndex)
{
    for (unsigned int dim = dimNum - 1; dim > 0; --dim)
    {
        uint64_t dimMul = multiplyElements(sizes, sizes + dim);
        outIndex[dim] = sizePos / dimMul;
        sizePos %= dimMul;
    }
    HB_ASSERT(sizePos < sizes[0], "sizePos bigger than {}", sizes[0]);
    outIndex[0] = sizePos;
}

TSize calcAbsSize(const TSize* sizes, const TOffset* index, unsigned int dimNum)
{
    TSize ret = 0;
    for (unsigned int dim = 0; dim < dimNum; ++dim)
    {
        ret += index[dim] * multiplyElements(sizes, sizes + dim);
    }

    return ret;
}

int convInputDimSize(unsigned int outputSize,
                     unsigned int weightSize,
                     unsigned int stride,
                     int padding,
                     unsigned int dilation)
{
    int ret = ((int)outputSize - 1) * (int)stride - padding + ((int)weightSize + ((int)weightSize - 1) * ((int)dilation - 1));
    if(ret < 0)
    {
        LOG_WARN(GC, "Calculated conv input size is negative");
    }
    return ret;
}

int convOutputDimSize(TSize        inputSize,
                      unsigned int weightSize,
                      unsigned int stride,
                      int padding,
                      unsigned int dilation)
{
    int effectiveFilterSize = ((int)weightSize - 1) * (int)dilation + 1;
    int ret                 = ((int)inputSize + (int)padding - effectiveFilterSize + (int)stride) / (int)stride;
    if(ret < 0)
    {
        LOG_WARN(GC, "Calculated conv output size is negative");
    }
    return ret;
}

TSize convOutputDimSizeSamePadding(TSize inputSize, unsigned int stride)
{
    return (inputSize + stride - 1) / stride;
}

void convPaddingSamePadding(TSize         inputSize,
                            unsigned int  weightSize,
                            unsigned int  dilation,
                            unsigned int  stride,
                            unsigned int& before,
                            unsigned int& after)
{
    TSize    outputSize          = convOutputDimSizeSamePadding(inputSize, stride);
    unsigned effectiveFilterSize = (weightSize - 1) * dilation + 1;
    unsigned totalPadding = std::max(0, static_cast<int>((outputSize - 1) * stride + effectiveFilterSize - inputSize));
    before                = totalPadding / 2;
    after                 = totalPadding - before;
}

void copyData(void* dstData, void* srcData, synDataType dstType, synDataType srcType, uint64_t size)
{
    switch(srcType)
    {
        // a pair of 4 bits data type elements is arranged in a byte
        case syn_type_int4:
        case syn_type_fixed:
        {
            copyDataFrom<int8_t>( dstData, srcData, dstType, size );
            break;
        }
        // a pair of 4 bits data type elements is arranged in a byte
        case syn_type_uint4:
        case syn_type_uint8:
        {
            copyDataFrom<uint8_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_int16:
        {
            copyDataFrom<int16_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_uint16:
        {
            copyDataFrom<uint16_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_int32:
        {
            copyDataFrom<int32_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_uint32:
        {
            copyDataFrom<uint32_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_single:
        {
            copyDataFrom<int32_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_fp16:
        case syn_type_bf16:
        {
            copyDataFrom<int16_t>( dstData, srcData, dstType, size );
            break;
        }
        case syn_type_na:
        default:
        {
            HB_ASSERT(false, "Invalid data type");
        }
    }
}

bool NodeTypeMatchingFunc(const NodePtr& a, const NodePtr& b)
{
    if (a->getNodeType() != b->getNodeType())
    {
        return false;
    }
    if (HabanaGraph::runsOnTPC(a) && HabanaGraph::runsOnTPC(b))
    {
        // compare the guids without the data type suffix (e.g. "add_f32" and "add_i8" are the same type)
        std::string_view guidA = static_cast<TPCNode&>(*a).getGUIDWithoutDtype();
        std::string_view guidB = static_cast<TPCNode&>(*b).getGUIDWithoutDtype();
        return guidA == guidB;
    }
    return true;  // nodes are not TPC, so the simple type check at the beginning was enough
}

bool NodeTypeMatchingFuncWithPrecision(const NodePtr& a, const NodePtr& b)
{
    if (a->getNodeType() != b->getNodeType())
    {
        return false;
    }
    if (HabanaGraph::runsOnTPC(a) && HabanaGraph::runsOnTPC(b))
    {
        // compare the entire guid with data type (e.g. "add_f32" and "add_i8" are different types)
        std::string guidA = static_cast<TPCNode&>(*a).getGUID();
        std::string guidB = static_cast<TPCNode&>(*b).getGUID();
        return guidA == guidB;
    }
    return true;  // nodes are not TPC, so the simple type check at the beginning was enough
}

float bToKb(unsigned bytes)
{
    return (float)bytes / (1024.f);
}

float bToMb(unsigned bytes)
{
    return (float)bytes / (1024.f * 1024.f);
}

double bToMb(uint64_t bytes)
{
    return (double)bytes / (1024.f * 1024.f);
}

std::vector<std::string> splitString(const std::string& str, char delimiter)
{
    size_t current, previous = 0;
    std::vector<std::string> res;

    current = str.find(delimiter);

    while (current != std::string::npos)
    {
        res.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delimiter, previous);
    }

    res.push_back(str.substr(previous, current - previous));

    return res;
}

// Here size is in elements (each is int32)
static const unsigned tpcVectorSize = 64;

static bool getVectorsNumInTpcGroup( unsigned& vectorAmountInTpcGroup, synDataType targetDataType )
{
    bool status = true;

    switch( targetDataType )
    {
        case syn_type_fixed:
        case syn_type_uint8:
        {
            vectorAmountInTpcGroup = 4;
        }
        break;

        case syn_type_int16:
        case syn_type_uint16:
        {
            vectorAmountInTpcGroup = 2;
        }
        break;

        case syn_type_int64:
        case syn_type_uint64:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_single:
        case syn_type_bf16:
        case syn_type_na:
        case syn_type_max:
        case syn_type_int4:
        case syn_type_uint4:
        case syn_type_fp16:
        case syn_type_ufp16:
        case syn_type_tf32:
        case syn_type_hb_float:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
        {
            status = false;
        }
        break;
    }

    return status;
}

bool swizzle(int32_t* output, int32_t* input, unsigned inputSize, unsigned outputSize, synDataType targetDataType)
{
    unsigned vectorAmountInTpcGroup = 0;

    if( !getVectorsNumInTpcGroup( vectorAmountInTpcGroup, targetDataType ) )
    {
        return false;
    }

    unsigned groupSize = tpcVectorSize * vectorAmountInTpcGroup;
    unsigned groupMask = groupSize - 1;

    for (unsigned i = 0; i < inputSize ; i++)
    {
        unsigned inputIndexModVecSize = i % groupSize;
        unsigned groupOffset          = i & ~groupMask;
        unsigned inGroupOffset        = tpcVectorSize * (inputIndexModVecSize % vectorAmountInTpcGroup);
        unsigned inVectorOffset       = inputIndexModVecSize / vectorAmountInTpcGroup;

        unsigned coord                = groupOffset + inGroupOffset + inVectorOffset;

        if( coord < outputSize )
        {
            output[coord] = input[i];
        }
        else
        {
            return false;
        }
    }

    return true;
}

bool getSwizzleOutputNumOfElement(unsigned& swizzleOutputNumOfElements, unsigned numOfInputElements, synDataType targetDataType)
{
    unsigned vectorAmountInTpcGroup = 0;

    if( !getVectorsNumInTpcGroup( vectorAmountInTpcGroup, targetDataType ) )
    {
        return false;
    }

    unsigned groupSize = tpcVectorSize * vectorAmountInTpcGroup;
    swizzleOutputNumOfElements = ( 1 + ((numOfInputElements - 1) / groupSize) ) * groupSize;

    return true;
}

std::string_view getHighestGUIDDataType(const std::vector<std::string>& dtypes)
{
    // validate input data types
    for (const std::string& dtype : dtypes)
    {
        if (!isGUIDDataTypeSupported(dtype))
        {
            // input data type was not found in known data types
            LOG_WARN(GC, "{}: {} is an invalid data type", HLLOG_FUNC, dtype);
        }
    }

    // GUID data types ordered by "higher accuracy"
    static constexpr std::array<std::string_view, 15> orderedGUIDDataTypes =
        {"u64", "i64", "u32", "i32", "f32", "f16", "bf16", "u16", "i16", "u8", "i8", "f8", "hf8", "u4", "i4"};

    // find highest data type
    for (const std::string_view& dtype : orderedGUIDDataTypes)
    {
        if (std::find(dtypes.begin(), dtypes.end(), dtype) != dtypes.end())
        {
            return dtype;
        }
    }

    if (dtypes.empty())
    {
        LOG_WARN(GC, "{}: got an empty vector", HLLOG_FUNC);
    }
    return "";
}

synDataType getHighestGUIDDataType(const std::vector<synDataType>& dtypes)
{
    std::vector<std::string> dtypesSuffixFormat;
    dtypesSuffixFormat.reserve(dtypes.size());
    for (const synDataType& dtype : dtypes)
    {
        dtypesSuffixFormat.emplace_back(getDtypeSuffixFromSynDataType(dtype));
    }
    std::string_view ret = getHighestGUIDDataType(dtypesSuffixFormat);
    return getSynDataTypeFromDtypeSuffix(ret);
}

// divide given number of samples to given number of chunks. assign the bigger chunks to be at the beginning, unless
// firstBiggerIndex parameter is given (in that case use it to determine the first index with bigger value), i.e:
// if numSamples = 6, numOfChunks = 4 and firstBiggerIndex = 0 (default), the result is 2,2,1,1
// if numSamples = 6, numOfChunks = 4 and firstBiggerIndex = 1, the result is 1,2,2,1
// if numSamples = 6, numOfChunks = 4 and firstBiggerIndex = 3, the result is 2,1,1,2
std::vector<TSize> splitToChunks(TSize    numSamples,
                                 unsigned numOfChunks,
                                 unsigned firstBiggerIndex,
                                 unsigned numOfPhysicalEngs)
{
    HB_ASSERT(firstBiggerIndex < numOfChunks, "large chunk index is bigger than the total chunk number");
    std::vector<TSize> ret(numOfChunks);
    TSize chunk = numSamples / numOfChunks;

    // Optimized logical ROI split. Make sure all chuncks are divided by the number of physical engines
    if (GCFG_ENABLE_OPTIMIZED_LOGICAL_ROI_SPLIT.value()          // GCFG to control the use of optimization
        && (numOfPhysicalEngs != 0)                              // Set to 0 when we should not use the optimization (curremtly for TPC only)
        && (numOfChunks != 1)                                    // if numOfChunks == 1 there is no way to optimize the split
        && (numSamples / (numOfChunks - 1) > numOfPhysicalEngs)) // have at least numOfChunks - 1 chuncks that can be optimized
    {
        // fullChunks are units of numOfPhysicalEngs within numSamples
        TSize fullChunks = numSamples / numOfPhysicalEngs;
        unsigned numOfBiggerRois = fullChunks % numOfChunks;
        unsigned numOfSmallerRois = numOfChunks - (numOfBiggerRois);

        for (unsigned chunkIdx = 0; chunkIdx < numOfChunks; ++chunkIdx)
        {
            TSize chunkSize = 0;
            if (chunkIdx == numOfChunks - 1) // Last chunk. Need to take what has been left from numSamples
            {
                chunkSize = numSamples;
            }
            else if (chunkIdx < numOfBiggerRois)
            {
                chunkSize = ((fullChunks / numOfChunks) + 1) * numOfPhysicalEngs;
            }
            else // numOfSmallerRois
            {
                chunkSize = ((fullChunks / numOfChunks)) * numOfPhysicalEngs;
            }
            ret[chunkIdx] = chunkSize;
            numSamples -= chunkSize;
        }
    }
    else // Original logical ROI split. Do not consider the number of physical engines.
    {
        TSize remainder = numSamples % numOfChunks;
        for (unsigned chunkIdx = 0; chunkIdx < numOfChunks; ++chunkIdx)
        {
            TSize chunkSize = ((numOfChunks + chunkIdx - firstBiggerIndex) % numOfChunks) < remainder ? chunk + 1 : chunk;
            ret[chunkIdx] = chunkSize;
            numSamples -= chunkSize;
        }
    }
    HB_ASSERT(numSamples == 0, "splitToChunks - wrong split!");
    return ret;
}

// divide given number of samples to given number of chunks.
// in addition, fill firstMinimalIndex with the first index where the chunk size is smaller
std::vector<TSize> splitToChunksWithIndexes(TSize     numSamples,
                                            unsigned  numOfChunks,
                                            unsigned  firstBiggerIndex,
                                            unsigned& firstMinimalIndex)
{
    firstBiggerIndex %= numOfChunks;  // safety for case that can happen when we split differently on different rois
    TSize numOfBiggerChunks = numSamples % numOfChunks;
    firstMinimalIndex = (numOfBiggerChunks + firstBiggerIndex) % numOfChunks;
    return splitToChunks(numSamples, numOfChunks, firstBiggerIndex);
}


bool findSingleNonOneDim(pTensor &weightsTensor, unsigned &nonOneDim)
{
    unsigned nonOneCounter = 0;
    TSize weightSize[Tensor::c_tensorMaxDim];
    weightsTensor->getAllSizesInElements(weightSize, Tensor::c_tensorMaxDim);

    for (unsigned i = 0; i < Tensor::c_tensorMaxDim; i++)
    {
        if (weightSize[i] != 1)
        {
            nonOneCounter++;
            nonOneDim = i;
        }
    }

    if (nonOneCounter != 1)
    {
        return false;
    }
    return true;
}

bool isTpcMemcpy(const NodePtr& n)
{
    return n->getNodeType() == Node::TYPE_USER && static_cast<const TPCNode*>(n.get())->isGuidPrefix("memcpy");
}

bool isMemcpy(const Node& n)
{
    switch (n.getNodeType())
    {
        case Node::TYPE_USER:
        {
            return dynamic_cast<const TPCNode&>(n).isGuidPrefix("memcpy");
        }
        case Node::TYPE_DMA:
        {
            const auto& dmaNode = static_cast<const DMANode&>(n);
            return !dmaNode.isDynamicMemoryOp() && dmaNode.getOpType() == DMA_OP_COPY;
        }
        case Node::TYPE_MEMCOPY:
            return true;
        default:
            return false;
    }
}

bool fitsInBits(unsigned val, unsigned bits)
{
    HB_ASSERT(bits < 32, "maximum size is 32 bits");
    return (val < (unsigned)(1 << bits));
}

unsigned countSetBits(uint64_t val, unsigned numOfBits)
{
    return std::bitset<64>(val & (UINT64_MAX >> (NUM_BITS(uint64_t) - numOfBits))).count();
}

bool isHostDma(HabanaDeviceType type)
{
    return ((type == DEVICE_DMA_HOST_DEVICE) ||
            (type == DEVICE_DMA_SRAM_HOST) ||
            (type == DEVICE_DMA_DEVICE_HOST) ||
            (type == DEVICE_DMA_DRAM_HOST));
}

bool isCompletionQueue(HabanaDeviceType type)
{
    return (type == DEVICE_COMPLETION_QUEUE);
}

void copyString(std::string_view input, char* dst, size_t dst_size)
{
    strncpy(dst, input.data(), dst_size - 1);
    dst[dst_size - 1] = '\0';
}

void copyStringSafe(std::string_view input, char* dst, size_t dst_size, size_t maxSize)
{
    // copy at most maxSize characters
    unsigned actualCopyLength = dst_size <= maxSize ? dst_size : maxSize;
    copyString(input, dst, actualCopyLength);
}

unsigned  calcPaddingSize(uint64_t size, uint64_t alignment)
{
    return (size % alignment) == 0 ? 0 : alignment - (size % alignment);
}

unsigned  alignSizeDown(unsigned size, uint64_t alignment)
{
    return (size % alignment) == 0 ? size : (size / alignment) * alignment;
}

unsigned  alignSizeUp(unsigned size, uint64_t alignment)
{
    return (size % alignment) == 0 ? size : ((size / alignment) + 1) * alignment;
}

bool copyTensorData(pTensor& dest, pTensor& src)
{
    if (src->getQuantizationParams().m_qDataType != dest->getQuantizationParams().m_qDataType)
    {
        LOG_ERR(GC, "Failed to copy tensor data due to different data types "
                      "destNode={} destType={} != secNode={} srcType={}",
                dest->getName(), dest->getQuantizationParams().m_qDataType,
                src->getName(), src->getQuantizationParams().m_qDataType);
        return false;
    }

    if (src->getDenseSizeInElements() != dest->getDenseSizeInElements())
    {
        LOG_ERR(GC, "Failed to copy tensor data due to different number of elements "
                      "destNode={} destNumElements={} != secNode={} srcNumElements={}",
                dest->getName(), dest->getDenseSizeInElements(),
                src->getName(), src->getDenseSizeInElements());
        return false;
    }

    char* srcData = src->getData();

    uint64_t numElements = src->getDenseSizeInElements();
    int elementSizeInBytes = src->getElementSizeInBytes();
    int sizeInBytes = numElements * elementSizeInBytes;
    char* destData = new char[sizeInBytes];

    memcpy(destData, srcData, sizeInBytes);
    dest->bind(destData, true);
    return true;
}


int16_t castFloatToBFloat16(float floatVal, bool bigEndian /*= false*/)
{
    int16_t retVal;
    if (bigEndian)
    {
        memcpy(&retVal, &floatVal, sizeof retVal);
    }
    else
    {
        memcpy(&retVal, reinterpret_cast<char*>(&floatVal) + sizeof(floatVal) - sizeof(retVal), sizeof(retVal));
    }
    return retVal;
}

void getConvolutionSize(const SizeArray&              xSize,
                        uint32_t                      yChannels,
                        const synConvolutionParamsV2& convParams,
                        SizeArray&                    wSizeOut,
                        SizeArray&                    ySizeOut)
{
    wSizeOut = {yChannels, xSize[0], convParams.kW, convParams.kH, 1};
    ySizeOut = {yChannels,
                convOutputDimSize(xSize[1], convParams.kW, convParams.dW, convParams.padL + convParams.padR, convParams.dilW),
                convOutputDimSize(xSize[2], convParams.kH, convParams.dH, convParams.padT + convParams.padB, convParams.dilH),
                xSize[3], 1};
}

unsigned getNumEnginesForDeviceType(HabanaDeviceType deviceType, const HalReader& halReader)
{
    switch(deviceType)
    {
        case DEVICE_MME:
            return halReader.getNumMmeEngines();
        case DEVICE_TPC:
            return halReader.getNumTpcEngines();
        case DEVICE_DMA_HOST_DEVICE:
            return 1;
        case DEVICE_DMA_DEVICE_HOST:
            return 1;
        case DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL:
            return halReader.getNumInternalDmaEngines();
        case DEVICE_ROTATOR:
            return halReader.getNumRotatorEngines();
        case DEVICE_COMPLETION_QUEUE:
            return 1;
        case DEVICE_CME:
            return 0;
        default:
            HB_ASSERT(false, "Device type is not supported");
            return 0;
    }
}

// expand tensor with size [X,Y,Z,W] into [X,1,Y,Z,W] ('fillValue' is inserted in dim)
TensorPtr createExpandedTensor(const TensorPtr& tensor, unsigned dim, unsigned fillValue)
{
    HB_ASSERT(tensor->getDim() <= Tensor::c_tensorMaxNDim - 1, "unable to expand {}D tensor", tensor->getDim());
    // This cloned tensor will be used as intermediate -> no need to copy the data
    TensorPtr  expandedTensor = tensor->clone(false, false, false);
    expandedTensor->unsetPermutation();
    NSizeArray expandMaxSize;
    NSizeArray expandMinSize;
    expandMaxSize.fill(fillValue);
    expandMinSize.fill(fillValue);

    for (unsigned i = 0; i < dim; ++i)
    {
        expandMaxSize[i] = tensor->getSizeInElements(i);
        expandMinSize[i] = tensor->getMinimalSizeInElements(i);
    }
    for (unsigned i = dim; i < tensor->getDim(); ++i)
    {
        expandMaxSize[i + 1] = tensor->getSizeInElements(i);
        expandMinSize[i + 1] = tensor->getMinimalSizeInElements(i);
    }
    expandedTensor->reshape(tensor->getDim() + 1, expandMaxSize.data(), nullptr, expandMinSize.data());
    return expandedTensor;
}

std::tuple<TensorPtr, NodePtr> expandTensor(const TensorPtr& tensor, unsigned dim)
{
    TensorPtr out = createExpandedTensor(tensor, dim);
    auto      guid =
        tensor->isShapeTensor() ? NodeFactory::expandDimsShapeNodeTypeName : NodeFactory::expandDimsNodeTypeName;
    NodePtr expand = NodeFactory::createNode({tensor}, {out}, &dim, guid, tensor->getName() + "_expand_dims");
    return std::tie(out, expand);
}

std::tuple<TensorPtr, NodePtr> expandShapeTensorWithValue(const TensorPtr& expandIn, unsigned dim, TSize fillValue)
{
    SifMergeShapesMetadata params;
    params.outputDim = expandIn->getDim() + 1;
    params.fillValue = fillValue;
    for (unsigned i = 0; i < expandIn->getDim() + 1; i++)
    {
        params.dimMap[i].inputIdx = i == dim ? -1 : 0;
        params.dimMap[i].dimIdx   = i - 1;
    }
    TensorPtr expandShapeOut = createExpandedTensor(expandIn, dim /* dim */, fillValue /* fill value */);
    NodePtr   expandNode     = NodeFactory::createNode({expandIn},
                                                       {expandShapeOut},
                                                 &params,
                                                 sizeof(params),
                                                 NodeFactory::mergeShapesNodeTypeName,
                                                 fmt::format("expand_{}", expandIn->getName()));
    return std::tie(expandShapeOut, expandNode);
}

// squeeze tensor with size [X,1,Y,Z,W] into [X,Y,Z,W] (1 is inserted in dim)
TensorPtr createSqueezedTensor(const TensorPtr& tensor, const Settable<unsigned> dim)
{
    HB_ASSERT(!dim.is_set() || tensor->getSizeInElements(dim.value()) == 1,
              "{}: size in dim {} is {} but must be 1",
              tensor->getName(),
              dim.value(),
              tensor->getSizeInElements(dim.value()));
    // This cloned tensor will be used as intermediate -> no need to copy the data
    TensorPtr  squeezedTensor = tensor->clone(false, false, false);
    NSizeArray squeezeMaxSize;
    NSizeArray squeezeMinSize;
    squeezeMaxSize.fill(1);
    squeezeMinSize.fill(1);
    unsigned newDim = 0;
    if (dim.is_set())
    {
        for (; newDim < dim.value(); ++newDim)
        {
            squeezeMaxSize[newDim] = tensor->getSizeInElements(newDim);
            squeezeMinSize[newDim] = tensor->getMinimalSizeInElements(newDim);
        }
        for (; newDim < tensor->getDim() - 1; ++newDim)
        {
            squeezeMaxSize[newDim] = tensor->getSizeInElements(newDim + 1);
            squeezeMinSize[newDim] = tensor->getMinimalSizeInElements(newDim + 1);
        }
    }
    else
    {
        for (unsigned i = 0; i < tensor->getDim(); ++i)
        {
            if (tensor->getSizeInElements(i) != 1)
            {
                squeezeMaxSize[newDim] = tensor->getSizeInElements(i);
                squeezeMinSize[newDim] = tensor->getMinimalSizeInElements(i);
                ++newDim;
            }
        }
    }
    squeezedTensor->reshape(newDim, squeezeMaxSize.data(), nullptr, squeezeMinSize.data());
    return squeezedTensor;
}

std::tuple<TensorPtr, NodePtr> squeezeTensor(const TensorPtr& tensor, const unsigned* const dim)
{
    TensorPtr out    = createSqueezedTensor(tensor, (dim == nullptr) ? Settable<unsigned>() : *dim);
    auto      guid = tensor->isShapeTensor() ? NodeFactory::squeezeShapeNodeTypeName : NodeFactory::squeezeNodeTypeName;
    NodePtr   expand = NodeFactory::createNode({tensor}, {out}, dim, guid, tensor->getName() + "_squeeze");
    return std::tie(out, expand);
}

TensorPtr createFlattenedTensor(const TensorPtr& tensor, unsigned axis)
{
    HB_ASSERT(axis < tensor->getDim(),
              "axis is equal or bigger than tensor dim, axis={}, dim={}",
              axis,
              tensor->getDim());
    unsigned dim0Size    = 1;
    unsigned dim0MinSize = 1;
    for (unsigned dim = 0; dim <= axis; ++dim)
    {
        dim0Size *= tensor->getSizeInElements(dim);
        dim0MinSize *= tensor->getMinimalSizeInElements(dim);
    }

    unsigned dim1Size    = 1;
    unsigned dim1MinSize = 1;
    for (unsigned dim = axis + 1; dim < tensor->getDim(); ++dim)
    {
        dim1Size *= tensor->getSizeInElements(dim);
        dim1MinSize *= tensor->getMinimalSizeInElements(dim);
    }

    SizeArray flattenSizes    = {dim0Size, dim1Size};
    SizeArray flattenMinSizes = {dim0MinSize, dim1MinSize};

    // This cloned tensor will be used as intermediate -> no need to copy the data
    TensorPtr flattendTensor = tensor->clone(false, false, false);
    flattendTensor->reshape(2, flattenSizes.data(), nullptr, flattenMinSizes.data());
    return flattendTensor;
}

TensorPtr createHost2DeviceTensor(synDataType dtype, uint64_t sizeInElements, const std::string& name)
{
    TSize sizes[] = {sizeInElements};
    TensorPtr h2dTensor = std::make_shared<Tensor>(1, sizes, dtype);
    h2dTensor->setTensorType(HOST_TO_DEVICE_TENSOR);
    // We need 2 times the size for min and max data
    h2dTensor->setDeviceSizeInBytes(h2dTensor->getTotalSizeInBytes() * 2);
    h2dTensor->bind(new char[h2dTensor->getTotalSizeInBytes()], true);
    h2dTensor->setAsDataTypeMatchData();
    h2dTensor->setProp(synTensorPropHostPtr);
    h2dTensor->setName(fmt::format("{}_host_data", name));
    return h2dTensor;
}

void insertReshapeNodeAfter(HabanaGraph&       g,
                            const pTensor&     input,
                            const pNode&       n,
                            const std::string& name,
                            const NSizeArray&  sizes,
                            insertNodeLocation location)
{
    pNode   newNode;
    pTensor newTensor;
    std::tie(newNode, newTensor) = createReshapeNode(input, name, sizes, location);

    // add the new node with the corresponding new tensor
    GraphEditor::replaceTensor(g, n, input, newTensor);
    GraphEditor::addNode(g, newNode);

    LOG_DEBUG(GC,
              "Inserted {} node \"{}\" on input \"{}\" of node \"{}\"",
              "reshape",
              name,
              input->getName(),
              n->getNodeName());
}

std::pair<pNode, pTensor> createReshapeNode(const pTensor&     input,
                                            const std::string& name,
                                            const NSizeArray&  sizes,
                                            insertNodeLocation location,
                                            bool               enforceLogical)
{
    // generate the new tensor
    unsigned dims      = input->getDim();
    pTensor  newTensor = std::make_shared<Tensor>(dims, sizes.begin(), input->getElementType());
    newTensor->setName(name);
    newTensor->setAllQuantizationParams(input->getAllQuantizationParams());
    newTensor->setDynamicRange(input->getDynamicRange());

    // generate the reshape node
    if (location == insertNodeLocation::AFTER_INPUT)
    {
        return {NodeFactory::createNode({input}, {newTensor}, &enforceLogical, NodeFactory::reshapeNodeTypeName, name),
                newTensor};
    }
    else
    {
        return {NodeFactory::createNode({newTensor}, {input}, &enforceLogical, NodeFactory::reshapeNodeTypeName, name),
                newTensor};
    }
}

std::pair<pNode, pTensor> createReshapeNode(const pTensor&     input,
                                            const std::string& name,
                                            const SizeArray&   sizes,
                                            insertNodeLocation location,
                                            bool               enforceLogical)
{
    NSizeArray sizes2;
    std::copy(sizes.begin(), sizes.end(), sizes2.begin());
    return createReshapeNode(input, name, sizes2, location, enforceLogical);
}

/**
 * @brief Reinterprets input tensor to/from target datatype based on bool isInput.
 *        The following sequence is created:
 *        (isInput=true): tensor[X,Y] -> reinterpret -> [factor*X, Y]
 *        (isInput=false): [factor*X, Y] -> reinterpret -> tensor[X,Y]
 *        where factor=dataTypeSizeInBytes(tensor->elementType())/dataTypeSizeInBytes(target type)
 * @return Returns resulting reinterpret cast node and created tensor with input dtype elements.
 */
std::pair<NodePtr, TensorPtr> reinterpretTensor(const TensorPtr& tensor, bool isInput, synDataType type)
{
    HB_ASSERT_PTR(tensor);
    HB_ASSERT(type != syn_type_na, "Expecting a valid reinterpret target type");
    std::array<NSizeArray, 2> tensorMinMaxSizes {tensor->getNMinimalSizesInElements(),
                                                 tensor->getAllNSizesInElements()};
    constexpr auto            MIN_SIZES = 0, MAX_SIZES = 1, FCD = 0;
    const auto factor   = static_cast<float>(dataTypeSizeInBytes(tensor->getElementType())) / dataTypeSizeInBytes(type);
    for (auto& sizes : tensorMinMaxSizes)
    {
        const auto newfcdSize = sizes.at(FCD) * factor;
        HB_ASSERT(std::ceil(newfcdSize) == newfcdSize,
                  "Expecting new fcd size {} to be an integer. orig fcd size: {}, factor: {}, tensor: {}",
                  newfcdSize,
                  sizes.at(FCD),
                  factor,
                  tensor->getName());
        sizes.at(FCD) = static_cast<TSize>(newfcdSize);
    }

    const auto reinterpretedTensor = tensor->clone(false, false);
    reinterpretedTensor->reshape(tensor->getDim(),
                                 tensorMinMaxSizes[MAX_SIZES].data(),
                                 nullptr,
                                 tensorMinMaxSizes[MIN_SIZES].data());
    reinterpretedTensor->setElementType(type);
    const NodePtr reinterpret = NodeFactory::createNode({isInput ? tensor : reinterpretedTensor},
                                                        {isInput ? reinterpretedTensor : tensor},
                                                        nullptr,
                                                        NodeFactory::reinterpretCastNodeTypeName,
                                                        fmt::format("{}_reinterpret", tensor->getName()));
    return {reinterpret, reinterpretedTensor};
}

/*
    given the tensor with shape [X,Y], create the sequence:
    (from64Bit=true):  [X,Y] -> reinterpret -> [2X,Y] -> reshape -> [2,X,Y]
    (from64Bit=false): [2,X,Y] -> reshape -> [2X,Y] -> reinterpret -> [X,Y]
*/
std::tuple<TensorPtr, NodePtr, NodePtr>
reinterpret64BitTensor(const TensorPtr& tensor, bool from64Bit, synDataType type)
{
    HB_ASSERT_PTR(tensor);
    auto [reinterpret, reinterpretedTensor] = reinterpretTensor(tensor, from64Bit, type);
    auto maxSizes                           = reinterpretedTensor->getNSizesInElements();
    auto minSizes                           = reinterpretedTensor->getNMinimalSizesInElements();

    auto reshapedTensor = reinterpretedTensor->clone(false, false);
    for (auto j = reinterpretedTensor->getDim(); j > 0; j--)
    {
        maxSizes[j] = maxSizes[j - 1];
        minSizes[j] = minSizes[j - 1];
    }
    maxSizes[0] = 2;
    minSizes[0] = 2;
    maxSizes[1] /= 2;
    minSizes[1] /= 2;
    reshapedTensor->reshape(tensor->getDim() + 1, maxSizes.data(), nullptr, minSizes.data());
    NodePtr reshape = NodeFactory::createNode({from64Bit ? reinterpretedTensor : reshapedTensor},
                                              {from64Bit ? reshapedTensor : reinterpretedTensor},
                                              nullptr,
                                              NodeFactory::staticReshapeNodeTypeName,
                                              fmt::format("{}_reshape", tensor->getName()));

    return std::make_tuple(reshapedTensor, reinterpret, reshape);
}

bool validateConvPadding(const SizeArray&                xSize,
                         const SizeArray&                wSize,
                         const SizeArray&                ySize,
                         uint32_t                        dimNum,
                         const synConvolution3DParamsV2& params)
{
    if (params.paddingType == PADDING_SAME)
    {
        SizeArray ySamePad = {
            convOutputDimSizeSamePadding(xSize[DIM_W], params.stride[CONV_STRIDE_WIDTH]),
            convOutputDimSizeSamePadding(xSize[DIM_H], params.stride[CONV_STRIDE_HEIGHT]),
            (dimNum == CONV_3D_TENSOR_DIM
                 ? convOutputDimSizeSamePadding(xSize[DIM_D_FOR_5D_TENSOR], params.stride[CONV_STRIDE_DEPTH])
                 : 1)};

        uint32_t compareElements = dimNum == CONV_3D_TENSOR_DIM ? 3 : 2;
        bool     validateRes     = std::equal(ySamePad.begin(), ySamePad.begin() + compareElements, ySize.begin() + 1);

        // Now validate padding.
        // For PADDING_SAME, padding size is actually variable.
        // We expect the user (or framework) to set correct padding according to the maximal size
        // of the input, and validate it here.
        // TODO <===== PADDING_TYPE
        // If we relax this requirement and allow the user not to set padding amount
        // this code should be modified to calculate *and set* padding in the node parameters,
        // instead of validating the user input.

        if (validateRes)
        {
            unsigned padT, padB, padL, padR, padFr = 0, padBk = 0;
            convPaddingSamePadding(xSize[DIM_W],
                                   params.kernel[CONV_KERNEL_WIDTH],
                                   params.dilation[CONV_DIL_WIDTH],
                                   params.stride[CONV_STRIDE_WIDTH],
                                   padL,
                                   padR);
            convPaddingSamePadding(xSize[DIM_H],
                                   params.kernel[CONV_KERNEL_HEIGHT],
                                   params.dilation[CONV_DIL_HEIGHT],
                                   params.stride[CONV_STRIDE_HEIGHT],
                                   padT,
                                   padB);

            bool validatedPadding = padL == params.padding[CONV_PAD_LEFT] && padR == params.padding[CONV_PAD_RIGHT] &&
                                    padT == params.padding[CONV_PAD_TOP] && padB == params.padding[CONV_PAD_BOTTOM];

            if (validatedPadding && dimNum == CONV_3D_TENSOR_DIM)
            {
                convPaddingSamePadding(xSize[DIM_D_FOR_5D_TENSOR],
                                       params.kernel[CONV_KERNEL_DEPTH],
                                       params.dilation[CONV_DIL_DEPTH],
                                       params.stride[CONV_STRIDE_DEPTH],
                                       padFr,
                                       padBk);
                validatedPadding = padFr == params.padding[CONV_PAD_FRONT] && padBk == params.padding[CONV_PAD_BACK];
            }
            if (!validatedPadding)
            {
                LOG_ERR(GC,
                        "Unexpected SAME padding. Received [{} {} {} {} {} {}], calculated from other conv params [{} "
                        "{} {} {} {} {}]",
                        params.padding[CONV_PAD_LEFT],
                        params.padding[CONV_PAD_RIGHT],
                        params.padding[CONV_PAD_TOP],
                        params.padding[CONV_PAD_BOTTOM],
                        params.padding[CONV_PAD_FRONT],
                        params.padding[CONV_PAD_BACK],
                        padL,
                        padR,
                        padT,
                        padB,
                        padFr,
                        padBk);
                validateRes = false;
            }
        }
        else
        {
            LOG_ERR(GC,
                    "Unexpected tensor size when using padding mode SAME. Received (W,H,D) [{}], calculated from "
                    "convolution params [{}]",
                    toString(ySize.begin() + 1, ySize.begin() + 1 + compareElements, ','),
                    toString(ySamePad.begin(), ySamePad.begin() + compareElements, ','));
        }
        return validateRes;
    }  // end of PADDING_SAME

    // First validate symmetric padding - backward compatibility
    SizeArray ySymmetricPad = {convOutputDimSize(xSize[DIM_W],
                                                 params.kernel[CONV_KERNEL_WIDTH],
                                                 params.stride[CONV_STRIDE_WIDTH],
                                                 params.padding[CONV_PAD_LEFT] * 2,
                                                 params.dilation[CONV_DIL_WIDTH]),
                                convOutputDimSize(xSize[DIM_H],
                                                  params.kernel[CONV_KERNEL_HEIGHT],
                                                  params.stride[CONV_STRIDE_HEIGHT],
                                                  params.padding[CONV_PAD_TOP] * 2,
                                                  params.dilation[CONV_DIL_HEIGHT]),
                                dimNum == CONV_3D_TENSOR_DIM ?
                                                convOutputDimSize(xSize[DIM_D_FOR_5D_TENSOR],
                                                                  params.kernel[CONV_KERNEL_DEPTH],
                                                                  params.stride[CONV_STRIDE_DEPTH],
                                                                  params.padding[CONV_PAD_FRONT] * 2,
                                                                  params.dilation[CONV_DIL_DEPTH]) : 1, 1, 1};

    uint32_t compareElements = dimNum == CONV_3D_TENSOR_DIM ? 3 : 2;

    bool validateRes = std::equal(ySymmetricPad.begin(), ySymmetricPad.begin() + compareElements, ySize.begin() + 1);

    if (validateRes) return true;

    // validate A-symmetric padding

    SizeArray yAsymmetricPad = {convOutputDimSize(xSize[DIM_W],
                                                  params.kernel[CONV_KERNEL_WIDTH],
                                                  params.stride[CONV_STRIDE_WIDTH],
                                                  params.padding[CONV_PAD_LEFT] + params.padding[CONV_PAD_RIGHT],
                                                  params.dilation[CONV_DIL_WIDTH]),
                                convOutputDimSize(xSize[DIM_H],
                                                  params.kernel[CONV_KERNEL_HEIGHT],
                                                  params.stride[CONV_STRIDE_HEIGHT],
                                                  params.padding[CONV_PAD_TOP] + params.padding[CONV_PAD_BOTTOM],
                                                  params.dilation[CONV_DIL_HEIGHT]),
                                dimNum == CONV_3D_TENSOR_DIM ?
                                        convOutputDimSize(xSize[DIM_D_FOR_5D_TENSOR],
                                                          params.kernel[CONV_KERNEL_DEPTH],
                                                          params.stride[CONV_STRIDE_DEPTH],
                                                          params.padding[CONV_PAD_FRONT] + params.padding[CONV_PAD_BACK],
                                                          params.dilation[CONV_DIL_DEPTH]) : 1, 1, 1};

    validateRes = std::equal(yAsymmetricPad.begin(), yAsymmetricPad.begin() + compareElements, ySize.begin() + 1);

    if (! validateRes)
    {
        LOG_ERR(GC,
                "Wrong y tensor size (W,H,D) [{}] according to x size and convolution params. expected [{}]",
                toString(ySize.begin() + 1, ySize.begin() + 1 + compareElements, ','),
                toString(yAsymmetricPad.begin(), yAsymmetricPad.begin() + compareElements, ','));
        return false;
    }

    return true;
}

bool validateConvolutionSize(const SizeArray&                xSize,
                             const SizeArray&                wSize,
                             const SizeArray&                ySize,
                             uint32_t                        dimNum,
                             const synConvolution3DParamsV2& params)
{
    if (wSize[WEIGHT_DIM_C] != xSize[DIM_C] / params.nGroups)
    {
        LOG_ERR(GC,
                "Inconsistent size between IFM channels/nGroups [{}/{}] = {} and weights input channels [{}]",
                xSize[DIM_C],
                params.nGroups,
                xSize[DIM_C] / params.nGroups,
                wSize[WEIGHT_DIM_C]);
        return false;
    }
    if (wSize[WEIGHT_DIM_K] != ySize[DIM_C])
    {
        LOG_ERR(GC,
                "Inconsistent size between OFM channels [{}] and weights output channels [{}]",
                ySize[DIM_C],
                wSize[WEIGHT_DIM_K]);
        return false;
    }
    if (wSize[WEIGHT_DIM_R] != params.kernel[CONV_KERNEL_HEIGHT] ||
        wSize[WEIGHT_DIM_S] != params.kernel[CONV_KERNEL_WIDTH] ||
        wSize[WEIGHT_DIM_Q] != params.kernel[CONV_KERNEL_DEPTH])
    {
        LOG_ERR(GC,
                "Inconsistent between weights tensor kernel [{}, {}, {}] and conv params kernels[{}, {}, {}]",
                wSize[WEIGHT_DIM_S],
                wSize[WEIGHT_DIM_R],
                wSize[WEIGHT_DIM_Q],
                params.kernel[CONV_KERNEL_WIDTH],
                params.kernel[CONV_KERNEL_HEIGHT],
                params.kernel[CONV_KERNEL_DEPTH]);
        return false;
    }

    return true;
}

bool validateTransposedDedxSize(const SizeArray&                xSize,
                                const SizeArray&                wSize,
                                const SizeArray&                ySize,
                                uint32_t                        dimNum,
                                const synConvolution3DParamsV2& params)
{
    if (wSize[WEIGHT_DIM_K] != xSize[DIM_C] / params.nGroups)
    {
        LOG_ERR(GC,
                "Inconsistent size between IFM channels/nGroups [{}/{}] = {} and weights input channels [{}]",
                xSize[DIM_C],
                params.nGroups,
                xSize[DIM_C] / params.nGroups,
                wSize[WEIGHT_DIM_K]);
        return false;
    }
    if (wSize[WEIGHT_DIM_C] != ySize[DIM_C])
    {
        LOG_ERR(GC,
                "Inconsistent size between OFM channels [{}] and weights output channels [{}]",
                ySize[DIM_C],
                wSize[WEIGHT_DIM_K]);
        return false;
    }
    if (wSize[WEIGHT_DIM_R] != params.kernel[CONV_KERNEL_HEIGHT] ||
        wSize[WEIGHT_DIM_S] != params.kernel[CONV_KERNEL_WIDTH] ||
        wSize[WEIGHT_DIM_Q] != params.kernel[CONV_KERNEL_DEPTH])
    {
        LOG_ERR(GC,
                "Inconsistent between weights tensor kernel [{}, {}, {}] and conv params kernels[{}, {}, {}]",
                wSize[WEIGHT_DIM_S],
                wSize[WEIGHT_DIM_R],
                wSize[WEIGHT_DIM_Q],
                params.kernel[CONV_KERNEL_WIDTH],
                params.kernel[CONV_KERNEL_HEIGHT],
                params.kernel[CONV_KERNEL_DEPTH]);
        return false;
    }

    return true;
}

/***
 * Get all indices where selected dimI equals d
 * for example if layout is KCSR and we want all indices where dim of K equals 1
 * @param sizes
 * @param dimI
 * @param d
 * @return a set containing all indices
 */
std::set<unsigned> dimIndexArray(const SizeArray& sizes, unsigned dimI, unsigned d)
{
    // assert no zeros in sizes array
    HB_ASSERT(std::find(sizes.begin(), sizes.end(), 0) == sizes.end(), "zeros in sizes array");
    HB_ASSERT(dimI < MAX_DIMENSIONS_NUM, "dimI is bigger than maximum dimensions number");

    std::set<unsigned> indexArray;

    for (int i = 0; i < sizes[0]; i++)
    {
        for (int j = 0; j < sizes[1]; j++)
        {
            for (int k = 0; k < sizes[2]; k++)
            {
                for (int l = 0; l < sizes[3]; l++)
                {
                    for (int m = 0; m < sizes[4]; m++)
                    {
                        int I = dimI == 0 ? d : i;
                        int J = dimI == 1 ? d : j;
                        int K = dimI == 2 ? d : k;
                        int L = dimI == 3 ? d : l;
                        int M = dimI == 4 ? d : m;
                        indexArray.insert(I + J * sizes[0] + K * sizes[0] * sizes[1] +
                                          L * sizes[0] * sizes[1] * sizes[2] +
                                          M * sizes[0] * sizes[1] * sizes[2] * sizes[3]);
                    }
                }
            }
        }
    }

    return indexArray;
}

/***
 * Name: extractScalarFromStaticTensor
 * gets the float value in case of static tensor that holds a scalar
 * @param tensor: in
 * @param scalarVal: out
 * @return bool whether the operation was completed successfully
 */
bool extractScalarFromStaticTensor(const pTensor& tensor, float& scalarVal)
{
    if (!tensor->isStaticParam())
    {
        LOG_WARN(GC, "extractScalarFromStaticTensor: tensor {} is not static", tensor->getName());
        return false;
    }

    if (tensor->getTotalElements() != 1)
    {
        LOG_WARN(GC,
                 "extractScalarFromStaticTensor: tensor {} is not a scalar, has {} elements",
                 tensor->getName(),
                 tensor->getTotalElements());
        return false;
    }

    synDataType dtype = tensor->isDataTypeMatchData() == false ? syn_type_float : tensor->getElementType();

    switch (dtype)
    {
    case syn_type_float:
    {
        scalarVal = *(float*)tensor->getData();
        break;
    }
    default:
    {
        LOG_WARN(GC,
                 "extractScalarFromStaticTensor: tensor {} has unsupported dtype {}",
                 tensor->getName(),
                 tensor->getElementType());
        return false;
    }
    }

    return true;
}

bool changeTensorElementTypeSafe(pTensor tensor, synDataType type)
{
    if (tensor->getData() != nullptr)
    {
        if (tensor->getElementType() != syn_type_float && tensor->isDataTypeMatchData())
        {
            LOG_ERR(GC, "Can't change tensor dtype, it is already quantized");
            return false;
        }

        bool shouldFreeBuffer = tensor->getShouldFreeBuffer();
        char* data = tensor->getData();

        if (shouldFreeBuffer && tensor->isStaticParam())
        {
            // assuming raw data is in FLOAT, no matter the data type
            unsigned dataSize = tensor->getTotalElements() * sizeof(float);
            char* newData = new char[dataSize];
            memcpy(newData, data, dataSize);
            data = newData;
        }
        else if (!tensor->isStaticParam())
        {
            // new buffer size is according new dtype
            unsigned dataSize = tensor->getTotalElements() * dataTypeSizeInBytes(type);
            char* newData = new char[dataSize];
            memset(newData, 0, dataSize); // no interest in the values of the tensor
            data             = newData;
            shouldFreeBuffer = true;
        }

        tensor->unbind();
        tensor->setElementType(type);
        tensor->bind(data, shouldFreeBuffer);
    }
    else
    {
        tensor->setElementType(type);
    }

    return true;
}
const char* synTensorType2Txt(synTensorType type)
{
    static_assert(TENSOR_TYPE_MAX == 7, "Update the table below");

    switch(type)
    {
        case DATA_TENSOR                   : return "DATA_TENSOR";
        case SHAPE_TENSOR                  : return "SHAPE_TENSOR";
        case INPUT_DESCRIBING_SHAPE_TENSOR : return "INPUT_DESCRIBING_SHAPE_TENSOR";
        case DATA_TENSOR_DYNAMIC           : return "DATA_TENSOR_DYNAMIC";
        case DEVICE_SHAPE_TENSOR           : return "DEVICE_SHAPE_TENSOR";
        case HOST_SHAPE_TENSOR:
            return "HOST_SHAPE_TENSOR";
        case HOST_TO_DEVICE_TENSOR:
            return "HOST_TO_DEVICE_TENSOR";

        default: return "Unknown type";
    }
}

bool verifyDeviceShapeTensor(unsigned         dims,
                             const TSize*     maxSizes,
                             synDataType      dataType,
                             std::string_view tensorName,
                             const TSize*     minSizes)
{
    if (dims != 1)
    {
        LOG_ERR(SYN_API, "{}: Shape tensor {} has invalid dims ({}) should be 1", HLLOG_FUNC, tensorName, dims);
        return false;
    }

    if (maxSizes[0] != MAX_DIMENSIONS_NUM || (minSizes != nullptr && minSizes[0] != maxSizes[0]))
    {
        LOG_ERR(SYN_API,
                "{}: Shape tensor {} has invalid size ({}), dim0 should be 5",
                HLLOG_FUNC,
                tensorName,
                maxSizes[0]);
        return false;
    }

    if (dataType != syn_type_uint32)
    {
        LOG_ERR(SYN_API,
                "{}: Shape tensor {} has invalid data type ({}),should be uint32",
                HLLOG_FUNC,
                tensorName,
                dataType);
        return false;
    }
    return true;
}

synTensorProperty quantizationPropertyToTensorProperty(synQuantizationProperty quantProp)
{
    switch (quantProp)
    {
        case SYN_QUANT_DYNAMIC_RANGE:
            return synTensorPropDynamicRange;
        case SYN_QUANT_PC_DYNAMIC_RANGE:
            return synTensorPropPCDynamicRange;
        case SYN_QUANT_METADATA:
            return synTensorPropQuantMetadata;
        case SYN_FP_QUANT_METADATA:
            return synTensorPropFpQuantMetadata;
        case SYN_QUANT_FLAGS:
            return synTensorPropFlags;
        default:
            HB_ASSERT(false, "Unknown synQuantizationProperty {}", quantProp);
            return synTensorPropUnknown;
    }
}

bool isOpSupportsNdims(const NodePtr& node)
{
    switch (node->getNodeType())
    {
        case Node::TYPE_INTERNAL_RESHAPE:
        case Node::TYPE_SLICE:
        case Node::TYPE_SLICE_BWD:
        case Node::TYPE_SLICE_GRAD:
        case Node::TYPE_SLICE_INSERT:
        case Node::TYPE_INTERNAL_CONCAT:
        case Node::TYPE_INTERNAL_SPLIT:
        case Node::TYPE_FCD_BROADCAST:
        case Node::TYPE_MEMCOPY:
        case Node::TYPE_STRIDED_INSERT:
        case Node::TYPE_STRIDED_VIEW:
        case Node::TYPE_SQUEEZE_NODE:
        case Node::TYPE_INTERNAL_EXPAND_DIMS:
        case Node::TYPE_REINTERPRET_CAST:
        // TPC nodes can  support ndims in some cases, and we don't know it in advance (the glue code will check it)
        case Node::TYPE_USER:
        {
            return true;
        }
        case Node::TYPE_INTERNAL_TRANSPOSE:
        {
            const auto mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
            return ((mmeNode && mmeNode->isTransposeViaGemm()) ? false : true);
        }
        default:
            return false;
    }
}

bool isTensorDimsValidForNode(const TensorPtr& tensor, const NodePtr& node, bool checkNDims)
{
    unsigned dim          = tensor->getDim();
    if (dim > SYN_GAUDI_MAX_TENSOR_DIM)
    {
        LOG_ERR(SYN_API, "{}: Tensor dim {} is not supported for this device", HLLOG_FUNC, dim);
        return false;
    }

    if (checkNDims && dim > SYN_MAX_TENSOR_DIM)
    {
        if (tensor->isDynamicShape())
        {
            LOG_ERR(SYN_API, "{}: Tensor dim {} is not supported together with DSD", HLLOG_FUNC, dim);
            return false;
        }
        // only allowed ops should support ndims
        if (!isOpSupportsNdims(node))
        {
            LOG_ERR(SYN_API, "{}: Tensor dim {} is not supported for this node type", HLLOG_FUNC, dim);
            return false;
        }
    }
    return true;
}

std::string sanitizeFileName(std::string fileName)
{
    if (!fileName.empty() && fileName[0] == '-')
    {
        fileName[0] = '_';
    }

    for (char& c : fileName)
    {
        c = ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '-' || c == '.') ? c
                                                                                                                 : '_';
    }

    return fileName;
}

bool isDirectory(const std::string& path)
{
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

// TODO: constexpr and inline into the header?
bool isFusedKernel(std::string_view guid)
{
    return startsWith(guid, "fused_kernel_");
}

SizeVector toSizeVector(const TensorPtr& tensor)
{
    auto         rank      = tensor->getDim();
    auto         sizes     = tensor->getAllSizesInElements();
    const TSize* sizesData = sizes.data();

    return SizeVector(sizesData, sizesData + rank);
}

SizeVector toSizeVector(const SizeArray& sizes, unsigned int rank)
{
    const TSize* sizesData = sizes.data();
    return SizeVector(sizesData, sizesData + rank);
}

SizeVector toSizeVector(const TSize sizes[Tensor::c_tensorMaxDim], unsigned int rank)
{
    return SizeVector(sizes, sizes + rank);
}

// this function receive tensor and sequences of dense dimensions and merge those sequences.
// since the start of each sequence is the dimension after the end of that last sequence, we receive only the
// end of the sequences.
static std::pair<TensorShape, NStrideArray>
mergeDenseDimensions(const TensorPtr& t, const DimVector& sequences, TStride maxStride)
{
    NSizeArray   maxDenseSizes;
    NSizeArray   minDenseSizes;
    NStrideArray denseStrides = {1};
    std::fill(maxDenseSizes.begin(), maxDenseSizes.end(), 1);
    std::fill(minDenseSizes.begin(), minDenseSizes.end(), 1);

    denseStrides[0] = t->getStrideInBytes(0);

    unsigned index = 0;
    for (unsigned dim = 0; dim < t->getDim(); ++dim)
    {
        maxDenseSizes[index] *= t->getSizeInElements(dim);
        minDenseSizes[index] *= t->getMinimalSizeInElements(dim);

        if (dim == sequences[index])  // end of the sequence
        {
            denseStrides[index + 1] = t->getStrideInBytes(dim + 1);
            ++index;
        }
    }
    // fill the rest of strides with the maximal stride
    std::fill(denseStrides.begin() + index, denseStrides.end(), maxStride);
    TensorShape shape(index, maxDenseSizes, minDenseSizes);
    return std::make_pair(shape, denseStrides);
}

// stride[i] is needed in order to calculate the amount of bytes between 2 consecutive indices in dimension i.
// Therefore, when the dimension size is 1, there is no real meaning for stride and we can set it to the default stride.
static void fixOneSizeDimsStrides(const TensorPtr& t, NStrideArray& strides)
{
    // the correctness is guaranteed only if the strides are sorted, which is not the case after logical transpose.
    if (std::is_sorted(strides.begin(), strides.begin() + (t->getDim() + 1)))
    {
        for (unsigned dim = 1; dim < t->getDim(); ++dim)
        {
            // check if min and max sizes are 1
            if (t->getSizeInElements(dim) == 1 && t->getMinimalSizeInElements(dim) == 1)
            {
                strides[dim] = strides[dim - 1] * t->getSizeInElements(dim - 1);
            }
        }
    }
}

std::pair<TensorShape, NStrideArray> mergeDenseDimensions(const TensorPtr& t)
{
    DimVector            sequences;
    auto                 tensorDim = t->getDim();
    NStrideArray         strides   = {1};
    t->getNStridesInBytes(strides.data());
    fixOneSizeDimsStrides(t, strides);
    for (size_t dim = 0; dim < tensorDim; ++dim)
    {
        auto currentStrideInBytes = strides[dim];
        auto nextStrideInBytes    = strides[dim + 1];
        auto sizeInElements       = t->getSizeInElements(dim);
        // dimension is end of dense sequence if one of the following happen:
        // (1) is the last dimension.
        // (2) is dynamic dim, therfore it may be sparse
        // (3) the stride of the next dimension isn't the multiplication of the current size by the current stride
        if (dim == tensorDim - 1 || t->isDynamicDim(dim) || sizeInElements * currentStrideInBytes != nextStrideInBytes)
        {
            sequences.push_back(dim);
        }
    }
    return mergeDenseDimensions(t, sequences, t->getMaxStride());
}

void* allocateBufferForSynType(synDataType type, uint64_t elementsNum)
{
    void* castToBuffer = nullptr;
    switch (type)
    {

        case syn_type_int4:
        case syn_type_uint4:
        case syn_type_int8:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
            castToBuffer = new uint8_t[elementsNum];
            break;
        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_fp16:
            castToBuffer = new uint16_t[elementsNum];
            break;
        case syn_type_single:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_hb_float:
        case syn_type_tf32:
            castToBuffer = new uint32_t[elementsNum];
            break;
        case syn_type_uint64:
        case syn_type_int64:
            castToBuffer = new uint64_t[elementsNum];
            break;
        default:
            HB_ASSERT(false, "Unexpected data type");
    }
    return castToBuffer;
}

bool isInferenceQuantization(const HabanaGraph& g)
{
    return g.getInferenceMode() && g.getQuantizationEnabled();
}

bool isGuidPrefix(const NodePtr& node, std::string_view prefix )
{
    HB_ASSERT_PTR(node);
    return startsWith(node->getGUID(), prefix);
}