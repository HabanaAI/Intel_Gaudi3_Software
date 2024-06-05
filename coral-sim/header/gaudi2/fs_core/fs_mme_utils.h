#pragma once
#include <stdint.h>

#include <condition_variable>
#include <mutex>

#include "gaudi2/mme.h"

#include "fs_assert.h"
#include "fs_log.h"
#include "fs_mme_md.h"

#define FP8_MODE143_EXP 4
#define FP8_MODE143_MAN 3
#define FP8_MODE152_EXP 5
#define FP8_MODE152_MAN 2
#define FP8_MODE152_BIAS 15

#define mme_min(a, b) (((a) < (b)) ? (a) : (b))
#define mme_div_ceil(n, d) (((n) + (d)-1) / (d))

#define MME_LOG "MME"

#define MME_LOG_TRACE(instance, fmt, ...)                                                                              \
    (instance)->getMme()->log_trace("{}: " fmt, (instance)->getInstanceName(), ##__VA_ARGS__)
#define MME_LOG_DEBUG(instance, fmt, ...)                                                                              \
    (instance)->getMme()->log_debug("{}: " fmt, (instance)->getInstanceName(), ##__VA_ARGS__)
#define MME_LOG_INFO(instance, fmt, ...)                                                                               \
    (instance)->getMme()->log_info("{}: " fmt, (instance)->getInstanceName(), ##__VA_ARGS__)
#define MME_LOG_WARNING(instance, fmt, ...)                                                                            \
    (instance)->getMme()->log_warning("{}: " fmt, (instance)->getInstanceName(), ##__VA_ARGS__)
#define MME_LOG_ERROR(instance, fmt, ...)                                                                              \
    (instance)->getMme()->log_error("{}: " fmt, (instance)->getInstanceName(), ##__VA_ARGS__)
#define MME_LOG_CRITICAL(instance, fmt, ...)                                                                           \
    (instance)->getMme()->log_critical("{}: " fmt, (instance)->getInstanceName(), ##__VA_ARGS__)

namespace Gaudi2
{
namespace Mme
{
typedef enum
{
    e_mme_agu0_idx      = 0,
    e_mme_agu1_idx      = 1,
    e_mme_agu2_idx      = 2,
    e_mme_agu3_idx      = 3,
    e_mme_agu4_idx      = 4,
    e_mme_agu_cout0_idx = 5,
    e_mme_agu_cout1_idx = 6,
    e_mme_eu_brain_idx  = 7,
    e_mme_ap_brain_idx  = 8,
} EMmeBrainIdx;

typedef enum
{
    e_mme_operand_0 = (1 << 0),
    e_mme_operand_1 = (1 << 1),
    e_mme_operand_2 = (1 << 2),
    e_mme_operand_3 = (1 << 3),
    e_mme_operand_4 = (1 << 4),
} EMmeInputOperand;

typedef enum
{
    e_mme_fp8_143_bias_3  = 3,
    e_mme_fp8_143_bias_7  = 7,
    e_mme_fp8_143_bias_11 = 11,
    e_mme_fp8_143_bias_15 = 15,
} EMmeFP8LegalBias;

class LbwWrMaster
{
   public:
    LbwWrMaster() {}

    virtual ~LbwWrMaster() {}

    virtual void write(uint64_t address, uint32_t value) = 0;
};

class HbwWrMaster
{
   public:
    HbwWrMaster() {}

    virtual ~HbwWrMaster() {}

    virtual void write(uint64_t address, const uint8_t* data, unsigned size, uint32_t usr) = 0;
};

class HbwRdMaster
{
   public:
    HbwRdMaster() {}

    virtual ~HbwRdMaster() {}

    virtual void read(uint64_t address, char* data, unsigned size, uint32_t usr) = 0;
};

inline bool full(uint64_t size, uint64_t wr, uint64_t rd)
{
    return size == wr - rd;
}

inline bool empty(uint64_t wr, uint64_t rd)
{
    return wr == rd;
}

inline uint8_t getLogElementSize(EMmeDataType dataType)
{
    uint8_t res = 0;
    switch (dataType) {
        case Gaudi2::Mme::e_mme_dt_fp16:
        case Gaudi2::Mme::e_mme_dt_bf16: res = 1; break;
        case Gaudi2::Mme::e_mme_dt_fp32:
        case Gaudi2::Mme::e_mme_dt_fp32ieee:
        case Gaudi2::Mme::e_mme_dt_tf32: res = 2; break;
        case Gaudi2::Mme::e_mme_dt_fp8_143:
        case Gaudi2::Mme::e_mme_dt_fp8_152: res = 0; break;
        default: FS_ASSERT(0);
    }
    return res;
}

inline uint8_t getElementSize(EMmeDataType dataType)
{
    return 1 << getLogElementSize(dataType);
}

struct AlignedAddr
{
    uint64_t addr;
    unsigned base;
    unsigned size;
};

static inline void
getAlignedAddrs(uint64_t addr, const unsigned lpad, const unsigned mpad, AlignedAddr& addr0, AlignedAddr& addr1)
{
    uint32_t sumPad = lpad + mpad;
    uint64_t size   = sumPad >= Mme::c_cl_size ? 0 : (Mme::c_cl_size - lpad - mpad);
    addr += lpad;

    addr0.addr = addr & ~((uint64_t)(Gaudi2::Mme::c_cl_size - 1));
    addr1.addr = addr0.addr + Gaudi2::Mme::c_cl_size;

    addr0.size = std::min(addr1.addr - addr, size);
    addr1.size = size - addr0.size;

    addr0.base = addr - addr0.addr;
    addr1.base = 0;
}

//        inline Gaudi2::Mme::MetaData::EMmeTEDataType encodeDataTypeForTE(EMmeDataType dataType)
//        {
//            uint8_t elemetSize = getElementSize(dataType);
//            return (elemetSize == 1 ? Gaudi2::Mme::MetaData::EMmeTEDataType::e_mme_size_8bits :
//                    (elemetSize == 2 ? Gaudi2::Mme::MetaData::EMmeTEDataType::e_mme_size_16bits :
//                     Gaudi2::Mme::MetaData::EMmeTEDataType::e_mme_size_32bits));
//        }

inline bool isOutputAgu(const Gaudi2::Mme::EMmeBrainIdx aguId)
{
    return (aguId == e_mme_agu_cout0_idx || aguId == e_mme_agu_cout1_idx);
}

inline unsigned encodeDataTypeForAGU(EMmeDataType dataType)
{
    // encoding:
    // 0: FP8
    // 1: FP16/BF16
    // 2: FP32
    return getElementSize(dataType) / 2;
}

} // namespace Mme
} // namespace Gaudi2
