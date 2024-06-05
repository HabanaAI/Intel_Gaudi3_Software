#include "memory_config.h"
#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
unsigned MmeMemoryConfigMgr::convertMmeDataTypeForReduction(EMmeDataType mmeDataType) const
{
    switch (mmeDataType)
    {
        case e_type_fp16:
            return 8;
        case e_type_bf16:
            return 6;
        case e_type_fp32:
            return 7;
        case e_type_tf32:
            MME_ASSERT(0, "tf32 not supported with reduction");
            return -1;
        case e_type_fp8_143:
            MME_ASSERT(0, "fp8_143 not supported with reduction");
            return -1;
        case e_type_fp8_152:
            return 0x9;
        default:
            MME_ASSERT(0, "Unknown data type in reduction");
            return -1;
    }
}

void MmeMemoryConfigMgr::setReductionParams(EMmeReductionOp reductionOp,
                                            EMmeReductionRm reductionRm,
                                            EMmeDataType dataTypeOut,
                                            bool clipFp)
{
    m_reductionOp = reductionOp;
    if (reductionEn())
    {
        m_reductionRoundingMode = reductionRm;
        m_reductionDataType = (uint8_t) convertMmeDataTypeForReduction(dataTypeOut);
        m_clipFp = clipFp;
    }
}

std::shared_ptr<MmeMemoryConfigMgr> MmeMemoryConfigMgr::getMmeMemoryConfigMgr(ChipType chipType)
{
    switch (chipType)
    {
        case e_mme_Gaudi2:
            return std::make_shared<Gaudi2MmeMemoryConfigMgr>();
        case e_mme_Gaudi3:
            return std::make_shared<Gaudi3MmeMemoryConfigMgr>();
        default:
            MME_ASSERT(0, "not supported yet");
            return nullptr;
    }
}

unsigned Gaudi2MmeMemoryConfigMgr::getAxiUserData() const
{
    if (!reductionEn())
    {
        return 0;
    }

    union MmeUserData
    {
        struct
        {
            uint32_t first : 10;
            uint32_t steady : 10;
            uint32_t mask : 6;
            uint32_t reserved : 6;
        };
        uint32_t dw;
    } ud;  // MmeUserData
    unsigned reductionCmd = 0;

    reductionCmd = getReductionCommand();
    // Set the axiUserData field in the descriptor
    ud.first = reductionCmd;
    ud.steady = ud.first;
    ud.mask = e_mme_outer_loop;
    ud.reserved = 0;

    return ud.dw;
}

unsigned Gaudi2MmeMemoryConfigMgr::getReductionCommand() const
{
    union
    {
        struct
        {
            // taken from H6 NoC spec
            uint32_t enable : 1;
            uint32_t dataType : 4;
            uint32_t operation1 : 2;
            uint32_t rounding : 2;
            uint32_t operation2 : 1;
            uint32_t padding : 18;
        };
        uint32_t val;
    } cmd;

    cmd.val = 0;
    cmd.enable = reductionEn() ? 1 : 0;
    cmd.dataType = m_reductionDataType;
    cmd.operation1 = (uint8_t) m_reductionOp & 0x03;
    cmd.rounding = (uint8_t) m_reductionRoundingMode;
    cmd.operation2 = ((uint8_t) m_reductionOp & 0x04) >> 2;

    return cmd.val;
}

unsigned Gaudi3MmeMemoryConfigMgr::getAxiUserData() const
{
    if (!reductionEn())
    {
        return 0;
    }

    union MmeUserData
    {
        struct
        {
            uint32_t first : 11;
            uint32_t steady : 11;
            uint32_t mask : 6;
            uint32_t reserved : 4;
        };
        uint32_t dw;
    } ud;  // MmeUserData
    unsigned reductionCmd = 0;

    reductionCmd = getReductionCommand();
    // Set the axiUserData field in the descriptor
    // TODO: SW-92279 really use first\steady
    ud.first = reductionCmd;
    ud.steady = ud.first;
    ud.mask = e_mme_outer_loop;
    ud.reserved = 0;

    return ud.dw;
}

unsigned Gaudi3MmeMemoryConfigMgr::getReductionCommand() const
{
    union
    {
        struct
        {
            // taken from H9 NoC spec
            uint32_t enable : 1;
            uint32_t operation : 3;
            uint32_t rounding : 2;
            uint32_t dataType : 4;
            uint32_t clip : 1;
            uint32_t padding : 21;
        };
        uint32_t val;
    } cmd;

    cmd.val = 0;
    cmd.enable = reductionEn() ? 1 : 0;
    cmd.operation = (uint8_t) m_reductionOp;
    cmd.rounding = (uint8_t) m_reductionRoundingMode;
    cmd.dataType = m_reductionDataType;
    cmd.clip = m_clipFp;

    return cmd.val;
}

void MmeMemoryConfigMgr::setCacheParams(std::array<EMmeCacheDirective, e_mme_op_nr> cacheDirective,
                                        std::array<EMmeCacheClass, e_mme_op_nr> clss,
                                        std::array<EMmeCacheQOS, e_mme_op_nr> qos,
                                        std::array<uint16_t, e_mme_op_nr> mcid)
{
    m_cacheDirective = cacheDirective;
    m_clss = clss;
    m_qos = qos;
    m_mcid = mcid;
}

uint32_t Gaudi3MmeMemoryConfigMgr::getCacheUserData(MmeCommon::EMmeInternalOperand operand) const
{
    union  // AxUser bits (see H9 NoC spec)
    {
        struct
        {
            uint32_t qosFirst : 4;
            uint32_t qosSteady : 4;
            uint32_t qosMask : 6;
            uint32_t mcid : 16;
            uint32_t clss : 2;
            uint32_t reserved : 10;
        };
        uint32_t dw;
    } ud;  // MmeUserData
    ud.dw = 0;
    // TODO: [SW-92279] really use first\steady
    ud.qosFirst = m_qos[operand];
    ud.qosSteady = m_qos[operand];
    ud.qosMask = 0;
    ud.mcid = m_mcid[operand];
    ud.clss = m_clss[operand];
    return ud.dw;
}
uint32_t Gaudi3MmeMemoryConfigMgr::getCacheDirectiveBits(MmeCommon::EMmeInternalOperand operand) const
{
    // values taken from H9 NoC specs
    switch (m_cacheDirective[operand])
    {
        case SkipCache:
            return 0x1;
        case NoAllocate:
            return 0x3;
        case HomeAllocate:
            return 0x7;
        case DcoreAllocate:
            return 0xB;
        case SharedAllocate:
            return 0xF;
        default:
            return 0;
    }
}
}  // namespace MmeCommon

#include "gaudi2/mme.h"
template<>
void MmeCommon::MmeMemoryConfigMgr::setReductionUserBits(Gaudi2::Mme::Desc& desc) const
{
    desc.axiUserData.dw = getAxiUserData();
}
#include "gaudi3/mme.h"
template<>
void MmeCommon::MmeMemoryConfigMgr::setReductionUserBits(gaudi3::Mme::Desc& desc) const
{
    desc.axiAwUserData.dw = getAxiUserData();
}
template<>
void MmeCommon::MmeMemoryConfigMgr::setEmptyReductionUserBits(gaudi3::Mme::Desc& desc) const
{
    desc.axiAwUserData.dw = 0;
}
template<>
void MmeCommon::MmeMemoryConfigMgr::setCacheUserBits(gaudi3::Mme::Desc& desc) const
{
    // operand A
    desc.axiCacheData.aguA = getCacheDirectiveBits(e_mme_op_a);
    desc.axiUserDataA.dw = getCacheUserData(e_mme_op_a);
    // operand B
    desc.axiCacheData.aguB = getCacheDirectiveBits(e_mme_op_b);
    desc.axiUserDataB.dw = getCacheUserData(e_mme_op_b);
    // operand C
    desc.axiCacheData.aguOut = getCacheDirectiveBits(e_mme_op_c);
    desc.axiUserDataCout.dw = getCacheUserData(e_mme_op_c);
}
