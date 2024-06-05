#ifndef MME__MEMORY_CONFIG_H
#define MME__MEMORY_CONFIG_H

#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
// handle and configure memory related parts of the descriptor: reduction, cache config
class MmeMemoryConfigMgr
{
public:
    static std::shared_ptr<MmeMemoryConfigMgr> getMmeMemoryConfigMgr(ChipType chipType);

    // handle reduction
    void
    setReductionParams(EMmeReductionOp reductionOp, EMmeReductionRm reductionRm, EMmeDataType dataTypeOut, bool clipFp);
    template<typename Desc>
    void setReductionUserBits(Desc& desc) const;
    template<typename Desc>
    void setEmptyReductionUserBits(Desc& desc) const;

    // handle cache
    void setCacheParams(std::array<EMmeCacheDirective, e_mme_op_nr> cacheDirective,
                        std::array<EMmeCacheClass, e_mme_op_nr> clss,
                        std::array<EMmeCacheQOS, e_mme_op_nr> qos,
                        std::array<uint16_t, e_mme_op_nr> mcid);
    template<typename Desc>
    void setCacheUserBits(Desc& desc) const;

protected:
    MmeMemoryConfigMgr() = default;
    virtual ~MmeMemoryConfigMgr() = default;
    unsigned convertMmeDataTypeForReduction(EMmeDataType mmeDataType) const;
    bool reductionEn() const { return m_reductionOp != e_mme_reduction_none; }
    virtual unsigned getAxiUserData() const = 0;

    EMmeReductionRm m_reductionRoundingMode = EMmeReductionRm::e_mme_reduction_round_half_to_nearest_even;
    uint8_t m_reductionDataType = 0;
    MmeCommon::EMmeReductionOp m_reductionOp = EMmeReductionOp::e_mme_reduction_none;
    bool m_clipFp = false;

    virtual uint32_t getCacheUserData(MmeCommon::EMmeInternalOperand operand) const { return 0; }
    virtual uint32_t getCacheDirectiveBits(MmeCommon::EMmeInternalOperand operand) const { return 0; }

    std::array<EMmeCacheDirective, e_mme_op_nr> m_cacheDirective = {EMmeCacheDirective::NoAllocate};
    std::array<EMmeCacheClass, e_mme_op_nr> m_clss = {EMmeCacheClass::Normal};
    std::array<EMmeCacheQOS, e_mme_op_nr> m_qos = {EMmeCacheQOS::Bucket1};
    std::array<uint16_t, e_mme_op_nr> m_mcid = {0};
};

class Gaudi2MmeMemoryConfigMgr : public MmeMemoryConfigMgr
{
public:
    Gaudi2MmeMemoryConfigMgr() = default;
    virtual ~Gaudi2MmeMemoryConfigMgr() = default;
    virtual unsigned getAxiUserData() const override;

private:
    unsigned getReductionCommand() const;
};

class Gaudi3MmeMemoryConfigMgr : public MmeMemoryConfigMgr
{
public:
    Gaudi3MmeMemoryConfigMgr() = default;
    virtual ~Gaudi3MmeMemoryConfigMgr() = default;
    virtual unsigned getAxiUserData() const override;

protected:
    virtual uint32_t getCacheDirectiveBits(MmeCommon::EMmeInternalOperand operand) const override;
    virtual uint32_t getCacheUserData(MmeCommon::EMmeInternalOperand operand) const override;

private:
    unsigned getReductionCommand() const;
};

}  // namespace MmeCommon

#endif //MME__MEMORY_CONFIG_H
