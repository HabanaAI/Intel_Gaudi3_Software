#pragma once

#include "include/mme_common/mme_common_enum.h"
#include "mme_user.h"
#include "mme_code_generation.h"
#include "include/gaudi2/mme_descriptor_generator.h"

namespace gaudi2
{
class Gaudi2MmeUser : public MmeCommon::MmeCodeGenerator<Gaudi2::Mme::Desc, Gaudi2::Mme::MmeCmd, Gaudi2::Mme::RegBlock>
{
public:
    Gaudi2MmeUser(unsigned mmeLimit) : MmeCodeGenerator(MmeCommon::e_mme_Gaudi2, mmeLimit) {};

    virtual std::list<CPProgram> createSimulatorProgram(const std::list<CPProgram>& progs, unsigned seed) override;
    virtual void setSoForPowerTest(CPProgram& prog, bool isPowerProg) final;

protected:
    unsigned getAguInAMask(Gaudi2::Mme::Desc& desc) override;
    unsigned getAguInBMask(Gaudi2::Mme::Desc& desc) override;
    bool doesCmdExecutesOperand(const Gaudi2::Mme::Desc* desc, const Gaudi2::Mme::MmeCmd& cmd, const bool isA) override;

    void
    addMmeLfsrInitSequence(CPProgram& prog, unsigned seed, const MmeCommon::LfsrData& lfsrData, bool configLfsr) final;
    void addClipInfInputConfig(CPProgram& prog, const bool clipInfIn) final;
    void addMessageBarrier(CPProgram& prog) final;

    uint64_t getMmeCtrlBase() final;
    unsigned getMmeQueueId(unsigned mmeIdx, unsigned stream) final;
    void pushFenceCmd(CPProgram& prog, unsigned fenceIdx, unsigned incWaitVal) final;
    void pushWaitCmd(CPProgram& prog, unsigned waitCycles, unsigned waitValue, unsigned waitIdx) final;
    void addReductionToPowerDesc(Gaudi2::Mme::Desc& powerDesc, MmeCommon::EMmeReductionOp op) final;
    void removeReductionToPowerDesc(Gaudi2::Mme::Desc& powerDesc) final;
    void zeroSignalForPowerTest(Gaudi2::ActivationVec& activation) final;
};
}  // namespace gaudi2
