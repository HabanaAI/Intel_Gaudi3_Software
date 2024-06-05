#include "mme_user.h"
#include "mme_code_generation.h"
#include "gaudi3/mme_descriptor_generator.h"

namespace gaudi3
{
class Gaudi3MmeUser : public MmeCommon::MmeCodeGenerator<Mme::Desc, Mme::MmeCmd, Mme::RegBlock>
{
public:
    Gaudi3MmeUser(unsigned mmeLimit) : MmeCodeGenerator(MmeCommon::e_mme_Gaudi3, mmeLimit) {};

    virtual std::list<CPProgram> createSimulatorProgram(const std::list<CPProgram>& progs, unsigned seed) override;
    void setSoForPowerTest(CPProgram& prog, bool isPowerProg) override;

protected:
    unsigned getAguInAMask(Mme::Desc& desc) override;
    unsigned getAguInBMask(Mme::Desc& desc) override;
    bool doesCmdExecutesOperand(const Mme::Desc* desc, const Mme::MmeCmd& cmd, const bool isA) override;

    void
    addMmeLfsrInitSequence(CPProgram& prog, unsigned seed, const MmeCommon::LfsrData& lfsrData, bool configLfsr) final;
    void addClipInfInputConfig(CPProgram& prog, const bool clipInfIn) final;
    void addMessageBarrier(CPProgram& prog) final;

    uint64_t getMmeCtrlBase() final;
    unsigned getMmeQueueId(unsigned mmeIdx, unsigned stream) final;
    void pushFenceCmd(CPProgram& prog, unsigned fenceIdx, unsigned incWaitVal) final;
    void pushWaitCmd(CPProgram& prog, unsigned waitCycles, unsigned waitValue, unsigned waitIdx) final;
    void addReductionToPowerDesc(Mme::Desc& powerDesc, MmeCommon::EMmeReductionOp op) final;
    void removeReductionToPowerDesc(Mme::Desc& powerDesc) final;
    void zeroSignalForPowerTest(ActivationVec& activation) final;
};
}  // namespace gaudi3