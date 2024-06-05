#pragma once

#include <vector>
#include "include/mme_common/mme_common_enum.h"
#include "mme_test_data_gen.h"
#include "common/mme_mem_access_checker.h"
#include "mme_verification/common/mme_reg_write_cmd.h"
#include "sync_object_manager.h"
#include "mme_user.h"

namespace MmeCommon
{
template<typename Desc, typename MmeCmd, typename RegBlock>
class MmeCodeGenerator : public MmeUser
{
public:
    MmeCodeGenerator(ChipType chipType, unsigned mmeNr) : MmeUser(chipType, mmeNr)
    {
        m_deviceDescMme.resize(m_mmeNr);
        m_deviceDescDma.resize(m_mmeNr);
    };
    virtual ~MmeCodeGenerator() = default;
    void buildCmds(const MmeTestParams& testParams,
                   std::vector<MmeQmanCmd> cmds[],
                   MmeDataParams& dataParams,
                   bool& firstValidTest,
                   MmeMemoryUsage testMemUsage) final;
    void createNullDescCmds(MmeDataParams& dataParams, std::vector<MmeQmanCmd> cmds[]) final;

protected:
    virtual unsigned getAguInAMask(Desc& desc) = 0;
    virtual unsigned getAguInBMask(Desc& desc) = 0;
    virtual bool doesCmdExecutesOperand(const Desc* desc, const MmeCmd& cmd, const bool isA) = 0;

private:
    void pushCmd(std::vector<MmeQmanCmd>& cmds, const size_t offset, const unsigned size, const void* const value);
    inline unsigned getFenceIdxForFenceWaitCmds() { return c_localFenceIdx; }
    void range2cmds(const Desc* desc, int start, int end, std::vector<MmeQmanCmd>& cmds);
    void range2cmdsMme(const Desc* desc, int start, int end, std::vector<MmeQmanCmd>& cmds);
    void range2cmdsDma(const Desc* desc, int start, int end, std::vector<MmeQmanCmd>& cmds);
    void desc2cmds(const bool fullDesc,
                   const bool mask[sizeof(Desc)],
                   const Desc* currDesc,
                   const Desc* prevDesc,
                   std::vector<MmeQmanCmd>& cmds,
                   Desc* deviceDesc);
    void logMemAccess(const MmeCmd cmd,
                      unsigned mmeIdx,
                      const MmeActivation<Desc>& activation,
                      MmeMemAccessChecker* accessChecker);
    unsigned getWkldID(const unsigned wkldID, const unsigned descIdx);
    void addReductionToPowerActivations(ActivationVec<Desc>& powerActivation, EMmeReductionOp op);
    virtual void addReductionToPowerDesc(Desc& powerDesc, EMmeReductionOp op) = 0;
    virtual void removeReductionToPowerDesc(Desc& powerDesc) = 0;
    virtual void zeroSignalForPowerTest(ActivationVec<Desc>& activation) = 0;

    std::vector<Desc> m_deviceDescMme;
    std::vector<Desc> m_deviceDescDma;
    static const unsigned c_localFenceIdx = 3;
};
}  // namespace MmeCommon
