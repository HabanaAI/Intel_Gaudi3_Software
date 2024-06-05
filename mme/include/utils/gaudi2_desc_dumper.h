#ifndef MME__GAUDI2_DESC_DUMPER_H
#define MME__GAUDI2_DESC_DUMPER_H

#include "gaudi2/mme.h"
#include "include/utils/descriptor_dumper.h"

class Gaudi2DescriptorDumper : public MmeDescriptorDumper<Gaudi2::Mme::Desc>
{
public:
    Gaudi2DescriptorDumper() : MmeDescriptorDumper<Gaudi2::Mme::Desc>() {}
    virtual ~Gaudi2DescriptorDumper() = default;

protected:
    virtual void dumpDescriptor(const Gaudi2::Mme::Desc& desc, bool fullDump) override;

private:
    void dumpBrain(const Gaudi2::Mme::MmeBrainsCtrl& brains);
    void dumpCtrl(const Gaudi2::Mme::MmeCtrl& ctrl);
    void dumpHeader(const Gaudi2::Mme::MmeHeader& header);
    void dumpTensor(const Gaudi2::Mme::MmeTensorDesc& tensorDesc, const std::string& context);
    void dumpSyncObject(const Gaudi2::Mme::MmeSyncObject& syncObject);
    void dumpAgu(const Gaudi2::Mme::Desc& desc);
    void dumpConv(const Gaudi2::Mme::MmeConvDesc& conv);
    void dumpOuterLoop(const Gaudi2::Mme::MmeOuterLoop& outerLoop);
    void dumpSBRepeat(const Gaudi2::Mme::MmeSBRepeat& sbRepeat);
    void dumpFp8Bias(const Gaudi2::Mme::MmeFP8Bias fp8Bias);
    void dumpAxiUserData(const Gaudi2::Mme::MmeUserData& userData);
    void dumpRateLimits(const Gaudi2::Mme::MmeRateLimiter& rateLimiter);
    void dumpPerfEvent(const Gaudi2::Mme::MmePerfEvt& perfEvt, const std::string& context);
    void dumpPcu(const Gaudi2::Mme::MmePCU& pcu);
    void dumpPowerLoop(const Gaudi2::Mme::MmePowerLoop& powerLoops);
};

#endif //MME__GAUDI2_DESC_DUMPER_H
