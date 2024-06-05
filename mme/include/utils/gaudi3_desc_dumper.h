#ifndef MME__GAUDI3_DESC_DUMPER_H
#define MME__GAUDI3_DESC_DUMPER_H

#include "gaudi3/mme.h"
#include "include/utils/descriptor_dumper.h"

class Gaudi3DescriptorDumper : public MmeDescriptorDumper<gaudi3::Mme::Desc>
{
public:
    Gaudi3DescriptorDumper() : MmeDescriptorDumper<gaudi3::Mme::Desc>() {}
    virtual ~Gaudi3DescriptorDumper() = default;

protected:
    virtual void dumpDescriptor(const gaudi3::Mme::Desc& desc, bool fullDump) override;

private:
    void dumpBrain(const gaudi3::Mme::MmeBrainsCtrl& brains);
    void dumpHeader(const gaudi3::Mme::MmeHeader& header);
    void dumpTensor(const gaudi3::Mme::MmeTensorDesc& tensorDesc, const std::string& context, bool dualGemm = false);
    void dumpAgu(const gaudi3::Mme::Desc& desc);
    void dumpConv(const gaudi3::Mme::MmeConvDesc& conv);
    void dumpOuterLoop(const gaudi3::Mme::MmeOuterLoop& outerLoop);
    void dumpSBRepeat(const gaudi3::Mme::MmeSBRepeat& sbRepeat);
    void dumpSyncObject(const gaudi3::Mme::MmeSyncObject& syncObject);
    void dumpNumerics(const gaudi3::Mme::MmeNumericFlavors& numerics);
    void dumpAxiAwUserData(const gaudi3::Mme::MmeAwUserData& awUserData);
    void dumpAxiUserData(const gaudi3::Mme::MmeUserData& userData, const std::string& context);
    void dumpCacheData(const gaudi3::Mme::MmeCacheData& cacheData);
    void dumpPerfEvent(const gaudi3::Mme::MmePerfEvt& perfEvt, const std::string& context);
    void dumpRateLimits(const gaudi3::Mme::MmeRateLimiter& rateLimiter);
    void dumpPower(const gaudi3::Mme::MmePower& power);
};

#endif //MME__GAUDI3_DESC_DUMPER_H
