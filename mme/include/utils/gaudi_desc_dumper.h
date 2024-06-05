#ifndef MME__GAUDI_DESC_DUMPER_H
#define MME__GAUDI_DESC_DUMPER_H

#include "gaudi/mme.h"
#include "include/utils/descriptor_dumper.h"

class GaudiDescriptorDumper : public MmeDescriptorDumper<Mme::Desc>
{
protected:
    virtual void dumpDescriptor(const Mme::Desc& desc, bool fullDump) override;

private:
    void dumpHeader(const Mme::MmeHeader& header);
    void dumpTensor(const Mme::MmeTensorDesc& tensorDesc, const std::string& context);
    void dumpSyncObject(const Mme::MmeSyncObject& syncObject);
    void dumpAgu(const Mme::Desc& desc);
    void dumpConv(const Mme::MmeConvDesc& conv);
    void dumpOuterLoop(const Mme::MmeOuterLoop& outerLoop);
    void dumpSBRepeat(const Mme::MmeSBRepeat& sbRepeat);
    void dumpAxiUserData(const Mme::MmeUserData& userData);
    void dumpRateLimits(const Mme::MmeRateLimeter& rateLimiter);
    void dumpPerfEvent(const Mme::MmePerfEvt& perfEvt, const std::string& context);
    void dumpPcu(const Mme::MmePCU& pcu);
};

#endif //MME__GAUDI_DESC_DUMPER_H
