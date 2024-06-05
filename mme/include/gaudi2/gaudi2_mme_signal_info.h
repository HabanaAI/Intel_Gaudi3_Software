#ifndef MME__GAUDI2_MME_SIGNAL_INFO_H
#define MME__GAUDI2_MME_SIGNAL_INFO_H

#include "gaudi2/mme.h"
#include "include/mme_common/mme_signal_info.h"

namespace Gaudi2
{
class Gaudi2SignalingInfo : public MmeCommon::CommonSignalingInfo<Mme::Desc>
{
public:
    Gaudi2SignalingInfo() = default;
    virtual ~Gaudi2SignalingInfo() = default;

protected:
    virtual void setSlaveSyncObjectAddr(unsigned colorSet, uint32_t addr, Mme::Desc* desc) const override;
    virtual unsigned getAguOutLoopMask(unsigned outputPortIdx, const Mme::Desc* desc) const override;
    virtual void handleNonStoreDescOnSignal(Mme::Desc* desc) const override;
};

}  // namespace Gaudi2

#endif //MME__GAUDI2_MME_SIGNAL_INFO_H
