#ifndef MME__GAUDI3_MME_SIGNAL_INFO_H
#define MME__GAUDI3_MME_SIGNAL_INFO_H

#include "gaudi3/mme.h"
#include "include/mme_common/mme_signal_info.h"

namespace gaudi3
{
class Gaudi3SignalingInfo : public MmeCommon::CommonSignalingInfo<Mme::Desc>
{
public:
    Gaudi3SignalingInfo() = default;
    virtual ~Gaudi3SignalingInfo() = default;

protected:
    virtual void setSlaveSyncObjectAddr(unsigned colorSet, uint32_t addr, Mme::Desc* desc) const override;
    virtual unsigned getAguOutLoopMask(unsigned outputPortIdx, const Mme::Desc* desc) const override;
    virtual void handleNonStoreDescOnSignal(Mme::Desc* desc) const override;
};

}  // namespace gaudi3

#endif //MME__GAUDI3_MME_SIGNAL_INFO_H
