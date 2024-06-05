#ifndef MME__SIGNAL_INFO_H
#define MME__SIGNAL_INFO_H

#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
template<typename Desc>
class MmeDescriptorSignalingInfo
{
public:
    void addSignalInfo(const EMmeSignalingMode signalingMode,
                       const bool slaveSignaling,
                       const bool isLast,
                       const bool squashIORois,
                       const unsigned signalAmount,
                       Desc* desc);
    unsigned generateSignalAmountMask(const unsigned signalAmount, const Desc* desc);
    unsigned countSignals(const Desc* desc);
    void patchSyncObject(Desc* desc,
                         const uint32_t addr0,
                         const uint32_t addr1,
                         const uint32_t slaveAddr0,
                         const uint32_t slaveAddr1);
    void patchSignalColoring(Desc& desc, const bool addr0isSram, const bool addr1isSram, bool useSameColorSet);

protected:
    // only instantiate platform specific derivatives.
    MmeDescriptorSignalingInfo() = default;
    ~MmeDescriptorSignalingInfo() = default;

    virtual void resetSignal(unsigned colorSet, Desc* desc) const = 0;
    virtual void setSignal(unsigned colorSet, unsigned mask, Desc* desc, bool signalEnable = true) const = 0;
    virtual bool getSignalEn(unsigned colorSet, const Desc* desc) const = 0;
    virtual EMmeLoopMask getSignalMask(unsigned colorSet, const Desc* desc) const = 0;
    virtual void initializeSlaveSignaling(Desc* desc, bool slaveSignaling = false) const = 0;
    virtual void initializeSyncObjectConfig(Desc* desc) const = 0;
    virtual bool getStoreEn(unsigned outputIdx, const Desc* desc) const = 0;
    virtual unsigned getAguOutLoopMask(unsigned outputIdx, const Desc* desc) const = 0;
    virtual void handleNonStoreDescOnSignal(Desc* desc) const = 0;
    virtual unsigned getOuterLoopSize(const Desc* desc) const = 0;
    virtual unsigned getTetrisLoopSize(const Desc* desc) const = 0;
    virtual unsigned getConvLoopSize(unsigned loop, const Desc* desc) const = 0;
    virtual void setColorSet(unsigned outputIdx, unsigned colorSet, Desc* desc) const = 0;
    virtual uint32_t getColorSet(unsigned outputIdx, Desc* desc) const = 0;
    virtual void setSyncObject(Desc* desc,
                               unsigned colorSet,
                               unsigned signalMask,
                               uint32_t masterAddr,
                               uint32_t slaveAddr = 0,
                               bool signalEnable = true) const = 0;
    virtual void setSlaveSyncObject(Desc* desc, unsigned colorSet, uint32_t slaveAddr) const = 0;
};

// Because most platform descriptors has the same field names - we have this common implementation.
// in case platform has some different field names - it needs to override the relevant method.
template<typename Desc>
class CommonSignalingInfo : public MmeCommon::MmeDescriptorSignalingInfo<Desc>
{
public:
    CommonSignalingInfo() = default;
    virtual ~CommonSignalingInfo() = default;

protected:
    virtual void resetSignal(unsigned colorSet, Desc* desc) const override;
    virtual void setSignal(unsigned colorSet, unsigned mask, Desc* desc, bool signalEnable = true) const override;
    virtual bool getSignalEn(unsigned colorSet, const Desc* desc) const override;
    virtual MmeCommon::EMmeLoopMask getSignalMask(unsigned colorSet, const Desc* desc) const override;
    virtual void initializeSlaveSignaling(Desc* desc, const bool slaveSignaling = false) const override;
    virtual void initializeSyncObjectConfig(Desc* desc) const override;
    virtual void setSlaveSyncObjectAddr(unsigned colorSet, uint32_t addr, Desc* desc) const = 0;
    virtual bool getStoreEn(unsigned outputIdx, const Desc* desc) const override;
    virtual unsigned getOuterLoopSize(const Desc* desc) const override;
    virtual unsigned getTetrisLoopSize(const Desc* desc) const override;
    virtual unsigned getConvLoopSize(unsigned loop, const Desc* desc) const override;
    virtual void setColorSet(unsigned outputIdx, unsigned colorSet, Desc* desc) const override;
    virtual uint32_t getColorSet(unsigned outputIdx, Desc* desc) const override;
    virtual void setSyncObject(Desc* desc,
                               unsigned colorSet,
                               unsigned signalMask,
                               uint32_t masterAddr,
                               uint32_t slaveAddr = 0,
                               bool signalEnable = true) const override;
    virtual void setSlaveSyncObject(Desc* desc, unsigned colorSet, uint32_t slaveAddr) const override;

private:
    constexpr static unsigned c_outer_loop_idx = (1 << 6) - 1;
};

}  // namespace MmeCommon

#endif //MME__SIGNAL_INFO_H
