#pragma once

#include "sync_object_manager.h"

namespace gaudi3
{
class Gaudi3SyncObjectManager : public MmeCommon::SyncObjectManager
{
public:
    Gaudi3SyncObjectManager(uint64_t smBase, unsigned mmeLimit);

protected:
    virtual void addMonitor(CPProgram& prog,
                            const SyncInfo& syncInfo,
                            uint32_t monitorIdx,
                            uint64_t agentAddr,
                            unsigned agentPayload) override;
    virtual uint64_t getSoAddress(unsigned soIdx) const override;
    virtual unsigned uploadDmaQidFromStream(unsigned stream) override;
};
}  // namespace gaudi3