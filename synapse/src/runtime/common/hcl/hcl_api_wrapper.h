#pragma once
#include "synapse_common_types.h"
#include "hcl_public_streams.h"
#include "scal.h"
#include <memory>
#include "runtime/common/device/dfa_base.hpp"

struct ScalEvent;

class hclApiWrapper
{
public:
    hclApiWrapper()  = default;
    ~hclApiWrapper() = default;

    synStatus provideScal(scal_handle_t scal);
    synStatus createStream(hcl::hclStreamHandle& streamHandle);
    synStatus destroyStream(const hcl::hclStreamHandle& streamHandle) const;
    synStatus eventRecord(ScalEvent* scalEvent, const hcl::hclStreamHandle streamHandle) const;
    synStatus streamWaitEvent(const hcl::hclStreamHandle streamHandle, const ScalEvent* scalEvent) const;
    synStatus synchronizeStream(const hcl::hclStreamHandle streamHandle) const;
    synStatus streamQuery(const hcl::hclStreamHandle streamHandle) const;
    synStatus eventSynchronize(const ScalEvent* scalEvent) const;
    synStatus eventQuery(const ScalEvent* scalEvent) const;
    synStatus checkHclFailure(const DfaStatus dfaStatus, void (*logFunc)(int, const char*), hcl::HclPublicStreams::DfaLogPhase options) const;

private:
    std::unique_ptr<hcl::HclPublicStreams> m_hclPublicStream;
};
