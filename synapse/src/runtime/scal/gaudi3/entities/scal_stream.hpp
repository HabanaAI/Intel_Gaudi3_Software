#pragma once

#include "runtime/scal/common/entities/scal_stream_compute.hpp"

class ScalStreamCopyGaudi3 : public ScalStreamCopySchedulerMode
{
public:
    using ScalStreamCopySchedulerMode::ScalStreamCopySchedulerMode;

protected:
    virtual synStatus memcopyImpl(ResourceStreamType               resourceType,
                                  const internalMemcopyParamEntry* memcpyParams,
                                  uint32_t                         params_count,
                                  bool                             send,
                                  uint8_t                          apiId,
                                  bool                             memsetMode,
                                  bool                             sendUnfence,
                                  uint32_t                         completionGroupIndex,
                                  MemcopySyncInfo&                 memcopySyncInfo) override;
};

class ScalStreamComputeGaudi3 : public ScalStreamCompute
{
public:
    using ScalStreamCompute::ScalStreamCompute;

protected:
    virtual synStatus memcopyImpl(ResourceStreamType               resourceType,
                                  const internalMemcopyParamEntry* memcpyParams,
                                  uint32_t                         params_count,
                                  bool                             send,
                                  uint8_t                          apiId,
                                  bool                             memsetMode,
                                  bool                             sendUnfence,
                                  uint32_t                         completionGroupIndex,
                                  MemcopySyncInfo&                 memcopySyncInfo) override;
};
