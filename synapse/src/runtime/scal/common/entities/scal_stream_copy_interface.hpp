#pragma once

#include "scal_stream_base_interface.hpp"

class ScalStreamCopyInterface : public ScalStreamBaseInterface
{
public:
    enum ePdmaTxSyncMechanism
    {
        PDMA_TX_SYNC_MECH_LONG_SO,
        PDMA_TX_SYNC_FENCE_ONLY,
        PDMA_TX_SYNC_NONE
    };

    struct MemcopySyncInfo
    {
        ePdmaTxSyncMechanism m_pdmaSyncMechanism;
        uint64_t             m_workCompletionAddress;
        uint32_t             m_workCompletionValue;
    };

    virtual synStatus memcopy(ResourceStreamType           resourceType,
                              const internalMemcopyParams& memcpyParams,
                              bool                         isUserRequest,
                              bool                         send,
                              uint8_t                      apiId,
                              ScalLongSyncObject&          longSo,
                              uint64_t                     overrideMemsetVal,
                              MemcopySyncInfo&             memcopySyncInfo) = 0;
};
