#pragma once

#include "define_synapse_common.hpp"

#include <memory>

class QueueInterface;

class RecipeProgramBuffer;
typedef std::shared_ptr<RecipeProgramBuffer> SpRecipeProgramBuffer;

class DeviceDownloaderInterface
{
public:
    virtual ~DeviceDownloaderInterface() = default;

    virtual synStatus downloadProgramCodeBuffer(uint64_t               recipeId,
                                                QueueInterface*        pPreviousStream,
                                                internalMemcopyParams& rMemcpyParams,
                                                uint64_t               hostBufferSize) const = 0;

    virtual synStatus downloadProgramDataBuffer(QueueInterface*        pPreviousStream,
                                                internalMemcopyParams& rMemcpyParams,
                                                SpRecipeProgramBuffer* pRecipeProgramDataBuffer) const = 0;
};