#pragma once

#include "synapse_common_types.h"

#include <cstdint>
#include <vector>

enum eCsDcExecutionType
{
    CS_DC_EXECUTION_TYPE_CS_READY,      // All is ready, just submit
    CS_DC_EXECUTION_TYPE_CP_DMA_READY,  // DCs are ready (only patch and copy the commands' DC)
    CS_DC_EXECUTION_TYPE_NOT_READY      // Full operation is required
};

enum eExecutionStage
{
    EXECUTION_STAGE_ACTIVATE = 0,    // Activate stage
    EXECUTION_STAGE_ENQUEUE  = 0x1,  // Enqueue stage
    EXECUTION_STAGE_LAST     = 0x2   // Not initialized stage
};

enum eCsDcProcessorStatus
{
    CS_DC_PROCESSOR_STATUS_FAILED,               // The processor failed to submit, and have not stored the CS-DC
    CS_DC_PROCESSOR_STATUS_STORED_ONLY,          // The processor failed to submit, but stored the CS-DC, for re-use
    CS_DC_PROCESSOR_STATUS_STORED_AND_SUBMITTED  // The processor submitted the CS-DC, and stored it as an in-flight
};

enum eAnalyzeValidateStatus
{
    ANALYZE_VALIDATE_STATUS_NOT_REQUIRED = 0,
    ANALYZE_VALIDATE_STATUS_DO_ANALYZE   = 0x1,
    ANALYZE_VALIDATE_STATUS_DO_VALIDATE  = 0x2
};

typedef uint64_t blobAddressType;

struct blobAddrToMappedAddr
{
    blobAddressType addr;
    blobAddressType mappedAddr;  // for debug only
    bool            shouldDelete;
};

// A Map between a specific blob's part (device address) to the size of this part

struct DevAddrAndSize
{
    uint64_t devAddr;
    uint64_t size;
};

struct HostAndDevAddr
{
    uint64_t                    hostAddr;
    DevAddrAndSize              devAddrAndSize;
    std::vector<DevAddrAndSize> extraDevAddrAndSize;  // in 99% of the cases, we have only one set of addr/size.
                                                      // So keeping one copy of addr/size as part
                                                      // of the struct and adding a vector only if needed.
};

typedef synInternalQueue       primeQueueCommand;

struct CachedAndNot
{
    uint64_t cached;
    uint64_t notCached;
};
