#pragma once

#include <string>

#include "synapse_common_types.h"
#include "synapse_api_types.h"
#include "hcl_public_streams.h"
#include "internal/dfa_defines.hpp"
#include "hl_logger/hllog_core.hpp"

synStatus synGenerateApiId(uint8_t& rApiId);

synStatus synStreamSyncHCLStreamHandle(synStreamHandle streamHandle);

synStatus synStreamIsInitialized(synStreamHandle streamHandle, bool& rIsInitialized);

uint32_t synStreamGetPhysicalQueueOffset(synStreamHandle streamHandle);

synStatus synStreamFlushWaitsOnCollectiveHandle(synStreamHandle streamHandle);

hcl::hclStreamHandle synStreamGetHclStreamHandle(synStreamHandle streamHandle);

uint64_t hclNotifyFailure(DfaErrorCode dfaErrorCode, uint64_t options);
uint64_t hclNotifyFailureV2(DfaErrorCode dfaErrorCode, uint64_t options, std::string msg);

struct DfaLoggersV3
{
    hl_logger::LoggerSPtr dfaSynDevFailLogger;
    hl_logger::LoggerSPtr dfaDmesgLogger;
    hl_logger::LoggerSPtr dfaFailedRecipeLogger;
    hl_logger::LoggerSPtr dfaNicInfoLogger;
    hl_logger::LoggerSPtr dfaApi;
    hl_logger::LoggerSPtr dfaApiInfo;
};

DfaLoggersV3 getDfaLoggersV3();

struct DfaLoggersV2
{
    hl_logger::LoggerSPtr dfaSynDevFailLogger;
    hl_logger::LoggerSPtr dfaDmesgLogger;
    hl_logger::LoggerSPtr dfaFailedRecipeLogger;
    hl_logger::LoggerSPtr dfaNicInfoLogger;
};

DfaLoggersV2 getDfaLoggersV2();