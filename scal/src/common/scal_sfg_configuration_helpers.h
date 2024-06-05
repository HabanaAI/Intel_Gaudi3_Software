#pragma once
#include <assert.h>
#include <cstdint>
#include <string>
#include <vector>
#include "scal.h"

enum class EngineTypes
{
    mme,
    tpc,
    edma,
    rot,
    items_count
};

inline std::string getEngTypeStr(EngineTypes type)
{
    switch(type)
    {
        case EngineTypes::mme:  return "mme";
        case EngineTypes::tpc:  return "compute_tpc";
        case EngineTypes::edma: return "compute_edma";
        case EngineTypes::rot:  return "rotator";
        case EngineTypes::items_count: return "invalid";
    }
    return "invalid";
}


template <class TGaudiSobTypes>
struct RearmMonPayload
{
    uint32_t monIdx;
    uint64_t monAddr;  // Addr of the monitor we need to rearm
    uint32_t data;  // Payload data of the monitor
};


struct EngineTypeInfo
{
    uint32_t numOfPhysicalEngines = 0;
    uint32_t sfgSignals = 1;
    std::string engineType;
    uint32_t baseSobIdx;
};

template <class TGaudiSobTypes>
struct SfgMonitorsHierarchyMetaData
{
    uint32_t numOfComputeEngines = 0;
    EngineTypeInfo engines[unsigned(EngineTypes::items_count)];

    // SM info
    uint32_t smIdx;
    uint64_t smBaseAddr;   // monitors + sync objects
    uint32_t longSoSmIdx;
    uint64_t longSoSmBaseAddr;
    uint32_t cqSmIdx;
    uint64_t cqSmBaseAddr;

    // Running SOB and MON ids
    uint32_t baseSobIdx;
    uint32_t baseMonIdx;
    uint32_t curSobIdx;
    uint32_t curMonIdx;
    uint32_t baseCqMonIdx;
    uint32_t curCqMonIdx;

    // Common data for Layers
    uint32_t interSobIdx;  // Intermidiate-SSOB
    uint32_t rearmSobIdx;  // Rearm-SSOB
    uint32_t longSobIdx;   // LongSO
    uint32_t cqIdx;        // CQ ID
    uint32_t maxNbMessagesPerMonitor;

    std::vector<RearmMonPayload<TGaudiSobTypes>> rearmMonitors;

    uint32_t decSobPayloadData = {};
    uint32_t incSobPayloadData = {};
};


