#pragma once

#include "scal.h"
#include "synapse_types.h"

#include "drm/habanalabs_accel.h"

#include "define_synapse_common.hpp"

#include <cstdint>


// A namespace representing the architectural parameters of a Gaudi3 chip

class Gaudi3HalReader;

namespace gaudi3
{
#include "gaudi3/gaudi3.h"

enum class EChipFlavor
{
    DOUBLE_DIE,
    SINGLE_DIE,
    PLDM_ALL_CACHE
};

struct HalChipFlavorSpecificInfo
{
    EChipFlavor chipFlavor;
    unsigned    sramSize;
    unsigned    numMmeEngines;
    unsigned    numTpcEngines;
    uint64_t    tpcEnginesMask;
    unsigned    numDmaEngines;
    unsigned    numRotatorEngines;
    unsigned    numDcores;
    unsigned    hbmBwGBps;
};

struct hal
{
    // following values are common for full and half chip
    static constexpr unsigned clSize                      = 256;
    static constexpr unsigned addressAlignmentSizeInBytes = 256;
    static constexpr unsigned tpcVectorSize               = 256;
    static constexpr unsigned mmeVectorSize               = 256;  //  TODO: delete and move to mmeBrain
    static constexpr unsigned mmeMaximalEUHeight          = 128;  //  TODO: delete and move to mmeBrain
    static constexpr uint64_t sramBaseAddress             = SRAM_BASE_ADDR;
    static constexpr uint64_t dramSize                    = DRAM_MAX_SIZE;  // 256TB
    static constexpr uint64_t dramBaseAddress             = DRAM_PHYS_BASE;
    static constexpr unsigned cpDmaAlignment              = 8;
    static constexpr unsigned prefetchAlignmentMask       = 0xFFFFE000;
    static constexpr unsigned tpcICacheSize               = 64 * 1024;
    static constexpr unsigned numSyncObjects              = 0;    // Gaudi3 works in ARC mode 3
    static constexpr unsigned numMonitors                 = 0;    // Gaudi3 works in ARC mode 3
    static constexpr unsigned firstSyncObjId              = 0;    // Gaudi3 works in ARC mode 3
    static constexpr unsigned firstMonObjId               = 0;    // Gaudi3 works in ARC mode 3
    static constexpr unsigned numInternalDmaEngines       = 0;    // No EDMA allocated for GC
    static constexpr unsigned internalDmaEnginesMask      = 0x0;  // No EDMA allocated for GC
    static constexpr unsigned firstInternalDmaEngineId    = 0;    // No EDMA allocated for GC
    static constexpr float    rotateMaxSupportedAngle     = 59.0;
    static constexpr unsigned numEngineStreams            = 1;
    static constexpr unsigned supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_fp8_143 | syn_type_fp8_152 |
                                               syn_type_int16 | syn_type_uint16 | syn_type_fp16 | syn_type_bf16 |
                                               syn_type_hb_float | syn_type_tf32 | syn_type_int32 | syn_type_uint32 |
                                               syn_type_single;
    static constexpr unsigned supportedMmeTypes = syn_type_bf16 | syn_type_fp16 | syn_type_tf32 | syn_type_single |
                                                  syn_type_hb_float | syn_type_fp8_143 | syn_type_fp8_152 |
                                                  syn_type_ufp16;
    static constexpr bool     mmeCinSupported          = false;  // no bias
    static constexpr bool     isNonLinearDmaSupported  = true;
    static constexpr bool     isSRAMReductionSupported = true;
    static constexpr bool     isDRAMReductionSupported = true;
    static constexpr unsigned clockFreqMHz             = 2000;
    static constexpr uint64_t maxRegSizeforTpcAndLargeDTMme =
        ((uint64_t)1 << 31) - 1;  // the TPC and MME (with DT size > 2 bytes) is allowing 31 bit in the register
    static constexpr uint64_t maxRegSizeforExtendedDimInTpc =
        ((uint64_t)1 << 47) - 1;  // the last dim in the TPC is allowed 47 bit in the register
    static constexpr uint64_t maxRegSizeforDmaAndSmallDTMme =
        ((uint64_t)1 << 32) - 1;  // the DMA and MME (with DT size <= 2 bytes) is allowed 32 bit in the register
    static constexpr uint64_t maxRegSizeForSigned44BitMode =
        ((uint64_t)1 << 43) - 1;  // the TPC has an option working with 43 bit mode
    static constexpr unsigned numTpcEnginesOnDcore            = 16;
    static constexpr unsigned numMmeEnginesOnDcore            = 2;
    static constexpr unsigned numMmeCoresPerEngine            = 2;
    static constexpr unsigned maxNumMmeGemmsPerCore           = 2;
    static constexpr unsigned numInternalDmaEnginesOnDcore    = 0;  // No EDMA allocated for GC
    static constexpr unsigned numRotatorEnginesOnDcore        = 4;
    static constexpr unsigned rotateStripeWidth               = 256;
    static constexpr unsigned rotateStripeHeightStraightAngle = 256;
    static constexpr unsigned baseRegistersCacheSize          = 32;
    static constexpr unsigned skipCache                       = 0x1;  // Cache Directives taken from H9 NoC specs
    static constexpr unsigned noAllocate                      = 0x3;
    static constexpr unsigned homeAllocate                    = 0x7;
    static constexpr unsigned dcoreAllocate                   = 0xB;
    static constexpr unsigned sharedAllocate                  = 0xF;
    static constexpr bool     isIRF44ModeSupported            = true;
    static constexpr unsigned numFastConfigMcidSRFs           = 0;  // Set to 0 till we support SRF use in TPC kernels
    static constexpr unsigned numSRFs                         = 60; // 4 last SRFs are reserved for TPC use
    static constexpr unsigned totalNumBaseRegs                = 32;
    static constexpr unsigned numBaseRegsForAddress           = 28;
    static constexpr unsigned numBaseRegsForMcid              = 4;
    static constexpr unsigned mmeMaxInterleavedSpatialPortsNr = 32;
    // As a workaround for HW bug H6-3374 we have WA adding Wreg32 before use of Engine Barrier
    static constexpr unsigned numUploadKernelEbPad            = 13;

    // As a workaround for HW bug H9-5662, a minimum SRF will make sure the trace context id of end events will be correct
    // (if not the HW bug, no min srf requirement)
    static constexpr unsigned minTpcSRF                       = 8;
};

// following values differ between full and half-chip
static constexpr HalChipFlavorSpecificInfo halFullChipSpecificInfo {
    .chipFlavor        = EChipFlavor::DOUBLE_DIE,
    .sramSize          = 0x0000000006000000ull,  // 96MB
    .numMmeEngines     = 8,
    .numTpcEngines     = 64,
    .tpcEnginesMask    = 0xFFFFFFFFFFFFFFFF,  // 64 engines
    .numDmaEngines     = 6,                   // 2*PDMA + 4*EDMA
    .numRotatorEngines = 8,
    .numDcores         = 4,
    .hbmBwGBps = 2918  // 8(numOfHBM) * 0.85 (efficiency) * 128 (Bytes/cycle) * (3.6Ghz (freq) * 10^9) / 2^30 (GB)
};

static constexpr HalChipFlavorSpecificInfo halHalfChipSpecificInfo {
    .chipFlavor        = EChipFlavor::SINGLE_DIE,
    .sramSize          = 0x0000000006000000ull / 2,  // 96MB / 2 dies
    .numMmeEngines     = 4,
    .numTpcEngines     = 32,
    .tpcEnginesMask    = 0xFFFFFFFF,  // 32 engines
    .numDmaEngines     = 4,           // 2*PDMA + 2*EDMA
    .numRotatorEngines = 4,
    .numDcores         = 2,
    .hbmBwGBps         = 1459  // 2918 GBps / 2 dies
};

static constexpr HalChipFlavorSpecificInfo halAllCacheSpecificInfo {
    .chipFlavor        = EChipFlavor::PLDM_ALL_CACHE,
    .sramSize          = 0x0000000006000000ull,  // 96MB
    .numMmeEngines     = 4,
    .numTpcEngines     = 32,
    .tpcEnginesMask    = 0xFFFFFFFF,  // 32 engines
    .numDmaEngines     = 4,           // 2*PDMA + 2*EDMA
    .numRotatorEngines = 4,
    .numDcores         = 4,
    .hbmBwGBps         = 1459  // 2918 GBps / 2 dies
};

} // namespace gaudi3
