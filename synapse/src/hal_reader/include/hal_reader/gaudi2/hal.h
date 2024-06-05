#pragma once

#include "drm/habanalabs_accel.h"
#include "synapse_types.h"
#include "scal.h"

#include "define_synapse_common.hpp"

#include <cstdint>


// A namespace representing the architectural parameters of a Gaudi2 chip

class Gaudi2HalReader;

namespace gaudi2
{
#include "gaudi2/gaudi2.h"

class hal
{
public:
    static constexpr unsigned clSize                            = 128;
    static constexpr unsigned addressAlignmentSizeInBytes       = 128;
    static constexpr unsigned mmeVectorSize                     = 128;
    static constexpr unsigned tpcVectorSize                     = 256;
    static constexpr unsigned mmeSBCacheSize                    = 8;
    static constexpr uint64_t sramBaseAddress                   = SRAM_BASE_ADDR;
    static constexpr unsigned sramSize                          = SRAM_SIZE - SCAL_RESERVED_SRAM_SIZE_H6;
    static constexpr uint64_t dramSize                          = 96L * 1024L * 1024L * 1024L; //96 GB
    static constexpr uint64_t dramBaseAddress                   = DRAM_PHYS_BASE;
    static constexpr unsigned cpDmaAlignment                    = 8;
    static constexpr unsigned prefetchAlignmentMask             = 0xFFFFE000;
    static constexpr uint64_t maxRegSizeforTpcAndLargeDTMme     = ((uint64_t)1<<31) - 1; // the TPC and MME (with DT size > 2 bytes) is allowing 31 bit in the register
    static constexpr uint64_t maxRegSizeforExtendedDimInTpc     = ((uint64_t)1<<47) - 1; // the last dim in the TPC is allowed 47 bit in the register
    static constexpr uint64_t maxRegSizeforDmaAndSmallDTMme     = ((uint64_t)1<<32) - 1; // the DMA and MME (with DT size <= 2 bytes) is allowed 32 bit in the register
    static constexpr uint64_t maxRegSizeForSigned44BitMode      = ((uint64_t)1<<43) - 1; // the TPC has an option working with 43 bit mode


    static constexpr uint32_t dCacheLineNr    = 32;
    static constexpr uint32_t dCacheMaxStride = 24;  // we do not want to fill up the whole cache
    static constexpr uint32_t dCacheMinStride = 2;   // 1 line stride does not contribute to the performance -
                                   // prefetches will not happen, as there always be a miss

    // 1)
    // Although the HW has 8K sync objects, the current implementation of the monitor setup and arm in GC is
    // not accustomed to support more than 2K sync objects. In the monitor setup we do not program the field
    // mon_config.msb_sid that specifies on which quarter of the 8K SOBs we are working on, and in the monitor
    // arm we do not divide the SOB ID so it will fit into the 8 bit we have in sync_group_id. Since we are
    // moving to the new sync scheme that puts the responsibility of the physical objects on the Arc, and GC
    // is working in the virtual domain, we choose not to fix these gaps now. -- SW-43485
    // 2)
    // The number of available sync objects is further reduced to 632 (GAUDI2_NUM_SYNC_OBJECTS_FOR_GC) since
    // the rest are used by the Arc. We take the first available sync object and monitor and the number of
    // resources we have from the SCAL json config file (scal/configs/default.json).
    // This ensures SCAL and GC don't overlap in their resources so long we support the legacy sync scheme.
    static constexpr unsigned numSyncObjects                    = GAUDI2_NUM_SYNC_OBJECTS_FOR_GC;
    static constexpr unsigned numMonitors                       = GAUDI2_NUM_MONITORS_FOR_GC;
    static constexpr unsigned firstSyncObjId                    = GAUDI2_FIRST_AVAILABLE_SYNC_OBJECT_FOR_GC;
    static constexpr unsigned firstMonObjId                     = GAUDI2_FIRST_AVAILABLE_MONITOR_FOR_GC;

    static constexpr unsigned tpcICacheSize                     = 64 * 1024;
    static constexpr unsigned numTpcEngines                     = 24;
    static constexpr unsigned tpcEnginesMask                    = 0x00FFFFFF; //24 engines
    static constexpr unsigned numMmeEngines                     = 2;
    static constexpr unsigned numMmeCoresPerEngine              = 2;
    static constexpr unsigned maxNumMmeGemmsPerCore             = 2;
    static constexpr unsigned numDmaEngines                     = 10;   //10 DMA engines in total, 8 internals and 2 externals
    static constexpr unsigned numInternalDmaEngines             = 5;    //5 internal engines are given for GC subjected to GCFG_EDMA_NUM_BINNED
    static constexpr unsigned internalDmaEnginesMask            = 0x1F; //mask is subjected to GCFG_EDMA_NUM_BINNED
    static constexpr unsigned numRotatorEngines                 = 2;
    static constexpr float    rotateMaxSupportedAngle           = 59.0;
    static constexpr unsigned firstInternalDmaEngineId          = 0; //starting from 0 in contrast to previous platforms
    static constexpr unsigned numEngineStreams                  = 4; //in legacy mode, in ARC mode the streams are global and not per engine
    static constexpr unsigned supportedTypes                    = syn_type_int8 | syn_type_uint8 | syn_type_fp8_143 | syn_type_fp8_152 | syn_type_int16 | syn_type_uint16 | syn_type_fp16 | syn_type_bf16 | syn_type_hb_float | syn_type_tf32 | syn_type_int32 | syn_type_uint32 | syn_type_single;
    static constexpr unsigned supportedMmeTypes                 = syn_type_bf16 | syn_type_fp16 | syn_type_tf32 | syn_type_single | syn_type_hb_float | syn_type_fp8_143 | syn_type_fp8_152;
    static constexpr bool     mmeCinSupported                   = false; //no bias
    static constexpr bool     isNonLinearDmaSupported           = true;
    static constexpr bool     isSRAMReductionSupported          = true;
    static constexpr bool     isDRAMReductionSupported          = true;
    static constexpr unsigned clockFreqMHz                      = 2000;
    static constexpr unsigned hbmBwGBps                         = 1945;  // 6(numOfHBM) * 0.85 (efficiency) * 128 (Bytes/cycle) * 3.2Ghz (freq)
    static constexpr unsigned sramBwGBps                        = 6554;  // 32 * 4(numOfSRAM) * 128 (Bytes/cycle) * 0.4Ghz (freq)
    static constexpr unsigned dmaBwGBps                         = 970; // 5(numofDMA) * 0.93 (toggle rate) * 128 (Bytes/cycle) * 1.75Ghz (freq)
    static constexpr unsigned mmeAccumulatorH                   = 256;
    static constexpr unsigned dmaTransposeEngineMaxSrc0InBytes  = 128;
    static constexpr unsigned dmaTransposeEngineMaxDst0InBytes  = 128;
    static constexpr unsigned dmaTransposeEngineNumLinesDivisor = 128;
    static constexpr unsigned numTpcEnginesOnDcore              = 6;
    static constexpr unsigned numMmeEnginesOnDcore              = 1;
    static constexpr unsigned numInternalDmaEnginesOnDcore      = 2;
    static constexpr unsigned numRotatorEnginesOnDcore          = 2;
    static constexpr unsigned rotateStripeWidth                 = 128;
    static constexpr unsigned rotateStripeHeightStraightAngle   = 256;
    static constexpr unsigned baseRegistersCacheSize            = 16;
    static constexpr bool     isIRF44ModeSupported              = true;
    static constexpr unsigned mmeMaxInterleavedSpatialPortsNr   = 16;
    // As a workaround for HW bug H6-3374 we have WA adding Wreg32 before use of Engine Barrier
    static constexpr unsigned numUploadKernelEbPad              = 13;
};


} // namespace gaudi2
