#pragma once

#include "drm/habanalabs_accel.h"
#include "gaudi/gaudi.h"
#include "hal_reader/hal_reader.h"
#include "synapse_types.h"

#include <cstdint>


//A class representing the architectural parameters of a Gaudi card

class GaudiHalReader;

namespace gaudi
{

    class Hal
    {
    protected:
        friend class ::GaudiHalReader; // access is allowed ONLY via the HAL reader

        const unsigned clSize                      = 128; // cache line size
        const unsigned addressAlignmentSizeInBytes = 128;
        const unsigned mmeVectorSize               = 128;
        const unsigned tpcVectorSize               = 256;
        const unsigned mmeSBCacheSize              = 8;
        const uint64_t sramBaseAddress             = SRAM_BASE_ADDR + GAUDI_DRIVER_SRAM_RESERVED_SIZE_FROM_START;
        const unsigned SramSize                    = SRAM_SIZE - GAUDI_DRIVER_SRAM_RESERVED_SIZE_FROM_START;
        const uint64_t DramSize                    = 32L * 1024L * 1024L * 1024L; //32 GB
        const uint64_t DramBaseAddress             = 0x20000000;
        const unsigned CPDmaAlignment              = 32;
        const unsigned prefetchAlignmentMask       = 0xFFFFE000;
        const unsigned numMonitors                 = 512;
        const unsigned numSyncObjects              = 2048;
        const unsigned tpcICacheSize               = 64 * 1024;
        const unsigned numTpcEngines               = 8;
        const unsigned tpcEnginesMask              = 0xFF; //8 engines
        const unsigned numMmeEngines               = 2;
        const unsigned numMmeCoresPerEngine        = 2;
        const unsigned maxNumMmeGemmsPerCore       = 1;
        const unsigned numDmaEngines               = 8;
        const unsigned numNicEngines               = 10;
        const uint64_t maxRegSizeforTpcAndLargeDTMme = ((uint64_t)1<<31) - 1; // the TPC and MME (with DT size > 2 bytes) is allowing 31 bit in the register
        const uint64_t maxRegSizeforExtendedDimInTpc = ((uint64_t)1<<47) - 1; // the last dim in the TPC is allowed 47 bit in the register
        const uint64_t maxRegSizeforDmaAndSmallDTMme = ((uint64_t)1<<32) - 1; // the DMA and MME (with DT size <= 2 bytes) is allowed 32 bit in the register
        const bool     isIRF44ModeSupported          = false;
        const unsigned mmeMaxInterleavedSpatialPortsNr = 4;

        // DMA engines for internal Dram Sram transferres subjected to the internalDmaEnginesMask
        const unsigned numInternalDmaEngines    = 6;

        // As the project evolved, engine #3 of the internal engines (i.e. engine #5 globally) has been converted
        // to external engine, so we use mask (like in TPC) to drop it from the quota of the internal engines
        const unsigned internalDmaEnginesMask   = 0x37;

        // 0 and 1 are reserved for Host->Device and Device->Host respectively
        const unsigned firstInternalDmaEngineId = 2;

        const unsigned numEngineStreams         = 4;
        unsigned       supportedTypes           = 0; //set by derived class
        const unsigned supportedMmeTypes        = syn_type_single | syn_type_bf16;
        const bool     mmeCinSupported          = false;
        const bool     isNonLinearDmaSupported  = true;
        const bool     isSRAMReductionSupported = true;
        const bool     isDRAMReductionSupported = false;
        const unsigned clockFreqMHz             = 1900;
        const unsigned hbmBwGBps                = 950;
        const unsigned sramBwGBps               = 1045;
        const unsigned dmaBwGBps                = 894;  // 5(numofDMA) * 128 (Bytes/cycle) * 1.5Ghz (freq)
        const unsigned numPredicateBits         = 31;

        // As a workaround for HW bug H3-2069, a minimum SRF will make sure the trace context id of begin events will be correct
        // (if not the HW bug, no min srf requirement)
        unsigned TPCMinSRF                 = 7;

        const unsigned dmaTransposeEngineMaxSrc0InBytes   = 64;
        const unsigned dmaTransposeEngineMaxDst0InBytes   = 128;
        const unsigned dmaTransposeEngineNumLinesDivisor  = 128;
        virtual void concrete() = 0; // make it abstract class
        };

    class HalGaudiAChip : public Hal
    {
    private:
        friend class ::GaudiHalReader; // access is allowed ONLY via the HAL reader

        HalGaudiAChip()
        {
            supportedTypes = syn_type_int8 | syn_type_uint8 | syn_type_int16 | syn_type_single | syn_type_bf16 | syn_type_int32 | syn_type_uint16 | syn_type_uint32;
        }

        virtual void concrete() override {}
        HalGaudiAChip(const HalGaudiAChip&) = delete;
        HalGaudiAChip& operator=(const HalGaudiAChip&) = delete;
    };
}
