#pragma once

#include "dma_transpose_engine_params.h"
#include "habana_device_types.h"
#include "cache_types.h"
#include "synapse_common_types.h"
#include "synapse_types.h"

#include <memory>
#include <vector>

class HalReader
{
public:
    // getters
    virtual unsigned                         getSupportedTypes()                      const = 0;
    virtual unsigned                         getSupportedMmeTypes()                   const = 0;
    virtual unsigned                         getMmeVectorSize()                       const = 0;
    virtual unsigned                         getTpcVectorSize()                       const = 0;
    virtual unsigned                         getCacheLineSizeInBytes()                const = 0;
    virtual unsigned                         getAddressAlignmentSizeInBytes()         const = 0;
    virtual unsigned                         getTPCICacheSize()                       const = 0;
    virtual unsigned                         getClockFreqMHz()                        const = 0;
    virtual unsigned                         getHbmBwGBps()                           const = 0;
    virtual unsigned                         getSramBwGBps()                          const = 0;
    virtual bool                             isNonLinearDmaSupported()                const = 0;
    virtual unsigned                         getSRAMSizeInBytes()                     const = 0;
    virtual unsigned                         getNumTpcEngines()                       const = 0;
    virtual unsigned                         getNumMmeEngines()                       const = 0;
    virtual unsigned                         getNumMmeCoresPerEngine()                const = 0;
    virtual unsigned                         getNumMmeMaxGemmsPerCore()               const = 0;
    virtual unsigned                         getNumDmaEngines()                       const = 0;
    virtual unsigned                         getNumInternalDmaEngines()               const = 0;
    virtual unsigned                         getDmaBwGBps()                           const = 0;
    virtual synDeviceType                    getDeviceType()                          const = 0;
    virtual unsigned                         getMmeAccumulatorH()                     const = 0;
    virtual DmaTransposeEngineParams         getDmaTransposeEngineParams()            const = 0;
    virtual bool                             isDmaTransposeSupported(synDataType type) const = 0;
    virtual synDataType getDmaTransposeSupportedDataType(TSize dataTypeSize) const { return syn_type_na; }
    virtual uint64_t                         getTpcEnginesMask()                      const = 0;
    virtual unsigned                         getInternalDmaEnginesMask()              const { return 0; }
    virtual unsigned                         getNumRotatorEngines()                   const { return 0; }
    virtual unsigned                         getRotateStripeWidth()                   const { return 0; }
    virtual unsigned                         getRotateStripeHeightStraightAngle()     const { return 0; }
    virtual float                            getRotateMaxSupportedAngle()             const { return 0.0; }
    virtual bool                             isRotateAngleSupported(float angle)      const { return false; }
    virtual unsigned                         getTPCMinSRF()                           const { return 0; }
    virtual unsigned                         getNumNicEngines()                       const { return 0; }
    virtual uint64_t                         getDRAMSizeInBytes()                     const { return 0; }
    virtual unsigned                         getNumEngineStreams()                    const { return 0; }
    virtual unsigned                         getNumPredicateBits()                    const { return 0; }
    virtual unsigned                         getDmaTransposeEngineMaxDst0InBytes()    const { return 0; }
    virtual unsigned                         getCpDmaAlignment()                      const = 0;
    virtual unsigned                         getNumSyncObjects()                      const { return 0; }
    virtual unsigned                         getNumMonitors()                         const { return 0; }
    virtual uint64_t                         getSRAMBaseAddr()                        const { return 0; }
    virtual uint64_t                         getDRAMBaseAddr()                        const { return 0; }
    virtual unsigned                         getPrefetchAlignmentMask()               const { return 0; }
    virtual unsigned                         getBaseRegistersCacheSize()              const { return 0; }
    virtual unsigned                         getNumUploadKernelEbPad()                const { return 0; }
    virtual unsigned                         getSyncObjectMaxValue()                  const { return 32768; }
    virtual double                           getDmaMinimalOverhead()                  const { return 0; } // The minimal overhead in usec of dma engine activation
                                                                                                         // Needed for DMA cost model
    virtual unsigned                         getMmeMinimalWidthInElems(synDataType type)  const = 0;
    virtual unsigned                         getMmeMaximalEUHeightInElems(synDataType type) const = 0;
    virtual unsigned                         getMmeSymmetricWidthInElems(synDataType type)  const = 0;
    virtual unsigned                         getMmeSymmetricHeightInElems(synDataType type) const = 0;
    virtual unsigned                         getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const  = 0;
    virtual synDataType                      getMmeHighPrecisionTypeForPartials() const { return syn_type_float; }
    virtual unsigned                         getMmeMaxInterleavedSpatialPortsNr() const { return 0; }

    virtual const std::vector<HabanaDeviceType>&    getSupportedDeviceTypes()         const = 0;
    virtual HabanaDeviceType                        getTransposeEngine() const { return HabanaDeviceType::DEVICE_EDMA; }
    virtual HabanaDeviceType                        getBroadcastEngine() const { return HabanaDeviceType::DEVICE_EDMA; }

    virtual uint32_t                         getDCacheLineSize()              const { return getCacheLineSizeInBytes(); }
    virtual uint32_t                         getDCacheLineNr()                const { return 0; } // 0 means we are not using TPC DCache Prefetcher
    virtual uint32_t                         getDCacheMaxStride()             const { return 24; }
    virtual uint32_t                         getDCacheMinStride()             const { return 2; }
    virtual unsigned                         getNumDcores()      const { return 1; }
    virtual unsigned                         getNumMmeEnginesWithSlaves()     const;
    virtual uint64_t                         getDefaultMaxRegVal()            const = 0;
    virtual uint64_t                         getMaxRegValForDma()             const = 0;
    virtual uint64_t                         getMaxRegValForMME(unsigned dataTypeSize) const = 0;
    virtual uint64_t                         getMaxRegValForExtendedDimInTPC() const = 0;
    virtual uint64_t                         getMaxRegValForSigned44BitMode() const {return 0;}; // for Asics that aren't supported
    virtual unsigned                         getCacheDirectiveBits(CacheDirective cacheDirective) const {return NotSupported;}
    virtual bool                             isReducibleMemory(MemoryType memoryType) const = 0;
    virtual bool                             isAsicSupportIRF44Mode() const = 0;
    virtual unsigned                         getMcidBaseRegsFirstIndex() const = 0;

    // checkers
    virtual bool     isMmeCinSupported()                                  const = 0;
    bool             isSupportedDataType(synDataType type)                const;
    bool             isSupportedMmeDataType(synDataType type)             const;
    virtual bool     isTPCMemcpySupportedDataType(synDataType type) const = 0;
    virtual bool     isTPCMemsetSupportedDataType(synDataType type) const = 0;
    virtual bool     isSupportedMmeInputDataType(synDataType type)        const;
    virtual bool     isSRAMReductionSupported()                           const = 0;
    virtual bool     isDRAMReductionSupported()                           const = 0;
    virtual bool     isMMETransposeAandTransposeBSupported()              const = 0;
    virtual bool     isDuplicateMmeOutputSupported()                      const { return false; }
    virtual bool     isMmeTransposeSupported()                            const { return false; }
    virtual bool     isCacheSupported()                                   const { return false; }
    virtual bool     isGcDmaSupported()                                   const = 0;
    virtual bool     isInvScaleExpBiasHWAligned(float invScale)           const { return false; }
    virtual bool     isScaleExpBiasHWAligned(float scale)                 const { return false; }
};

using HalReaderPtr = std::shared_ptr<HalReader>;
