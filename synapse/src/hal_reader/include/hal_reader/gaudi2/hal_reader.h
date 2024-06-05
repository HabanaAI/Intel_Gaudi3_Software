#pragma once

#include "hal_reader/hal_reader.h"

#include <memory>

namespace gaudi2
{
    class hal;
}

class Gaudi2HalReader final : public HalReader
{
public:
    static const std::shared_ptr<Gaudi2HalReader>& instance();
    // getters
    unsigned                          getSupportedTypes()                       const override;
    unsigned                          getSupportedMmeTypes()                    const override;
    unsigned                          getMmeVectorSize()                        const override;
    unsigned                          getTpcVectorSize()                        const override;
    unsigned                          getCacheLineSizeInBytes()                 const override;
    unsigned                          getAddressAlignmentSizeInBytes()          const override;
    unsigned                          getTPCICacheSize()                        const override;
    bool                              isNonLinearDmaSupported()                 const override;
    unsigned                          getSRAMSizeInBytes()                      const override;
    unsigned                          getNumTpcEngines()                        const override;
    uint64_t                          getTpcEnginesMask()                       const override;
    unsigned                          getNumMmeEngines()                        const override;
    unsigned                          getNumMmeCoresPerEngine()                 const override;
    unsigned                          getNumMmeMaxGemmsPerCore()                const override;
    unsigned                          getNumDmaEngines()                        const override;
    unsigned                          getNumInternalDmaEngines()                const override;
    unsigned                          getDmaBwGBps()                            const override;
    double                            getDmaMinimalOverhead()                   const override;
    unsigned                          getNumRotatorEngines()                    const override;
    bool                              isRotateAngleSupported(float angle)       const override;
    synDeviceType                     getDeviceType()                           const override;
    unsigned                          getMmeAccumulatorH()                      const override;
    DmaTransposeEngineParams          getDmaTransposeEngineParams()             const override;
    bool                              isDmaTransposeSupported(synDataType type) const override;
    synDataType                       getDmaTransposeSupportedDataType(TSize dataTypeSize) const override;
    unsigned                          getInternalDmaEnginesMask()               const override;
    unsigned                          getCpDmaAlignment()                       const override;
    unsigned                          getBaseRegistersCacheSize()               const override;
    uint64_t                          getDRAMSizeInBytes()                      const override;
    uint64_t                          getSRAMBaseAddr()                         const override;
    uint64_t                          getDRAMBaseAddr()                         const override;
    unsigned                          getMmeSBCacheSize()                       const;
    unsigned                          getPrefetchAlignmentMask()                const override;
    unsigned                          getNumMonitors()                          const override;
    unsigned                          getNumSyncObjects()                       const override;
    unsigned                          getFirstInternalDmaEngineId()             const;
    unsigned                          getNumEngineStreams()                     const override;
    unsigned                          getClockFreqMHz()                         const override;
    unsigned                          getHbmBwGBps()                            const override;
    unsigned                          getSramBwGBps()                           const override;
    unsigned                          getFirstSyncObjId()                       const;
    unsigned                          getFirstMonObjId()                        const;
    unsigned                          getNumTpcEnginesOnDcore()                 const;
    unsigned                          getNumMmeEnginesOnDcore()                 const;
    unsigned                          getNumInternalDmaEnginesOnDcore()         const;
    unsigned                          getNumRotatorEnginesOnDcore()             const;
    unsigned                          getRotateStripeWidth()                    const override;
    unsigned                          getRotateStripeHeightStraightAngle()      const override;
    unsigned                          getMmeMinimalWidthInElems(synDataType type) const override;
    unsigned                          getMmeMaximalEUHeightInElems(synDataType type) const override;
    unsigned                          getMmeSymmetricWidthInElems(synDataType type)  const override;
    unsigned                          getMmeSymmetricHeightInElems(synDataType type) const override;
    unsigned                          getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const override;
    unsigned                          getNumUploadKernelEbPad()                 const override;

    const std::vector<HabanaDeviceType>&  getSupportedDeviceTypes()             const override;

    uint32_t                         getDCacheLineSize()                        const override;
    uint32_t                         getDCacheLineNr()                          const override;
    uint32_t                         getDCacheMaxStride()                       const override;
    uint32_t                         getDCacheMinStride()                       const override;
    uint64_t                         getDefaultMaxRegVal()                      const override;
    uint64_t                         getMaxRegValForDma()                       const override;
    uint64_t                         getMaxRegValForMME(unsigned dataTypeSize)  const override;
    uint64_t                         getMaxRegValForExtendedDimInTPC()          const override;
    uint64_t                         getMaxRegValForSigned44BitMode()           const override;
    bool                             isReducibleMemory(MemoryType memoryType)   const override {return true;};
    bool                             isAsicSupportIRF44Mode()                   const override;
    unsigned                         getMcidBaseRegsFirstIndex()                const override;

    // checkers
    bool             isMmeCinSupported()                                        const override;
    bool             isSRAMReductionSupported()                                 const override;
    bool             isDRAMReductionSupported()                                 const override;
    bool             isMMETransposeAandTransposeBSupported()                    const override;
    bool             isDuplicateMmeOutputSupported()                            const override;
    bool             isTPCMemcpySupportedDataType(synDataType type)             const override;
    bool             isTPCMemsetSupportedDataType(synDataType type)             const override;
    unsigned         getMmeMaxInterleavedSpatialPortsNr()                       const override;
    bool             isGcDmaSupported()                                         const override { return true; }
    bool             isInvScaleExpBiasHWAligned(float invScale)                 const override;
    bool             isScaleExpBiasHWAligned(float scale)                       const override;

    Gaudi2HalReader(const Gaudi2HalReader&) = delete;
    Gaudi2HalReader& operator=(const Gaudi2HalReader&) = delete;

private:
    Gaudi2HalReader() = default;
};
