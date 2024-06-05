#pragma once

#include "hal_reader/hal_reader.h"

#include <memory>

namespace gaudi3
{
struct hal;
struct HalChipFlavorSpecificInfo;
enum class EChipFlavor;
}

class Gaudi3HalReader final : public HalReader
{
public:
    static const std::shared_ptr<Gaudi3HalReader>& instance();
    // getters
    unsigned                          getSupportedTypes()                       const override;
    unsigned                          getSupportedMmeTypes()                    const override;
    unsigned                          getMmeVectorSize()                        const override;  // MME stack
    unsigned                          getTpcVectorSize()                        const override;
    unsigned                          getCacheLineSizeInBytes()                 const override;
    unsigned                          getAddressAlignmentSizeInBytes()          const override;
    unsigned                          getTPCICacheSize()                        const override;
    unsigned                          getClockFreqMHz()                         const override;
    unsigned                          getHbmBwGBps()                            const override;
    unsigned                          getSramBwGBps()                           const override;
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
    unsigned                          getNumRotatorEngines()                    const override;
    bool                              isRotateAngleSupported(float angle)       const override;
    synDeviceType                     getDeviceType()                           const override;
    unsigned                          getMmeAccumulatorH()                      const override;  // MME stack
    DmaTransposeEngineParams          getDmaTransposeEngineParams()             const override;
    bool                              isDmaTransposeSupported(synDataType type) const override;
    bool                              isMmeTransposeSupported()                 const override;
    unsigned                          getCpDmaAlignment()                       const override;
    unsigned                          getInternalDmaEnginesMask()               const override;
    unsigned                          getBaseRegistersCacheSize()               const override;
    uint64_t                          getDRAMSizeInBytes()                      const override;
    uint64_t                          getSRAMBaseAddr()                         const override;
    uint64_t                          getDRAMBaseAddr()                         const override;
    unsigned                          getMmeSBCacheSize()                       const; // MME stack
    unsigned                          getPrefetchAlignmentMask()                const override;
    unsigned                          getNumMonitors()                          const override;
    unsigned                          getNumSyncObjects()                       const override;
    unsigned                          getFirstInternalDmaEngineId()             const;
    unsigned                          getNumEngineStreams()                     const override;
    unsigned                          getFirstSyncObjId()                       const;
    unsigned                          getFirstMonObjId()                        const;
    unsigned                          getNumDcores()               const override;
    unsigned                          getNumTpcEnginesOnDcore()                 const;
    unsigned                          getNumMmeEnginesOnDcore()                 const;
    unsigned                          getNumInternalDmaEnginesOnDcore()         const;
    unsigned                          getNumRotatorEnginesOnDcore()             const;
    unsigned                          getNumFastConfigMcidSRFs()                const;
    unsigned                          getNumSRFs()                              const;
    unsigned                          getTotalNumBaseRegs()                     const;
    unsigned                          getNumBaseRegsForAddress()                const;
    unsigned                          getNumBaseRegsForMcid()                   const;
    unsigned                          getMcidBaseRegsFirstIndex()               const override;
    unsigned                          getNumUploadKernelEbPad()                 const override;
    unsigned                          getTPCMinSRF()                            const override;
    unsigned                          getRotateStripeWidth()                    const override;
    unsigned                          getRotateStripeHeightStraightAngle()      const override;
    unsigned                          getMmeMinimalWidthInElems(synDataType type)    const override; // MME stack
    unsigned                          getMmeMaximalEUHeightInElems(synDataType type) const override; // MME stack
    unsigned                          getMmeSymmetricWidthInElems(synDataType type)  const override;
    unsigned                          getMmeSymmetricHeightInElems(synDataType type) const override;
    unsigned                          getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const override; // MME stack
    synDataType                       getMmeHighPrecisionTypeForPartials()           const override;
    unsigned                          getCacheDirectiveBits(CacheDirective cacheDirective) const override;
    const std::vector<HabanaDeviceType>&  getSupportedDeviceTypes()             const override;
    uint64_t                          getDefaultMaxRegVal()                     const override;
    uint64_t                          getMaxRegValForDma()                      const override;
    uint64_t                          getMaxRegValForMME(unsigned dataTypeSize) const override;
    uint64_t                          getMaxRegValForExtendedDimInTPC()         const override;
    uint64_t                          getMaxRegValForSigned44BitMode()          const override;
    bool                              isReducibleMemory(MemoryType memoryType)  const override {return true;};
    bool                              isAsicSupportIRF44Mode()                  const override;
    HabanaDeviceType                  getTransposeEngine()                      const override;
    HabanaDeviceType                      getBroadcastEngine() const override;
    // checkers
    bool                              isMmeCinSupported()                       const override;
    bool                              isSRAMReductionSupported()                const override;
    bool                              isDRAMReductionSupported()                const override;
    bool                              isMMETransposeAandTransposeBSupported()   const override;
    bool                              isDuplicateMmeOutputSupported()           const override;
    bool                              isCacheSupported()                        const override { return true; }
    bool                              isTPCMemcpySupportedDataType(synDataType type) const override;
    bool                              isTPCMemsetSupportedDataType(synDataType type) const override;
    unsigned                          getMmeMaxInterleavedSpatialPortsNr() const override;
    bool                              isGcDmaSupported() const override { return false; }

    bool                              isInvScaleExpBiasHWAligned(float invScaleTensor) const override;
    bool                              isScaleExpBiasHWAligned(float scaleTensor)       const override;

    Gaudi3HalReader(const Gaudi3HalReader&) = delete;
    Gaudi3HalReader& operator=(const Gaudi3HalReader&) = delete;

private:
    Gaudi3HalReader(gaudi3::EChipFlavor chipFlavor);
    // following is different between full and half chip
    const gaudi3::HalChipFlavorSpecificInfo& m_halChipFlavorSpecificInfo;
};
