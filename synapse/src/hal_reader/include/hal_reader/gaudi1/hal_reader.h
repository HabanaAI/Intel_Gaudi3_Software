#pragma once

#include "hal_reader/hal_reader.h"
#include "synapse_common_types.h"

namespace gaudi
{
    class Hal;
}


class GaudiHalReader : public HalReader
{
public:
    static const HalReaderPtr& instance(synDeviceType deviceType);
    // getters
    virtual unsigned                         getSupportedTypes()              const override;
    virtual unsigned                         getSupportedMmeTypes()           const override;
    virtual unsigned                         getMmeVectorSize()               const override;
    virtual unsigned                         getTpcVectorSize()               const override;
    virtual unsigned                         getCacheLineSizeInBytes()        const override;
    virtual unsigned                         getAddressAlignmentSizeInBytes() const override;
    virtual unsigned                         getTPCICacheSize()               const override;
    virtual unsigned                         getHbmBwGBps()                   const override;
    virtual unsigned                         getSramBwGBps()                  const override;
    virtual unsigned                         getClockFreqMHz()                const override;
    virtual bool                             isNonLinearDmaSupported()        const override;
    virtual unsigned                         getSRAMSizeInBytes()             const override;
    virtual unsigned                         getNumTpcEngines()               const override;
    virtual unsigned                         getNumMmeEngines()               const override;
    virtual unsigned                         getNumMmeCoresPerEngine()        const override;
    virtual unsigned                         getNumMmeMaxGemmsPerCore()       const override;
    virtual unsigned                         getNumDmaEngines()               const override;
    virtual unsigned                         getNumInternalDmaEngines()       const override;
    virtual unsigned                         getInternalDmaEnginesMask()      const override;
    virtual unsigned                         getDmaBwGBps()                   const override;
    virtual synDeviceType                    getDeviceType()                  const override;
    virtual unsigned                         getMmeAccumulatorH()             const override;
    virtual DmaTransposeEngineParams         getDmaTransposeEngineParams()    const override;
    virtual unsigned                         getCpDmaAlignment()              const override;
    virtual unsigned                         getNumSyncObjects()              const override;
    virtual unsigned                         getNumMonitors()                 const override;
    virtual uint64_t                         getSRAMBaseAddr()                const override;
    virtual uint64_t                         getDRAMBaseAddr()                const override;
    virtual unsigned                         getPrefetchAlignmentMask()       const override;
    virtual uint64_t                         getTpcEnginesMask()              const override;
    virtual synDataType                      getDmaTransposeSupportedDataType(TSize dataTypeSize) const override;
    // TODO SW-47037 - move the following funcs to new class with MME HAL logic
    unsigned                                 getMmeMinimalWidthInElems(synDataType type) const override;
    unsigned                                 getMmeMaximalEUHeightInElems(synDataType type) const override;
    unsigned                                 getMmeSymmetricWidthInElems(synDataType type)  const override;
    unsigned                                 getMmeSymmetricHeightInElems(synDataType type) const override;
    unsigned                                 getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const override;
    uint64_t                                 getDefaultMaxRegVal()                     const override;
    uint64_t                                 getMaxRegValForDma()                      const override;
    uint64_t                                 getMaxRegValForMME(unsigned dataTypeSize) const override;
    uint64_t                                 getMaxRegValForExtendedDimInTPC()         const override;
    bool                                     isReducibleMemory(MemoryType memoryType)  const override;
    bool                                     isAsicSupportIRF44Mode()                  const override;
    unsigned                                 getMcidBaseRegsFirstIndex()               const override;
    unsigned                                 getMmeMaxInterleavedSpatialPortsNr() const override;

    virtual const std::vector<HabanaDeviceType>& getSupportedDeviceTypes()     const override;

    // checkers
    virtual bool     isMmeCinSupported()                  const override;
    virtual bool     isSRAMReductionSupported()           const override;
    virtual bool     isDRAMReductionSupported()           const override;
    virtual bool     isDmaTransposeSupported(synDataType type) const override;
    virtual bool     isMMETransposeAandTransposeBSupported()   const override;
    bool             isTPCMemcpySupportedDataType(synDataType type) const override;
    bool             isTPCMemsetSupportedDataType(synDataType type) const override;

    unsigned                                 getTPCMinSRF()                const override;
    unsigned                                 getNumNicEngines()            const override;
    uint64_t                                 getDRAMSizeInBytes()          const override;
    unsigned                                 getNumEngineStreams()         const override;
    unsigned                                 getNumPredicateBits()         const override;
    unsigned                                 getDmaTransposeEngineMaxDst0InBytes() const override;
    bool                                     isGcDmaSupported() const override { return true; }

    GaudiHalReader(const GaudiHalReader&) = delete;
    GaudiHalReader& operator=(const GaudiHalReader&) = delete;

private:
    std::unique_ptr<const gaudi::Hal> getHalFlavor(synDeviceType deviceType);
    GaudiHalReader(synDeviceType deviceType);

    const std::unique_ptr<const gaudi::Hal> m_halFlavor;
    const synDeviceType               m_deviceType;
};
