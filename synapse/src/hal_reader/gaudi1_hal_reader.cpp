#include "hal_reader/gaudi1/hal_reader.h"

#include "hal_reader/gaudi1/hal.h"
#include "type_utils.h"

HalReaderPtr instantiateGaudiHalReader()
{
    return GaudiHalReader::instance(synDeviceGaudi);
}

const HalReaderPtr& GaudiHalReader::instance(synDeviceType deviceType)
{
    static const HalReaderPtr halGaudiAChip(new GaudiHalReader(synDeviceGaudi));
    return halGaudiAChip;
}

std::unique_ptr<const gaudi::Hal> GaudiHalReader::getHalFlavor(synDeviceType deviceType)
{
    return std::unique_ptr<const gaudi::Hal>(new gaudi::HalGaudiAChip());
}

GaudiHalReader::GaudiHalReader(synDeviceType deviceType)
: m_halFlavor(getHalFlavor(deviceType)), m_deviceType(deviceType)
{
}

unsigned GaudiHalReader::getSupportedTypes() const
{
    return m_halFlavor->supportedTypes;
}

unsigned GaudiHalReader::getNumEngineStreams() const
{
    return m_halFlavor->numEngineStreams;
}

unsigned GaudiHalReader::getHbmBwGBps() const
{
    return m_halFlavor->hbmBwGBps;
}

unsigned GaudiHalReader::getSramBwGBps() const
{
    return m_halFlavor->sramBwGBps;
}

unsigned GaudiHalReader::getSupportedMmeTypes() const
{
    return m_halFlavor->supportedMmeTypes;
}

unsigned GaudiHalReader::getDmaTransposeEngineMaxDst0InBytes() const
{
    return m_halFlavor->dmaTransposeEngineMaxDst0InBytes;
}

unsigned GaudiHalReader::getMmeVectorSize() const
{
    return m_halFlavor->mmeVectorSize;
}

unsigned GaudiHalReader::getMmeMinimalWidthInElems(synDataType type) const
{
    return m_halFlavor->mmeVectorSize / dataTypeSizeInBytes(type);
}

unsigned GaudiHalReader::getMmeMaximalEUHeightInElems(synDataType type) const
{
    return m_halFlavor->mmeVectorSize / dataTypeSizeInBytes(type);
}

unsigned GaudiHalReader::getMmeSymmetricWidthInElems(synDataType type) const
{
    return 2 * m_halFlavor->mmeVectorSize / dataTypeSizeInBytes(type);
}

unsigned GaudiHalReader::getMmeSymmetricHeightInElems(synDataType type) const
{
    return m_halFlavor->mmeVectorSize / dataTypeSizeInBytes(type);
}

unsigned GaudiHalReader::getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const
{
    // input: bf16, output: bf16 -> minCD: 128
    // input: fp32, output: fp32 -> minCD: 64
    // input: bf16, output: fp32 -> minCD: 256
    // input: fp32, output: bf16 -> minCD: 32

    unsigned minCDElements = (inDataType == syn_type_bf16 ? getMmeVectorSize() : getMmeVectorSize() / 2);

    if (inDataType != outDataType)
    {
        if (inDataType == syn_type_bf16 && outDataType == syn_type_float)
        {
            minCDElements *= 2;
        }
        else if (inDataType == syn_type_float && outDataType == syn_type_bf16)
        {
            minCDElements /= 2;
        }
    }

    return minCDElements;
}

unsigned GaudiHalReader::getTpcVectorSize() const
{
    return m_halFlavor->tpcVectorSize;
}

unsigned GaudiHalReader::getCacheLineSizeInBytes() const
{
    return m_halFlavor->clSize;
}

unsigned GaudiHalReader::getAddressAlignmentSizeInBytes() const
{
    return m_halFlavor->addressAlignmentSizeInBytes;
}

uint64_t GaudiHalReader::getDRAMSizeInBytes() const
{
    return m_halFlavor->DramSize;
}

unsigned GaudiHalReader::getPrefetchAlignmentMask() const
{
    return m_halFlavor->prefetchAlignmentMask;
}

unsigned GaudiHalReader::getTPCMinSRF() const
{
    return m_halFlavor->TPCMinSRF;
}

unsigned GaudiHalReader::getNumNicEngines() const
{
    return m_halFlavor->numNicEngines;
}

unsigned GaudiHalReader::getClockFreqMHz() const
{
    return m_halFlavor->clockFreqMHz;
}

uint64_t GaudiHalReader::getDRAMBaseAddr() const
{
    return m_halFlavor->DramBaseAddress;
}

bool GaudiHalReader::isMmeCinSupported() const
{
    return m_halFlavor->mmeCinSupported;
}

unsigned GaudiHalReader::getTPCICacheSize() const
{
    return m_halFlavor->tpcICacheSize;
}

unsigned GaudiHalReader::getNumPredicateBits() const
{
    return m_halFlavor->numPredicateBits;
}

unsigned GaudiHalReader::getNumSyncObjects() const
{
    return m_halFlavor->numSyncObjects;
}

unsigned GaudiHalReader::getNumMonitors() const
{
    return m_halFlavor->numMonitors;
}

uint64_t GaudiHalReader::getSRAMBaseAddr() const
{
    return m_halFlavor->sramBaseAddress + GCFG_SRAM_SIZE_RESERVED_FOR_HCL.value();
}

bool GaudiHalReader::isNonLinearDmaSupported() const
{
    return m_halFlavor->isNonLinearDmaSupported;
}

bool GaudiHalReader::isSRAMReductionSupported() const
{
    return m_halFlavor->isSRAMReductionSupported;
}

bool GaudiHalReader::isDRAMReductionSupported() const
{
    return m_halFlavor->isDRAMReductionSupported;
}

unsigned GaudiHalReader::getSRAMSizeInBytes() const
{
    unsigned sramSize = m_halFlavor->SramSize - GCFG_SRAM_SIZE_RESERVED_FOR_HCL.value();
    if (GCFG_SET_SRAM_SIZE.value() != 0)
    {
        sramSize = std::min((unsigned)GCFG_SET_SRAM_SIZE.value(), sramSize);
    }
    return sramSize;
}

unsigned GaudiHalReader::getNumTpcEngines() const
{
    return m_halFlavor->numTpcEngines;
}

unsigned GaudiHalReader::getNumMmeEngines() const
{
    return m_halFlavor->numMmeEngines;
}

unsigned GaudiHalReader::getNumMmeCoresPerEngine() const
{
    return m_halFlavor->numMmeCoresPerEngine;
}

unsigned GaudiHalReader::getNumMmeMaxGemmsPerCore() const
{
    return m_halFlavor->maxNumMmeGemmsPerCore;
}

unsigned GaudiHalReader::getNumDmaEngines() const
{
    return m_halFlavor->numDmaEngines;
}

unsigned GaudiHalReader::getNumInternalDmaEngines() const
{
    return m_halFlavor->numInternalDmaEngines;
}

unsigned GaudiHalReader::getInternalDmaEnginesMask() const
{
    return m_halFlavor->internalDmaEnginesMask;
}

unsigned GaudiHalReader::getDmaBwGBps() const
{
    return m_halFlavor->dmaBwGBps;
}

synDeviceType GaudiHalReader::getDeviceType() const
{
    return m_deviceType;
}

unsigned GaudiHalReader::getMmeAccumulatorH() const
{
    HB_ASSERT(false, "getMmeAccumulatorH should not be called for Gaudi" );
    return 0;
}

bool GaudiHalReader::isReducibleMemory(MemoryType memoryType) const
{
    return (memoryType == MEMORY_TYPE_SRAM);
}

DmaTransposeEngineParams GaudiHalReader::getDmaTransposeEngineParams() const {
    return DmaTransposeEngineParams{
        .maxSrc0 = m_halFlavor->dmaTransposeEngineMaxSrc0InBytes,
        .maxDst0 = m_halFlavor->dmaTransposeEngineMaxDst0InBytes,
        .numLinesDivisor = m_halFlavor->dmaTransposeEngineNumLinesDivisor};
}

bool GaudiHalReader::isDmaTransposeSupported(synDataType dtype) const
{
    switch(dtype)
    {
        case syn_type_bf16:
        case syn_type_fp16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_single:
        case syn_type_int32:
        case syn_type_uint32:
            return true;
        default:
            return false;
    };
}

synDataType GaudiHalReader::getDmaTransposeSupportedDataType(TSize dataTypeSize) const
{
    switch (dataTypeSize)
    {
        case 2:
            return syn_type_bf16;
        case 4:
            return syn_type_single;
        default:
            return syn_type_na;
    };
}

bool GaudiHalReader::isMMETransposeAandTransposeBSupported() const
{
    return true;
}

unsigned GaudiHalReader::getCpDmaAlignment() const
{
    return m_halFlavor->CPDmaAlignment;
}

uint64_t GaudiHalReader::getTpcEnginesMask() const
{
    return m_halFlavor->tpcEnginesMask;
}

const std::vector<HabanaDeviceType>& GaudiHalReader::getSupportedDeviceTypes() const
{
    static const std::vector<HabanaDeviceType> SUPPORTED = {HabanaDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL,
                                                            HabanaDeviceType::DEVICE_MME,
                                                            HabanaDeviceType::DEVICE_TPC};
    return SUPPORTED;
}

uint64_t GaudiHalReader::getDefaultMaxRegVal() const
{
    return m_halFlavor->maxRegSizeforTpcAndLargeDTMme;
}

uint64_t GaudiHalReader::getMaxRegValForExtendedDimInTPC() const
{
    return m_halFlavor->maxRegSizeforExtendedDimInTpc;
}

uint64_t GaudiHalReader::getMaxRegValForDma() const
{
    return m_halFlavor->maxRegSizeforDmaAndSmallDTMme;
}

uint64_t GaudiHalReader::getMaxRegValForMME(unsigned dataTypeSize) const
{
    return dataTypeSize > 2 ? m_halFlavor->maxRegSizeforTpcAndLargeDTMme : m_halFlavor->maxRegSizeforDmaAndSmallDTMme;
}

bool GaudiHalReader::isAsicSupportIRF44Mode() const
{
    return m_halFlavor->isIRF44ModeSupported;
}

unsigned GaudiHalReader::getMcidBaseRegsFirstIndex() const
{
    // MCID not supportted for Gaudi
    return 0;
}

unsigned GaudiHalReader::getMmeMaxInterleavedSpatialPortsNr() const
{
    return m_halFlavor->mmeMaxInterleavedSpatialPortsNr;
}

bool GaudiHalReader::isTPCMemcpySupportedDataType(synDataType type) const
{
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_bf16:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_int64:
        case syn_type_uint64:
        case syn_type_single:
            return true;
        default:
            return false;
    }
}

bool GaudiHalReader::isTPCMemsetSupportedDataType(synDataType type) const
{
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_bf16:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_single:
            return true;
        default:
            return false;
    }
}
