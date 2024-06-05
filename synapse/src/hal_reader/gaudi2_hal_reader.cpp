#include "hal_reader/gaudi2/hal_reader.h"

#include "hal_reader/gaudi2/hal.h"
#include "synapse_common_types.h"
#include "utils.h"

#include <bitset>
#include <habana_global_conf.h>

using namespace gaudi2;

constexpr std::array<float, 4> hwAlignedInvScalesFp8143 {{0.0625, 1, 16, 256}};
constexpr std::array<float, 4> hwAlignedScalesFp8143 {{0.00390625, 0.0625, 1, 16}};

HalReaderPtr instantiateGaudi2HalReader()
{
    return Gaudi2HalReader::instance();
}

const std::shared_ptr<Gaudi2HalReader>& Gaudi2HalReader::instance()
{
    static const std::shared_ptr<Gaudi2HalReader> instance(new Gaudi2HalReader);
    return instance;
}

unsigned Gaudi2HalReader::getSupportedTypes() const
{
    return hal::supportedTypes;
}

unsigned Gaudi2HalReader::getSupportedMmeTypes() const
{
    return hal::supportedMmeTypes;
}

unsigned Gaudi2HalReader::getMmeVectorSize() const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi2HalReader::getTpcVectorSize() const
{
    return hal::tpcVectorSize;
}

unsigned Gaudi2HalReader::getCacheLineSizeInBytes() const
{
    return hal::clSize;
}

unsigned Gaudi2HalReader::getAddressAlignmentSizeInBytes() const
{
    return hal::addressAlignmentSizeInBytes;
}

bool Gaudi2HalReader::isMmeCinSupported() const
{
    return hal::mmeCinSupported;
}

unsigned Gaudi2HalReader::getTPCICacheSize() const
{
    return hal::tpcICacheSize;
}

unsigned Gaudi2HalReader::getMmeMinimalWidthInElems(synDataType type) const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi2HalReader::getMmeMaximalEUHeightInElems(synDataType type) const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi2HalReader::getMmeSymmetricWidthInElems(synDataType type) const
{
    return 2 * hal::mmeVectorSize;
}

unsigned Gaudi2HalReader::getMmeSymmetricHeightInElems(synDataType type) const
{
    return 2 * hal::mmeVectorSize;
}

unsigned Gaudi2HalReader::getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const
{
    HB_ASSERT((inDataType != syn_type_ufp16) && (outDataType != syn_type_ufp16), "syn_type_ufp16 unsupported in gaudi2");
    unsigned minCDElemsSameInOut = 0;
    unsigned outDTypeBytes       = dataTypeSizeInBytes(outDataType);
    unsigned inDTypeBytes        = dataTypeSizeInBytes(inDataType);
    switch (inDataType)
    {
        case syn_type_fp8_143:
        case syn_type_fp8_152:
        case syn_type_bf16:
        case syn_type_fp16:
        case syn_type_tf32:
            minCDElemsSameInOut = 256;
            break;
        case syn_type_hb_float:
            minCDElemsSameInOut = 128;
            break;
        case syn_type_float:
            minCDElemsSameInOut = 64;
            break;
        default:
            HB_ASSERT(false, "unexpected input data type: {}", inDataType);
    }
    return (minCDElemsSameInOut * outDTypeBytes) / inDTypeBytes;
}

unsigned Gaudi2HalReader::getNumUploadKernelEbPad() const
{
    return hal::numUploadKernelEbPad;
}

bool Gaudi2HalReader::isNonLinearDmaSupported() const
{
    return hal::isNonLinearDmaSupported;
}

bool Gaudi2HalReader::isSRAMReductionSupported() const
{
    return hal::isSRAMReductionSupported;
}

bool Gaudi2HalReader::isDRAMReductionSupported() const
{
    return hal::isDRAMReductionSupported;
}

unsigned Gaudi2HalReader::getSRAMSizeInBytes() const
{
    unsigned sramSize = hal::sramSize - GCFG_SRAM_SIZE_RESERVED_FOR_HCL.value();
    if (GCFG_SET_SRAM_SIZE.value() != 0)
    {
        sramSize = std::min((unsigned)GCFG_SET_SRAM_SIZE.value(), sramSize);
    }
    return sramSize;
}

uint64_t Gaudi2HalReader::getSRAMBaseAddr() const
{
    return hal::sramBaseAddress + GCFG_SRAM_SIZE_RESERVED_FOR_HCL.value();
}

uint64_t Gaudi2HalReader::getDRAMSizeInBytes() const
{
    return hal::dramSize;
}

unsigned Gaudi2HalReader::getNumTpcEngines() const
{
    return hal::numTpcEngines;
}

uint64_t Gaudi2HalReader::getTpcEnginesMask() const
{
    return hal::tpcEnginesMask;
}

unsigned Gaudi2HalReader::getNumMmeEngines() const
{
    return hal::numMmeEngines;
}

unsigned Gaudi2HalReader::getNumMmeCoresPerEngine() const
{
    return hal::numMmeCoresPerEngine;
}

unsigned Gaudi2HalReader::getNumMmeMaxGemmsPerCore() const
{
    return hal::maxNumMmeGemmsPerCore;
}

unsigned Gaudi2HalReader::getNumDmaEngines() const
{
    return hal::numDmaEngines;
}

unsigned Gaudi2HalReader::getNumInternalDmaEngines() const
{
    HB_ASSERT(GCFG_EDMA_NUM_BINNED.value() < hal::numInternalDmaEngines, "bad EDMA binning");
    return hal::numInternalDmaEngines - GCFG_EDMA_NUM_BINNED.value();
}

unsigned Gaudi2HalReader::getInternalDmaEnginesMask() const
{
    HB_ASSERT(GCFG_EDMA_NUM_BINNED.value() < std::bitset<32>(hal::numInternalDmaEngines).count(), "bad EDMA binning");
    return turnOffMSBs(hal::internalDmaEnginesMask, GCFG_EDMA_NUM_BINNED.value());
}

unsigned Gaudi2HalReader::getDmaBwGBps() const
{
    return hal::dmaBwGBps;
}

double Gaudi2HalReader::getDmaMinimalOverhead() const
{
    // The minimal overhead of dma engine activation - needed for DMA cost model
    // 0.3 usec is Gaudi2 minimal overhead observed
    return 0.3;
}

unsigned Gaudi2HalReader::getNumRotatorEngines() const
{
    if (!GCFG_RUNNING_ON_PLDM.value())
    {
        return hal::numRotatorEngines;
    }

    // when running on PLDM
    return 0;
}

bool Gaudi2HalReader::isRotateAngleSupported(float angle) const
{
    float maxSupportedAngle = hal::rotateMaxSupportedAngle;
    if ((angle >= 0.0) && (angle <= maxSupportedAngle))
    {
        return true;
    }
    if ((angle >= 180.0 - maxSupportedAngle) && (angle <= 180.0 + maxSupportedAngle))
    {
        return true;
    }
    if ((angle >= 360.0 - maxSupportedAngle) && (angle <= 360.0))
    {
        return true;
    }
    if ((angle == 90.0) || (angle == 270.0))
    {
        return true;
    }

    return false;
}

synDeviceType Gaudi2HalReader::getDeviceType() const
{
    return synDeviceGaudi2;
}

unsigned Gaudi2HalReader::getMmeAccumulatorH() const
{
    return hal::mmeAccumulatorH;
}

DmaTransposeEngineParams Gaudi2HalReader::getDmaTransposeEngineParams() const
{
    return DmaTransposeEngineParams {.maxSrc0         = hal::dmaTransposeEngineMaxSrc0InBytes,
                                     .maxDst0         = hal::dmaTransposeEngineMaxDst0InBytes,
                                     .numLinesDivisor = hal::dmaTransposeEngineNumLinesDivisor};
}

bool Gaudi2HalReader::isDmaTransposeSupported(synDataType type) const
{
    switch(type)
    {
        case syn_type_fp8_143:
        case syn_type_fp8_152:
        case syn_type_int8:
        case syn_type_uint8:
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

synDataType Gaudi2HalReader::getDmaTransposeSupportedDataType(TSize dataTypeSize) const
{
    switch (dataTypeSize)
    {
        case 1:
            return syn_type_int8;
        case 2:
            return syn_type_bf16;
        case 4:
            return syn_type_single;
        default:
            return syn_type_na;
    };
}

uint64_t Gaudi2HalReader::getDRAMBaseAddr() const
{
    return hal::dramBaseAddress;
}

unsigned Gaudi2HalReader::getMmeSBCacheSize() const
{
    return hal::mmeSBCacheSize;
}

unsigned Gaudi2HalReader::getPrefetchAlignmentMask() const
{
    return hal::prefetchAlignmentMask;
}

unsigned Gaudi2HalReader::getNumMonitors() const
{
    return hal::numMonitors;
}

unsigned Gaudi2HalReader::getNumSyncObjects() const
{
    return hal::numSyncObjects;
}

unsigned Gaudi2HalReader::getFirstInternalDmaEngineId() const
{
    return hal::firstInternalDmaEngineId;
}

unsigned Gaudi2HalReader::getNumEngineStreams() const
{
    return hal::numEngineStreams;
}

unsigned Gaudi2HalReader::getClockFreqMHz() const
{
    return hal::clockFreqMHz;
}

unsigned Gaudi2HalReader::getHbmBwGBps() const
{
    return hal::hbmBwGBps;
}

unsigned Gaudi2HalReader::getSramBwGBps() const
{
    return hal::sramBwGBps;
}

unsigned Gaudi2HalReader::getFirstSyncObjId() const
{
    return hal::firstSyncObjId;
}

unsigned Gaudi2HalReader::getFirstMonObjId() const
{
    return hal::firstMonObjId;
}

unsigned Gaudi2HalReader::getNumTpcEnginesOnDcore() const
{
    return hal::numTpcEnginesOnDcore;
}
unsigned Gaudi2HalReader::getNumMmeEnginesOnDcore() const
{
    return hal::numMmeEnginesOnDcore;
}
unsigned Gaudi2HalReader::getNumInternalDmaEnginesOnDcore() const
{
    return hal::numInternalDmaEnginesOnDcore;
}
unsigned Gaudi2HalReader::getNumRotatorEnginesOnDcore() const
{
    return hal::numRotatorEnginesOnDcore;
}

unsigned Gaudi2HalReader::getRotateStripeWidth() const
{
    return hal::rotateStripeWidth;
}

unsigned Gaudi2HalReader::getRotateStripeHeightStraightAngle() const
{
    return hal::rotateStripeHeightStraightAngle;
}

unsigned Gaudi2HalReader::getCpDmaAlignment() const
{
    return hal::cpDmaAlignment;
}

unsigned Gaudi2HalReader::getBaseRegistersCacheSize() const
{
    return hal::baseRegistersCacheSize;
}

bool Gaudi2HalReader::isMMETransposeAandTransposeBSupported() const
{
    return true;
}

bool Gaudi2HalReader::isDuplicateMmeOutputSupported() const
{
    return true;
}

const std::vector<HabanaDeviceType>& Gaudi2HalReader::getSupportedDeviceTypes() const
{
    if (!GCFG_RUNNING_ON_PLDM.value())
    {
        static const std::vector<HabanaDeviceType> SUPPORTED = {HabanaDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL,
                                                                HabanaDeviceType::DEVICE_MME,
                                                                HabanaDeviceType::DEVICE_TPC,
                                                                HabanaDeviceType::DEVICE_ROTATOR};
        return SUPPORTED;
    }

    // PLDM image does not include rotator
    static const std::vector<HabanaDeviceType> SUPPORTED = {HabanaDeviceType::DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL,
                                                            HabanaDeviceType::DEVICE_MME,
                                                            HabanaDeviceType::DEVICE_TPC};
    return SUPPORTED;
}

uint32_t Gaudi2HalReader::getDCacheLineSize() const
{
    return getCacheLineSizeInBytes();
}

uint32_t Gaudi2HalReader::getDCacheLineNr() const
{
    return hal::dCacheLineNr;
}

uint32_t Gaudi2HalReader::getDCacheMaxStride() const
{
    return hal::dCacheMaxStride;
}

uint32_t Gaudi2HalReader::getDCacheMinStride() const
{
    return hal::dCacheMinStride;
}

uint64_t Gaudi2HalReader::getDefaultMaxRegVal() const
{
    return hal::maxRegSizeforTpcAndLargeDTMme;
}

uint64_t Gaudi2HalReader::getMaxRegValForExtendedDimInTPC() const
{
    return hal::maxRegSizeforExtendedDimInTpc;
}

uint64_t Gaudi2HalReader::getMaxRegValForDma() const
{
    return hal::maxRegSizeforDmaAndSmallDTMme;
}

uint64_t Gaudi2HalReader::getMaxRegValForMME(unsigned dataTypeSize) const
{
    return dataTypeSize > 2 ? hal::maxRegSizeforTpcAndLargeDTMme : hal::maxRegSizeforDmaAndSmallDTMme;
}

bool Gaudi2HalReader::isAsicSupportIRF44Mode() const
{
    return hal::isIRF44ModeSupported;
}

uint64_t Gaudi2HalReader::getMaxRegValForSigned44BitMode() const
{
    return hal::maxRegSizeForSigned44BitMode;
}


unsigned Gaudi2HalReader::getMcidBaseRegsFirstIndex() const
{
    // MCID not supportted for Gaudi2
    return 0;
}

bool Gaudi2HalReader::isTPCMemcpySupportedDataType(synDataType type) const
{
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_fp16:
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

bool Gaudi2HalReader::isTPCMemsetSupportedDataType(synDataType type) const
{
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_fp16:
        case syn_type_bf16:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_single:
            return true;
        default:
            return false;
    }
}

unsigned Gaudi2HalReader::getMmeMaxInterleavedSpatialPortsNr() const
{
    return hal::mmeMaxInterleavedSpatialPortsNr;
}

bool Gaudi2HalReader::isInvScaleExpBiasHWAligned(float invScale) const
{
    return std::find(std::begin(hwAlignedInvScalesFp8143), std::end(hwAlignedInvScalesFp8143), invScale) != std::end(hwAlignedInvScalesFp8143);
}

bool Gaudi2HalReader::isScaleExpBiasHWAligned(float scale) const
{
    return std::find(std::begin(hwAlignedScalesFp8143), std::end(hwAlignedScalesFp8143), scale) != std::end(hwAlignedScalesFp8143);
}