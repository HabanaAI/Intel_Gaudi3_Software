#include "hal_reader/gaudi3/hal_reader.h"

#include "habana_global_conf.h"
#include "hal_reader/gaudi3/hal.h"
#include "synapse_common_types.h"
#include "utils.h"

using namespace gaudi3;

std::set<float> hwAlignedInvScalesFp8143 {{
        pow(2, -7), pow(2, -6), pow(2, -5), pow(2, -4), pow(2, -3), pow(2, -2), pow(2, -1), pow(2, 0),
        pow(2, 1), pow(2, 2), pow(2, 3), pow(2, 4), pow(2, 5), pow(2, 6), pow(2, 7), pow(2, 8), pow(2, 9), pow(2, 10),
        pow(2, 11), pow(2, 12), pow(2, 13), pow(2, 14), pow(2, 15), pow(2, 16), pow(2, 17), pow(2, 18), pow(2, 19), pow(2, 20),
        pow(2, 21), pow(2, 22), pow(2, 23), pow(2, 24), pow(2, 25), pow(2, 26), pow(2, 27), pow(2, 28), pow(2, 29), pow(2, 30),
        pow(2, 31), pow(2, 32), pow(2, 33), pow(2, 34), pow(2, 35), pow(2, 36), pow(2, 37), pow(2, 38), pow(2, 39), pow(2, 40),
        pow(2, 41), pow(2, 42), pow(2, 43), pow(2, 44), pow(2, 45), pow(2, 46), pow(2, 47), pow(2, 48), pow(2, 49), pow(2, 50),
        pow(2, 51), pow(2, 52), pow(2, 53), pow(2, 54), pow(2, 55), pow(2, 56)}};

std::set<float> hwAlignedScalesFp8143 = {{
        pow(2, 7), pow(2, 6), pow(2, 5), pow(2, 4), pow(2, 3), pow(2, 2), pow(2, 1), pow(2, 0),
        pow(2, -1), pow(2, -2), pow(2, -3), pow(2, -4), pow(2, -5), pow(2, -6), pow(2, -7), pow(2, -8), pow(2, -9), pow(2, -10),
        pow(2, -11), pow(2, -12), pow(2, -13), pow(2, -14), pow(2, -15), pow(2, -16), pow(2, -17), pow(2, -18), pow(2, -19), pow(2, -20),
        pow(2, -21), pow(2, -22), pow(2, -23), pow(2, -24), pow(2, -25), pow(2, -26), pow(2, -27), pow(2, -28), pow(2, -29), pow(2, -30),
        pow(2, -31), pow(2, -32), pow(2, -33), pow(2, -34), pow(2, -35), pow(2, -36), pow(2, -37), pow(2, -38), pow(2, -39), pow(2, -40),
        pow(2, -41), pow(2, -42), pow(2, -43), pow(2, -44), pow(2, -45), pow(2, -46), pow(2, -47), pow(2, -48), pow(2, -49), pow(2, -50),
        pow(2, -51), pow(2, -52), pow(2, -53), pow(2, -54), pow(2, -55), pow(2, -56)}};

HalReaderPtr instantiateGaudi3HalReader()
{
    return Gaudi3HalReader::instance();
}

const std::shared_ptr<Gaudi3HalReader>& Gaudi3HalReader::instance()
{
    static const std::shared_ptr<Gaudi3HalReader> singleDIEInstance(
        new Gaudi3HalReader(EChipFlavor::SINGLE_DIE));
    static const std::shared_ptr<Gaudi3HalReader> doubleDIEInstance(
        new Gaudi3HalReader(EChipFlavor::DOUBLE_DIE));
    static const std::shared_ptr<Gaudi3HalReader> allCacheDIEInstance(
        new Gaudi3HalReader(EChipFlavor::PLDM_ALL_CACHE));
    return GCFG_GAUDI3_SINGLE_DIE_CHIP.value()
               ? (GCFG_GAUDI3_PLDM_FULL_CACHE_CHIP.value() ? allCacheDIEInstance : singleDIEInstance)
               : doubleDIEInstance;
}

static const HalChipFlavorSpecificInfo halfChipInfo = halHalfChipSpecificInfo;
static const HalChipFlavorSpecificInfo fullChipInfo = halFullChipSpecificInfo;
static const HalChipFlavorSpecificInfo allCacheChipInfo = halAllCacheSpecificInfo;

Gaudi3HalReader::Gaudi3HalReader(EChipFlavor chipFlavor)
: m_halChipFlavorSpecificInfo(chipFlavor == EChipFlavor::SINGLE_DIE     ? halfChipInfo
                              : (chipFlavor == EChipFlavor::DOUBLE_DIE) ? fullChipInfo
                                                                        : allCacheChipInfo)
{
}

#undef GET_HAL_FIELD

unsigned Gaudi3HalReader::getSupportedTypes() const
{
    return hal::supportedTypes;
}

unsigned Gaudi3HalReader::getSupportedMmeTypes() const
{
    return hal::supportedMmeTypes;
}

unsigned Gaudi3HalReader::getMmeVectorSize() const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi3HalReader::getTpcVectorSize() const
{
    return hal::tpcVectorSize;
}

unsigned Gaudi3HalReader::getCacheLineSizeInBytes() const
{
    return hal::clSize;
}

unsigned Gaudi3HalReader::getAddressAlignmentSizeInBytes() const
{
    return hal::addressAlignmentSizeInBytes;
}

bool Gaudi3HalReader::isMmeCinSupported() const
{
    return hal::mmeCinSupported;
}

unsigned Gaudi3HalReader::getTPCICacheSize() const
{
    return hal::tpcICacheSize;
}

unsigned Gaudi3HalReader::getMmeMinimalWidthInElems(synDataType type) const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi3HalReader::getMmeMaximalEUHeightInElems(synDataType type) const
{
    return hal::mmeMaximalEUHeight;
}

unsigned Gaudi3HalReader::getMmeSymmetricWidthInElems(synDataType type) const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi3HalReader::getMmeSymmetricHeightInElems(synDataType type) const
{
    return hal::mmeVectorSize;
}

unsigned Gaudi3HalReader::getMmeMinCDInElements(synDataType inDataType, synDataType outDataType) const
{
    unsigned minCDElemsBase = 0;
    switch (inDataType)
    {
        case syn_type_fp8_143:
        case syn_type_fp8_152:
        case syn_type_bf16:
            minCDElemsBase = 256;
            break;
        case syn_type_fp16:
        case syn_type_tf32:
        case syn_type_ufp16:
            minCDElemsBase = 64;
            break;
        case syn_type_hb_float:
        case syn_type_float:
            minCDElemsBase = 32;
            break;
        default:
            HB_ASSERT(false, "unexpected input data type: {}", inDataType);
    }
    return minCDElemsBase * dataTypeSizeInBytes(outDataType);
}

synDataType Gaudi3HalReader::getMmeHighPrecisionTypeForPartials() const
{
    return syn_type_hb_float;
}

bool Gaudi3HalReader::isNonLinearDmaSupported() const
{
    return hal::isNonLinearDmaSupported;
}

bool Gaudi3HalReader::isSRAMReductionSupported() const
{
    return hal::isSRAMReductionSupported;
}

bool Gaudi3HalReader::isDRAMReductionSupported() const
{
    return hal::isDRAMReductionSupported;
}

unsigned Gaudi3HalReader::getSRAMSizeInBytes() const
{
    unsigned sramSize = m_halChipFlavorSpecificInfo.sramSize - GCFG_SRAM_SIZE_RESERVED_FOR_HCL.value();
    if (GCFG_SET_SRAM_SIZE.value() != 0)
    {
        sramSize = std::min((unsigned)GCFG_SET_SRAM_SIZE.value(), sramSize);
    }
    return sramSize;
}

uint64_t Gaudi3HalReader::getSRAMBaseAddr() const
{
    return hal::sramBaseAddress + GCFG_SRAM_SIZE_RESERVED_FOR_HCL.value();
}

uint64_t Gaudi3HalReader::getDRAMSizeInBytes() const
{
    return hal::dramSize;
}

unsigned Gaudi3HalReader::getNumTpcEngines() const
{
    return m_halChipFlavorSpecificInfo.numTpcEngines;
}

uint64_t Gaudi3HalReader::getTpcEnginesMask() const
{
    return m_halChipFlavorSpecificInfo.tpcEnginesMask;
}

unsigned Gaudi3HalReader::getNumMmeEngines() const
{
    return m_halChipFlavorSpecificInfo.numMmeEngines;
}

unsigned Gaudi3HalReader::getNumMmeCoresPerEngine() const
{
    return hal::numMmeCoresPerEngine;
}

unsigned Gaudi3HalReader::getNumMmeMaxGemmsPerCore() const
{
    return hal::maxNumMmeGemmsPerCore ;
}

unsigned Gaudi3HalReader::getNumDmaEngines() const
{
    return m_halChipFlavorSpecificInfo.numDmaEngines;
}

unsigned Gaudi3HalReader::getNumInternalDmaEngines() const
{
    return hal::numInternalDmaEngines;
}

unsigned Gaudi3HalReader::getDmaBwGBps() const
{
    HB_ASSERT(false, "Failure -  In Gaudi3 no EDMA allocated for GC");
    return 0;
}

unsigned Gaudi3HalReader::getInternalDmaEnginesMask() const
{
    return hal::internalDmaEnginesMask;
}

unsigned Gaudi3HalReader::getNumRotatorEngines() const
{
    if (!GCFG_RUNNING_ON_PLDM.value())
    {
        return m_halChipFlavorSpecificInfo.numRotatorEngines;
    }

    // when running on PLDM
    return 0;
}

bool Gaudi3HalReader::isRotateAngleSupported(float angle) const
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

synDeviceType Gaudi3HalReader::getDeviceType() const
{
    return synDeviceGaudi3;
}

unsigned Gaudi3HalReader::getMmeAccumulatorH() const
{
    HB_ASSERT(false, "Failure -  this is MME stack information");
    return 0;
}

DmaTransposeEngineParams Gaudi3HalReader::getDmaTransposeEngineParams() const
{
    HB_ASSERT(false, "Failure - In gaudi3 transpose is done on MME");
    return DmaTransposeEngineParams();
}

bool Gaudi3HalReader::isDmaTransposeSupported(synDataType type) const
{
    return false;
}

bool Gaudi3HalReader::isMmeTransposeSupported() const
{
    return true;
}

uint64_t Gaudi3HalReader::getDRAMBaseAddr() const
{
    return hal::dramBaseAddress;
}

unsigned Gaudi3HalReader::getMmeSBCacheSize() const
{
    HB_ASSERT(false, "Failure - this is MME stack information");
    return 0;
}

unsigned Gaudi3HalReader::getPrefetchAlignmentMask() const
{
    return hal::prefetchAlignmentMask;
}

unsigned Gaudi3HalReader::getNumMonitors() const
{
    return hal::numMonitors;
}

unsigned Gaudi3HalReader::getNumSyncObjects() const
{
    return hal::numSyncObjects;
}

unsigned Gaudi3HalReader::getFirstInternalDmaEngineId() const
{
    return hal::firstInternalDmaEngineId;
}

unsigned Gaudi3HalReader::getNumEngineStreams() const
{
    return hal::numEngineStreams;
}

unsigned Gaudi3HalReader::getClockFreqMHz() const
{
    return hal::clockFreqMHz;
}

unsigned Gaudi3HalReader::getHbmBwGBps() const
{
    return m_halChipFlavorSpecificInfo.hbmBwGBps;
}

unsigned Gaudi3HalReader::getSramBwGBps() const
{
    HB_ASSERT(false, "No SRAM in Gaudi3");
    return 0;
}

unsigned Gaudi3HalReader::getFirstSyncObjId() const
{
    HB_ASSERT(false, "ARC Mode 3 - no sync objects for GC");
    return hal::firstSyncObjId;
}

unsigned Gaudi3HalReader::getFirstMonObjId() const
{
    HB_ASSERT(false, "ARC Mode 3 - no monitor objects for GC");
    return hal::firstMonObjId;
}

unsigned Gaudi3HalReader::getNumDcores() const
{
    return m_halChipFlavorSpecificInfo.numDcores;
}

unsigned Gaudi3HalReader::getNumTpcEnginesOnDcore() const
{
    return hal::numTpcEnginesOnDcore;
}

unsigned Gaudi3HalReader::getNumMmeEnginesOnDcore() const
{
    return hal::numMmeEnginesOnDcore;
}

unsigned Gaudi3HalReader::getNumInternalDmaEnginesOnDcore() const
{
    return hal::numInternalDmaEnginesOnDcore;
}
unsigned Gaudi3HalReader::getNumRotatorEnginesOnDcore() const
{
    return hal::numRotatorEnginesOnDcore;
}

unsigned Gaudi3HalReader::getRotateStripeWidth() const
{
    return hal::rotateStripeWidth;
}

unsigned Gaudi3HalReader::getRotateStripeHeightStraightAngle() const
{
    return hal::rotateStripeHeightStraightAngle;
}

unsigned Gaudi3HalReader::getCpDmaAlignment() const
{
    return hal::cpDmaAlignment;
}

unsigned Gaudi3HalReader::getBaseRegistersCacheSize() const
{
    return hal::baseRegistersCacheSize;
}

bool Gaudi3HalReader::isMMETransposeAandTransposeBSupported() const
{
    return true;
}

bool Gaudi3HalReader::isDuplicateMmeOutputSupported() const
{
    return true;
}

unsigned Gaudi3HalReader::getCacheDirectiveBits(CacheDirective cacheDirective) const
{
    switch (cacheDirective)
    {
        case SkipCache:
            return hal::skipCache;
        case NoAllocate:
            return hal::noAllocate;
        case HomeAllocate:
            return hal::homeAllocate;
        case DcoreAllocate:
            return hal::dcoreAllocate;
        case SharedAllocate:
            return hal::sharedAllocate;
        default:
            return 0;
    }
}

uint64_t Gaudi3HalReader::getDefaultMaxRegVal() const
{
    return hal::maxRegSizeforTpcAndLargeDTMme;
}

uint64_t Gaudi3HalReader::getMaxRegValForExtendedDimInTPC() const
{
    return hal::maxRegSizeforExtendedDimInTpc;
}

uint64_t Gaudi3HalReader::getMaxRegValForDma() const
{
    return hal::maxRegSizeforDmaAndSmallDTMme;
}

uint64_t Gaudi3HalReader::getMaxRegValForMME(unsigned dataTypeSize) const
{
    return dataTypeSize > 2 ? hal::maxRegSizeforTpcAndLargeDTMme : hal::maxRegSizeforDmaAndSmallDTMme;
}

bool Gaudi3HalReader::isAsicSupportIRF44Mode() const
{
    return hal::isIRF44ModeSupported;
}

uint64_t Gaudi3HalReader::getMaxRegValForSigned44BitMode() const
{
    return hal::maxRegSizeForSigned44BitMode;
}

const std::vector<HabanaDeviceType>& Gaudi3HalReader::getSupportedDeviceTypes() const
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

HabanaDeviceType Gaudi3HalReader::getTransposeEngine() const
{
    return HabanaDeviceType::DEVICE_MME;
}

HabanaDeviceType Gaudi3HalReader::getBroadcastEngine() const
{
    return HabanaDeviceType::DEVICE_TPC;
}

unsigned Gaudi3HalReader::getNumFastConfigMcidSRFs() const
{
    unsigned numSrfs = hal::numFastConfigMcidSRFs;

    if (GCFG_TPC_MCID_NUM_SRF.value() != 0)
    {
        numSrfs = GCFG_TPC_MCID_NUM_SRF.value();
    }
    return numSrfs;
}

unsigned Gaudi3HalReader::getNumSRFs() const
{
    return hal::numSRFs;
}

unsigned Gaudi3HalReader::getTotalNumBaseRegs() const
{
    return hal::totalNumBaseRegs;
}

unsigned Gaudi3HalReader::getNumBaseRegsForAddress() const
{
    return hal::numBaseRegsForAddress;
}

unsigned Gaudi3HalReader::getNumBaseRegsForMcid() const
{
    return hal::numBaseRegsForMcid;
}

unsigned Gaudi3HalReader::getMcidBaseRegsFirstIndex() const
{
    return hal::totalNumBaseRegs - hal::numBaseRegsForMcid;
}

unsigned Gaudi3HalReader::getNumUploadKernelEbPad() const
{
    return hal::numUploadKernelEbPad;
}

unsigned Gaudi3HalReader::getTPCMinSRF() const
{
    return hal::minTpcSRF;
}

bool Gaudi3HalReader::isTPCMemcpySupportedDataType(synDataType type) const
{
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
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

bool Gaudi3HalReader::isTPCMemsetSupportedDataType(synDataType type) const
{
    switch (type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
        case syn_type_fp8_143:
        case syn_type_fp8_152:
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

unsigned Gaudi3HalReader::getMmeMaxInterleavedSpatialPortsNr() const
{
    return hal::mmeMaxInterleavedSpatialPortsNr;
}

bool Gaudi3HalReader::isInvScaleExpBiasHWAligned(float invScale) const
{
    // Should be a power of 2 between 2^-7 and 2^56
    return std::find(std::begin(hwAlignedInvScalesFp8143), std::end(hwAlignedInvScalesFp8143), invScale) != std::end(hwAlignedInvScalesFp8143);
}

bool Gaudi3HalReader::isScaleExpBiasHWAligned(float scale) const
{
    // Should be a power of 2 between 2^-56 and 2^7
    return std::find(std::begin(hwAlignedScalesFp8143), std::end(hwAlignedScalesFp8143), scale) != std::end(hwAlignedScalesFp8143);
}