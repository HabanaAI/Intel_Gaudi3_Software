#include "device_utils.hpp"

#include "defenders.h"
#include "device_interface.hpp"
#include "graph_compiler/graph_traits.h"
#include "habana_global_conf.h"
#include "hal_reader/hal_reader.h"
#include "log_manager.h"
#include "runtime/common/osal/osal.hpp"
#include "utils.h"

synStatus extractDeviceAttributes(const synDeviceAttribute* deviceAttr,
                                  const unsigned            querySize,
                                  uint64_t*                 retVal,
                                  const synDeviceInfo       deviceInfo,
                                  uint32_t*                 pCurrentClockRate,
                                  DeviceInterface*          pDevice)
{
    VERIFY_IS_NULL_POINTER(SYN_DEVICE, retVal, "Device-attribute array");
    VERIFY_IS_NULL_POINTER(SYN_DEVICE, deviceAttr, "Device-attribute identifier");

    synStatus status       = synSuccess;
    uint64_t* currentValue = retVal;
    for (unsigned queryIndex = 0; queryIndex < querySize; queryIndex++, currentValue++)
    {
        switch (deviceAttr[queryIndex])
        {
            case DEVICE_ATTRIBUTE_SRAM_BASE_ADDRESS:
            {
                *currentValue = deviceInfo.sramBaseAddress;
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_BASE_ADDRESS:
            {
                *currentValue = deviceInfo.dramBaseAddress;
                break;
            }
            case DEVICE_ATTRIBUTE_SRAM_SIZE:
            {
                *currentValue = deviceInfo.sramSize;
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_SIZE:
            {
                *currentValue = deviceInfo.dramSize;
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_FREE_SIZE:
            {
                if (pDevice == nullptr)
                {
                    LOG_ERR(SYN_DEVICE,
                            "{}: Dram-free-size attribute is supported only for an acquired device",
                            HLLOG_FUNC);

                    *currentValue = 0;
                    continue;
                }

                uint64_t total = 0;
                uint64_t free  = 0;
                status = pDevice->getDramMemInfo(free, total);
                if (status != synSuccess)
                {
                    LOG_ERR(SYN_DEVICE, "{}: Cannot retrieve dram-free-size attribute", HLLOG_FUNC);
                    status = synFail;
                }
                else
                {
                    *currentValue = free;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_TPC_ENABLED_MASK:
            {
                *currentValue = deviceInfo.tpcEnabledMask;
                break;
            }
            case DEVICE_ATTRIBUTE_DRAM_ENABLED:
            {
                *currentValue = deviceInfo.dramEnabled;
                break;
            }
            case DEVICE_ATTRIBUTE_DEVICE_TYPE:
            {
                *currentValue = deviceInfo.deviceType;
                break;
            }
            case DEVICE_ATTRIBUTE_CLK_RATE:
            {
                if (pDevice == nullptr)
                {
                    VERIFY_IS_NULL_POINTER(SYN_DEVICE, pCurrentClockRate, "Clock-rate");

                    *currentValue = *pCurrentClockRate;
                    continue;
                }

                DeviceClockRateInfo deviceClockRateInfo;
                status = OSAL::getInstance().getDeviceClockRateInfo(deviceClockRateInfo);
                if (status != synSuccess)
                {
                    LOG_ERR(SYN_DEVICE, "{}: Can not retrieve clock-rate-info attribute", HLLOG_FUNC);
                    if (status == synDeviceReset)
                    {
                        pDevice->notifyHlthunkFailure(DfaErrorCode::getDeviceClockRateFailed);
                    }
                    status = synFail;
                }
                else
                {
                    *currentValue = deviceClockRateInfo.currentClockRate;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_MAX_RMW_SIZE:
            {
                *currentValue = GCFG_RMW_SECTION_MAX_SIZE_BYTES.value();
                if (*currentValue == 0)
                {
                    LOG_ERR(SYN_DEVICE, "{}: Unsupported attribute DEVICE_ATTRIBUTE_MAX_RMW_SIZE", HLLOG_FUNC);
                    status = synUnsupported;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_STREAMS_TOTAL_MEM_SIZE:
            {
                if (pDevice == nullptr)
                {
                    LOG_ERR(SYN_DEVICE,
                            "{}: Total-stream-mapped-memory attribute is supported only for an acquired device",
                            HLLOG_FUNC);

                    *currentValue = 0;
                    continue;
                }

                status = pDevice->getDeviceTotalStreamMappedMemory(*currentValue);
                if (status != synSuccess)
                {
                    LOG_ERR(SYN_DEVICE, "{}: Can not retrieve total-stream-mapped-memory attribute", HLLOG_FUNC);
                    status = synFail;
                }
                break;
            }
            case DEVICE_ATTRIBUTE_ADDRESS_ALIGNMENT_SIZE:
            {
                GraphTraits                graphTraits(deviceInfo.deviceType);
                std::shared_ptr<HalReader> halReader = graphTraits.getHalReader();
                *currentValue                        = halReader->getAddressAlignmentSizeInBytes();
                break;
            }
            case DEVICE_ATTRIBUTE_MAX_DIMS:
            {
                *currentValue = SYN_GAUDI_MAX_TENSOR_DIM;
                break;
            }
            default:
            {
                LOG_ERR(SYN_DEVICE,
                        "{}: Unsupported attribute {} in query index {}",
                        HLLOG_FUNC,
                        deviceAttr[queryIndex],
                        queryIndex);
                status = synUnsupported;
            }
        }
    }

    return status;
}