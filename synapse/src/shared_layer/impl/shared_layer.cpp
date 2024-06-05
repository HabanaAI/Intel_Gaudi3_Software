#include <shared_layer_agent_api.hpp>
#include "op_validator.h"
#include "gc_ops_db.h"
#include "log_manager.h"
#include "data_type_utils.h"
#include "defs.h"
#include "synapse_common_types.h"
#include "utils.h"

#define GCSL_C_API extern "C"

static gc::ops::OpValidationContext sharedLayerParamsToOpValidationContext(const SharedLayer::Params_t* pParams)
{
    gc::ops::OpValidationContext ovc {};
    if (!pParams) return ovc;
    ovc.getInputs().reserve(pParams->inputTensorNr);
    for (unsigned inIdx = 0; inIdx < pParams->inputTensorNr; ++inIdx)
    {
        const auto& tensorGeometry = pParams->inputTensors[inIdx].geometry;
        ovc.getInputs().push_back(
            gc::ops::TensorValidationContext(tensorGeometry.dims,
                                             translateTensorDataType(tensorGeometry.dataType, syn_type_na),
                                             DATA_TENSOR));
    }

    ovc.getOutputs().reserve(pParams->outputTensorNr);
    for (unsigned outIdx = 0; outIdx < pParams->outputTensorNr; ++outIdx)
    {
        const auto& tensorGeometry = pParams->outputTensors[outIdx].geometry;
        ovc.getOutputs().push_back(
            gc::ops::TensorValidationContext(tensorGeometry.dims,
                                             translateTensorDataType(tensorGeometry.dataType, syn_type_na),
                                             DATA_TENSOR));
    }
    return ovc;
}

static SharedLayer::Return_t toSharedLayerReturnType(const gc::ops::ValidationResult& gcSharedLayerRc)
{
    using gc::ops::ValidationResult;
    using SharedLayer::Return_t;

    switch (gcSharedLayerRc)
    {
        case ValidationResult::SUCCESS:
            return Return_t::SHARED_LAYER_SUCCESS;
        case ValidationResult::GUID_NOT_FOUND:
            return Return_t::SHARED_LAYER_GUID_NOT_FOUND;
        case ValidationResult::INCOMPATIBLE_INPUT_COUNT:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_INPUT_COUNT;
        case ValidationResult::INCOMPATIBLE_INPUT_DIMENSION:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_INPUT_DIMENSION;
        case ValidationResult::INCOMPATIBLE_INPUT_SIZE:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_INPUT_SIZE;
        case ValidationResult::INCOMPATIBLE_OUTPUT_COUNT:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_OUTPUT_COUNT;
        case ValidationResult::INCOMPATIBLE_OUTPUT_DIMENSION:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_OUTPUT_DIMENSION;
        case ValidationResult::INCOMPATIBLE_OUTPUT_SIZE:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_OUTPUT_SIZE;
        case ValidationResult::INCOMPATIBLE_DATA_TYPE:
            return Return_t::SHARED_LAYER_INCOMPATIBLE_DATA_TYPE;
        default:
            LOG_ERR(GC_SHARED_LAYER, "{}: Failed translating gc shared layer enum {}", HLLOG_FUNC, gcSharedLayerRc);
        // intentional fall-through
        case ValidationResult::FAILURE:
            return Return_t::SHARED_LAYER_FAILED;
    }
}

// ========== Shared Layer API methods ==========

GCSL_C_API SharedLayer::Return_t HLSL_init()
{
    auto rc = SharedLayer::SHARED_LAYER_SUCCESS;
    try
    {
        gc::ops::GCOpsDB::instance();
    }
    catch (const std::exception& e)
    {
        LOG_ERR(GC_SHARED_LAYER, "{}: Failed with exception: {}", HLLOG_FUNC, e.what());
        rc = SharedLayer::SHARED_LAYER_FAILED;
    }
    return rc;
}

GCSL_C_API SharedLayer::Return_t HLSL_validate_guid(_IN_ const SharedLayer::Params_t* const params)
{
    auto sharedLayerRc = SharedLayer::SHARED_LAYER_SUCCESS;
    try
    {
        HB_ASSERT_PTR(params);

        const auto  deviceType = deviceIDToDeviceType(params->deviceId);
        std::string guid(params->guid.name);
        auto        testedOpCtx = sharedLayerParamsToOpValidationContext(params);
        const auto  rc          = gc::ops::OpValidator::validateOp(guid, testedOpCtx, deviceType);
        sharedLayerRc           = toSharedLayerReturnType(rc);
    }
    catch (const std::exception& e)
    {
        LOG_ERR(GC_SHARED_LAYER, "{}: Failed with exception: {}", HLLOG_FUNC, e.what());
        sharedLayerRc = SharedLayer::SHARED_LAYER_FAILED;
    }
    return sharedLayerRc;
}

GCSL_C_API SharedLayer::Return_t
HLSL_GetKernelNames(_INOUT_ char** guidNameArray, _INOUT_ int* guidCount, _IN_ const SharedLayer::DeviceId device)
{
    try
    {
        HB_ASSERT_PTR(guidCount);

        const auto  deviceType = deviceIDToDeviceType(device);
        const auto& gcOpsDb(gc::ops::GCOpsDB::instance());
        const auto  supportedGuids  = gcOpsDb.getSupportedGuids(deviceType);
        const int   nSupportedGuids = supportedGuids != nullptr ? supportedGuids->size() : 0;
        if (!guidNameArray)
        {
            // API Handshake #0: Supply user with number of guids supported by device
            *guidCount = nSupportedGuids;
        }
        else
        {
            // API Handshake #1: Fill output param guidNameArray with up to guidCount guids supported by device

            // No supported guids for input device, nothing to copy
            if (nSupportedGuids <= 0)
            {
                *guidCount = 0;
                return SharedLayer::SHARED_LAYER_SUCCESS;
            }
            HB_ASSERT_PTR(supportedGuids);

            const unsigned limit = std::min(*guidCount, nSupportedGuids);
            std::size_t    idx   = 0;
            for (const auto& guid : *supportedGuids)
            {
                if (idx >= limit) break;
                const auto guidStrlen =
                    std::min(static_cast<std::size_t>(tpc_lib_api::MAX_NODE_NAME - 1), guid.length());
                guid.copy(guidNameArray[idx], guidStrlen);
                guidNameArray[idx][guidStrlen + 1] = '\0';
                ++idx;
            }
            *guidCount = idx;
        }
    }
    catch (const std::exception& e)
    {
        LOG_ERR(GC_SHARED_LAYER, "{}: Failed with exception: {}", HLLOG_FUNC, e.what());
        return SharedLayer::SHARED_LAYER_FAILED;
    }
    return SharedLayer::SHARED_LAYER_SUCCESS;
}