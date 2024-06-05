#ifndef SHARED_LAYER_INTERFACE_H
#define SHARED_LAYER_INTERFACE_H

#include "tpc_kernel_lib_interface.h"

namespace SharedLayer
{
    using tpc_lib_api::UserParams;
    using tpc_lib_api::Tensor;
    using tpc_lib_api::TensorDataType;
    using tpc_lib_api::DeviceId;
    using tpc_lib_api::GuidInfo;

    using tpc_lib_api::MAX_NODE_NAME;
    using tpc_lib_api::MAX_TENSOR_DIM;

    static constexpr unsigned MAX_TENSOR_NR = 16;

    typedef enum
    {
        SHARED_LAYER_SUCCESS                         = 0,
        SHARED_LAYER_GUID_NOT_FOUND                  = 1,
        SHARED_LAYER_INCOMPATIBLE_INPUT_COUNT        = 2,
        SHARED_LAYER_INCOMPATIBLE_INPUT_DIMENSION    = 3,
        SHARED_LAYER_INCOMPATIBLE_INPUT_SIZE         = 4,
        SHARED_LAYER_INCOMPATIBLE_OUTPUT_COUNT       = 5,
        SHARED_LAYER_INCOMPATIBLE_OUTPUT_DIMENSION   = 6,
        SHARED_LAYER_INCOMPATIBLE_OUTPUT_SIZE        = 7,
        SHARED_LAYER_INCOMPATIBLE_DATA_TYPE          = 8,
        SHARED_LAYER_UNSUPPORTED_LAYER_CONFIGURATION = 9,
        SHARED_LAYER_UNSUPPORTED_QUANT_PARAMS        = 10,
        SHARED_LAYER_UNSUPPORTED_BROADCAST_MODE      = 11,
        SHARED_LAYER_KERNEL_INVALID_SCALAR_ARGUMENT  = 12,
        SHARED_LAYER_MISSING_PRIVATE_STRUCTURE       = 13,
        SHARED_LAYER_FAILED                          = -1,
    } Return_t;

    typedef struct{
        int                           apiVersion;
        DeviceId                      deviceId;
        GuidInfo                      guid;
        UserParams                    nodeParams;
        Tensor*                       inputTensors;
        unsigned                      inputTensorNr;
        Tensor*                       outputTensors;
        unsigned                      outputTensorNr;
    }Params_t;

}; // namespace SharedLayer

#endif // SHARED_LAYER_INTERFACE_H