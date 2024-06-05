#pragma once

#include <graph_compiler/habana_nodes/slice_node.h>
#include "perf_lib_layer_params.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "types.h"
// TODO: [SW-166081] Remove when fuser is moved to protocolIR
#include "gc_interface_private.hpp"

typedef synFlattenParams                 SifFlattenMetadata;
typedef SliceNode::SliceNodeStaticParams SifSliceMetadata;
typedef synSliceAxisParamsV2             SifSliceAxisMetadata;
typedef synExpandDimsParams              SifExpandDimsMetadata;
typedef synExpandDimsParams              SifExpandH2DMetadata;
typedef unsigned                         SifReinterpretH2DMetadata;
typedef ns_TileKernel::ParamsV2          SifTileShapeMetadata;

struct SifSplitHeader
{
    unsigned axis;
    unsigned splitDimSizesNr;
};

struct SifSplitMetadata
{
    SifSplitHeader header;
    TSize          splitDimSizes[1];
};

struct SifTensorViewHeader
{
    bool accessInput;
    unsigned viewsNr;
};

struct SifTensorViewData
{
    unsigned dims;
    TOffset  offsets[tpc_lib_api::MAX_TENSOR_DIM];
    TSize    sizes[tpc_lib_api::MAX_TENSOR_DIM];
};

struct SifTensorViewMetadata
{
    SifTensorViewHeader header;
    SifTensorViewData data[1];
};

struct SifTransposeMetadata
{
    unsigned permutation[HABANA_DIM_MAX];
};

struct SifConvolutionMetadata
{
    // conv params after modifications done on the node except for after-padding, which is kept unmodified
    // to support spatial slicing.
    synConvolution3DParamsV2 params;
    TSize maxOutputSizes[tpc_lib_api::MAX_TENSOR_DIM];

    // Mark if the max-dims are already known: In case of max-dim inference
    // which (if) happens at the very beginning they may not be known, and will
    // be deduced by the sif. Otherwise sif is used for min shape inference.
    bool maxOutputSizesKnown;
};

struct SifGemmMetadata
{
    synGEMMParams params;
};

struct SifDynamicSplitMetadata
{
    unsigned axis;
};

struct SifMergeShapesMetadata
{
    struct DimLocation
    {
        int inputIdx;
        int dimIdx;
    };
    DimLocation dimMap[tpc_lib_api::MAX_TENSOR_DIM + 1];  // +1 to support internal expand dims usage
    unsigned    outputDim;
    unsigned    fillValue;
};

struct SifSqueezeMetadata
{
    uint8_t squeezeDim[tpc_lib_api::MAX_TENSOR_DIM + 1];
};

struct SifEinsumMetadata
{
    unsigned output_dims;
    unsigned input_dims_to_labels[2][tpc_lib_api::MAX_TENSOR_DIM];
    unsigned output_dims_to_labels[tpc_lib_api::MAX_TENSOR_DIM];
    unsigned labels_to_types[tpc_lib_api::MAX_TENSOR_DIM * 2];
};

typedef gcapi::FusedKernelParams SifSplitFusedKernelMetadata;

struct SifDynamicReshapeMetadata
{
    unsigned input_dims;
    char     input_dims_to_labels[tpc_lib_api::MAX_TENSOR_DIM];
    char     output_eq[MAX_USER_PARAMS_SIZE];
};

struct SifEinsumExpandShapeMetadata
{
    unsigned numOfFreeDimsInFirstInput;
    unsigned numOfFreeDimsInSecondInput;
    unsigned freeDimsInFirstInput[tpc_lib_api::MAX_TENSOR_DIM];
    unsigned freeDimsInSecondInput[tpc_lib_api::MAX_TENSOR_DIM];
};

struct SifConcatenateMetadata
{
    unsigned axis;
    bool     withShapeTensor;
};

struct SifReinterpretCastMetadata
{
    unsigned inputElementSizeInBytes;
    unsigned outputElementSizeInBytes;
};

struct SifInferMaxShapeMetadata
{
    TSize             inputMaxSizes[HABANA_DIM_MAX];
    synInferMaxParams params;
};
