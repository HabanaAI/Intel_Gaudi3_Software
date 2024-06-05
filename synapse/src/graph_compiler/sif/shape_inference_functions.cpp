#include "shape_inference_functions.h"
#include "shape_inference_metadata.h"
#include "synapse_common_types.hpp"
#include "types.h"
#include "utils.h"
#include "tensor.h"
#include <bitset>
#include "conv_base_node.h"
#include "vtune_stat.h"
#include "habana_nodes.h"
#include "einsum_node.h"
#include "h2d_tensors.h"

// TODO: [SW-23614] CD 0.12.0 - Compile U20 - Fail build_synapse
// compilation fails on with gcc 9.3, remove when fixed
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

#define NULL_CHECK(pointer, message)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (pointer == nullptr)                                                                                        \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE, "null pointer passed to SIF: {}, error: {}", HLLOG_FUNC, message);                      \
            return tpc_lib_api::GLUE_SIF_NULL_PTR;                                                                     \
        }                                                                                                              \
    } while (0)

#define VALIDATE_TENSORS(tensors, size, message)                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        for (size_t i = 0; i < size; ++i)                                                                              \
        {                                                                                                              \
            NULL_CHECK(tensors[i], message);                                                                           \
        }                                                                                                              \
    } while (0)

#define VALIDATE_PARAMS(in, out)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        NULL_CHECK(in, "null input SIF params pointer");                                                               \
        NULL_CHECK(out, "null output SIF params pointer");                                                             \
        VALIDATE_TENSORS(in->inputTensors, in->inputTensorsNr, "null input TensorsShapeInfo");                         \
        VALIDATE_TENSORS(out->outputTensors, in->outputTensorsNr, "null output TensorsShapeInfo");                     \
    } while (0)

#define VALIDATE_METADATA(metadata, size)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (metadata == nullptr)                                                                                       \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE, "null metadata in SIF {}", HLLOG_FUNC);                                                 \
            return tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE;                                                        \
        }                                                                                                              \
        if (size <= 0 || (size != sizeof(*metadata)))                                                                  \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE,                                                                                         \
                    "incorrect metadata size in SIF {}: expected {}, actual {}",                                       \
                    HLLOG_FUNC,                                                                                        \
                    sizeof(*metadata),                                                                                 \
                    size);                                                                                             \
            return tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE;                                                        \
        }                                                                                                              \
    } while (0)

#define VALIDATE_INPUT_TENSORS_COUNT(cond)                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE, "invalid number of input tensors in SIF {}: expecting {}", HLLOG_FUNC, #cond);          \
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;                                                         \
        }                                                                                                              \
    } while (0)

#define VALIDATE_OUTPUT_TENSORS_COUNT(cond)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE, "invalid number of output tensors in SIF {}: expecting {}", HLLOG_FUNC, #cond);         \
            return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;                                                        \
        }                                                                                                              \
    } while (0)

#define VALIDATE_DIM_IN_RANGE(dim, maxRange)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (dim >= maxRange)                                                                                           \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE, "invalid dim in SIF {}: dim is {}, but tensor rank is {}", HLLOG_FUNC, dim, maxRange);  \
            return tpc_lib_api::GLUE_FAILED;                                                                           \
        }                                                                                                              \
    } while (0)

#define VALIDATE_DIMS_EQUAL(dims, expected)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if (dims != expected)                                                                                          \
        {                                                                                                              \
            LOG_ERR(DYN_SHAPE,                                                                                         \
                    "Unexpected tensor rank in SIF {}: expected {}, actual {}",                                        \
                    HLLOG_FUNC,                                                                                        \
                    expected,                                                                                          \
                    dims);                                                                                             \
            return tpc_lib_api::GLUE_FAILED;                                                                                  \
        }                                                                                                              \
    } while (0)

#define RETURN_ON_FAILURE(op)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        SifReturn ret = op;                                                                                            \
        if (ret != tpc_lib_api::GLUE_SUCCESS)                                                                                 \
        {                                                                                                              \
            return ret;                                                                                                \
        }                                                                                                              \
    } while (0)

#define VALIDATE_SAME_NUMBER_OF_ELEMENTS(one, two) \
    do { \
        auto numElementsInOne = multiplyElements(one->geometry.maxSizes, one->geometry.maxSizes + one->geometry.dims); \
        auto numElementsInTwo = multiplyElements(two->geometry.maxSizes, two->geometry.maxSizes + two->geometry.dims); \
        if (numElementsInOne != numElementsInTwo) \
        { \
            LOG_ERR(DYN_SHAPE, "{}: Incompatible shapes: {} size [{}] ({} elements), {} size [{}] ({} elements)", \
                HLLOG_FUNC, \
                #one, \
                fmt::join(one->geometry.maxSizes, one->geometry.maxSizes + one->geometry.dims, ","), \
                numElementsInOne, \
                #two, \
                fmt::join(two->geometry.maxSizes, two->geometry.maxSizes + two->geometry.dims, ","), \
                numElementsInTwo); \
            return tpc_lib_api::GLUE_FAILED; \
        } \
    } while (0)


static void invalidateTensor(const unsigned index, unsigned* invalidOutput)
{
    constexpr unsigned             containerBitsSize = sizeof(*invalidOutput) * CHAR_BIT;
    unsigned                       containerIndex    = index / containerBitsSize;
    std::bitset<containerBitsSize> container(invalidOutput[containerIndex]);
    container.set(index % containerBitsSize, 1);
    invalidOutput[containerIndex] = container.to_ulong();
}

static void invalidateTensors(const SifParams* input, SifOutputs* output)
{
    for (size_t t = 0; t < input->outputTensorsNr; ++t)
    {
        invalidateTensor(t, output->invalidMask);
    }
}

static SifReturn copyShapeTensorToOutput(const SifParams* input,
                                         SifOutputs*      output,
                                         unsigned         shapeTensorIndex,
                                         unsigned         outputTensorIndex)
{
    auto inputTensor  = input->inputTensors[shapeTensorIndex];
    auto outputTensor = output->outputTensors[outputTensorIndex];

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);
    memcpy(outputTensor->geometry.maxSizes, inputTensor->geometry.maxSizes, sizeof(tpc_lib_api::TensorGeometry::maxSizes));
    return tpc_lib_api::GLUE_SUCCESS;
}

// The following applies in case of min-size infer (used for dynamic shapes) where the inference happens with any graph-
// change, includng after sram slicing (But does not apply to max-size infer which has to happen at the beginning):
// Conv node may be a sliced conv node, which may be either a middle slice of the last slice in the dimension.
// A middle slice padding after should be 0, while the last slice padding should be the original padding after.
// However, which slice is last is unknown in compilation.
// Using the original padding after for all slices, and trimming the slice size to the slice max size protects
// middle slices by "removing" the unwanted padding from the size, while it allows the actual last slice to add it.
static TSize getConvDimSize(TSize* tensorDims, SifConvolutionMetadata* metadata, unsigned dim)
{
    ConvParamsIndices convIdx = ConvBaseNode::dimIndexToConvParamsIndices(dim);

    int outDimSize;

    if (metadata->params.paddingType == PADDING_SAME)
    {
        outDimSize = convOutputDimSizeSamePadding(tensorDims[dim], metadata->params.stride[convIdx.spatialIndex]);
    }
    else
    {
        outDimSize = convOutputDimSize(tensorDims[dim],
                                       metadata->params.kernel[convIdx.spatialIndex],
                                       metadata->params.stride[convIdx.spatialIndex],
                                       metadata->params.padding[convIdx.paddingBeforeIndex] +
                                           metadata->params.padding[convIdx.paddingAfterIndex],
                                       metadata->params.dilation[convIdx.spatialIndex]);
    }

    auto res = static_cast<TSize>(std::max(outDimSize, 0));

    if (metadata->maxOutputSizesKnown)
    {
        res = std::min(res, metadata->maxOutputSizes[dim]);
    }

    CHECK_MAX_VAL(res, unsigned);
    return res;
}

static SifReturn copyInputShape(const SifParams* input, SifOutputs* output, bool isMultiInput = false)
{
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(isMultiInput ? input->inputTensorsNr >= 1 : input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    // only input 0 actually provides data
    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);
    memcpy(outputTensor->geometry.maxSizes, inputTensor->geometry.maxSizes, sizeof(tpc_lib_api::TensorGeometry::maxSizes));

    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn
slice(const tpc_lib_api::TensorShapeInfo* inputTensor, tpc_lib_api::TensorShapeInfo* outputTensor, const SifSliceMetadata* metadata)
{
    const auto& inputTensorDimsSize  = inputTensor->geometry.maxSizes;
    auto&       outputTensorDimsSize = outputTensor->geometry.maxSizes;

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);

    for (size_t i = 0; i < inputTensor->geometry.dims; ++i)
    {
        VALIDATE_DIM_IN_RANGE(i, inputTensor->geometry.dims);
        TSize begin = metadata->starts[i];
        TSize end   = std::min((TSize)(inputTensorDimsSize[i]), metadata->ends[i]);
        TSize step  = metadata->steps[i];

        if (step == 0)
        {
            LOG_ERR(DYN_SHAPE, "step size must be greater than 0");
            return tpc_lib_api::GLUE_FAILED;
        }
        outputTensorDimsSize[i] = end > begin ? ceil((end - begin) / (double)step) : 0;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn transpose(const SifParams* input, SifOutputs* output)
{
    auto metadata = reinterpret_cast<SifTransposeMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    const auto& inputTensorDimsSize  = inputTensor->geometry.maxSizes;
    auto&       outputTensorDimsSize = outputTensor->geometry.maxSizes;

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);
    for (size_t d = 0; d < outputTensor->geometry.dims; ++d)
    {
        unsigned index          = metadata->permutation[d];
        outputTensorDimsSize[d] = inputTensorDimsSize[index];
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

static const unsigned GEMM_WIDTH_DIM              = 0;
static const unsigned GEMM_HEIGHT_DIM             = 1;
static const unsigned GEMM_OP_1_INDEX             = 0;
static const unsigned GEMM_OP_2_INDEX             = 1;
static const unsigned GEMM_BIAS_INDEX             = 2;
static const unsigned GEMM_OUTPUT_INDEX           = 0;
static const unsigned GEMM_TENSOR_COUNT_WITH_BIAS = 3;

static SifReturn validateGemmCommonDim(const SifParams* input, SifOutputs* output, bool transposeOutput)
{
    auto* metadata = reinterpret_cast<SifGemmMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2 || input->inputTensorsNr == 3);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto* op1              = input->inputTensors[GEMM_OP_1_INDEX];
    auto* op2              = input->inputTensors[GEMM_OP_2_INDEX];
    auto* outputTensor     = output->outputTensors[GEMM_OUTPUT_INDEX];
    auto* op1Dims          = op1->geometry.maxSizes;
    auto* op2Dims          = op2->geometry.maxSizes;
    auto* outputTensorDims = outputTensor->geometry.maxSizes;

    unsigned cd1Idx = metadata->params.transpose_a ? GEMM_HEIGHT_DIM : GEMM_WIDTH_DIM;
    unsigned cd2Idx = metadata->params.transpose_b ? GEMM_WIDTH_DIM : GEMM_HEIGHT_DIM;

    if (op1Dims[cd1Idx] != op2Dims[cd2Idx])
    {
        LOG_ERR(DYN_SHAPE, "mismatch between common dimensions for gemm SIF");
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    unsigned h1Idx = metadata->params.transpose_a ? GEMM_WIDTH_DIM : GEMM_HEIGHT_DIM;
    unsigned w2Idx = metadata->params.transpose_b ? GEMM_HEIGHT_DIM : GEMM_WIDTH_DIM;

    outputTensorDims[GEMM_WIDTH_DIM]  = transposeOutput ? op1Dims[h1Idx] : op2Dims[w2Idx];
    outputTensorDims[GEMM_HEIGHT_DIM] = transposeOutput ? op2Dims[w2Idx] : op1Dims[h1Idx];

    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn validateBatchDim(TSize* opSizes, unsigned opRank)
{
    for (int i = DIM_GEMM_BATCH; i < opRank; i++)
    {
        if (opSizes[i] > 1)
        {
            LOG_ERR(DYN_SHAPE, "Batch dims sizes in GEMM must be 1 or 0, but dim {} is {}", i, opSizes[i]);
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn validateOutputRank(unsigned op1Rank, unsigned op2Rank, unsigned outputRank)
{
    // Verify output tensor rank
    if (outputRank != std::max(op1Rank, op2Rank))
    {
        LOG_ERR(DYN_SHAPE, "Incorrect output rank, expected: {}, actual: {}", std::max(op1Rank, op2Rank), outputRank);
        return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn validateBgemmBroadcastShapes(const TensorShapeInfo* input1, const TensorShapeInfo* input2)
{
    TSize sizes1[input1->geometry.dims];
    TSize sizes2[input2->geometry.dims];
    memcpy(sizes1, input1->geometry.maxSizes, input1->geometry.dims * sizeof(TSize));
    memcpy(sizes2, input2->geometry.maxSizes, input2->geometry.dims * sizeof(TSize));
    SizeVector shape1Sizes = toSizeVector(sizes1, input1->geometry.dims);
    SizeVector shape2Sizes = toSizeVector(sizes2, input2->geometry.dims);

    if (!BatchGemmNode::isFullBroadcastLayout(shape1Sizes, shape2Sizes) &&
        !BatchGemmNode::isPartialBroadcastLayout(shape1Sizes, shape2Sizes) &&
        !BatchGemmNode::isSymmetricLayout(shape1Sizes, shape2Sizes))
    {
        LOG_ERR(DYN_SHAPE,
                "BGEMM input shapes cannot be broadcasted: [{},{},{},{},{}] vs [{},{},{},{},{}]",
                input1->geometry.maxSizes[0],
                input1->geometry.maxSizes[1],
                input1->geometry.maxSizes[2],
                input1->geometry.maxSizes[3],
                input1->geometry.maxSizes[4],
                input2->geometry.maxSizes[0],
                input2->geometry.maxSizes[1],
                input2->geometry.maxSizes[2],
                input2->geometry.maxSizes[3],
                input2->geometry.maxSizes[4]);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn validateBias(const TensorShapeInfo* output, const TensorShapeInfo* bias)
{
    bool isCompatible = false;

    for (int i = 0; i < std::max(output->geometry.dims, bias->geometry.dims); i++)
    {
        isCompatible = (output->geometry.maxSizes[i] == bias->geometry.maxSizes[i]) || (output->geometry.maxSizes[i] >= 1 && bias->geometry.maxSizes[i] <= 1);
        if (!isCompatible)
        {
            LOG_ERR(DYN_SHAPE,
                    "GEMM output & bias shapes are incompatible: [{},{},{},{},{}] vs [{},{},{},{},{}]",
                    output->geometry.maxSizes[0],
                    output->geometry.maxSizes[1],
                    output->geometry.maxSizes[2],
                    output->geometry.maxSizes[3],
                    output->geometry.maxSizes[4],
                    bias->geometry.maxSizes[0],
                    bias->geometry.maxSizes[1],
                    bias->geometry.maxSizes[2],
                    bias->geometry.maxSizes[3],
                    bias->geometry.maxSizes[4]);
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

static TSize accessBgemmElement(TSize* opSizes, unsigned opTotalSize, unsigned index)
{
    if (index >= opTotalSize)
    {
        return 1;
    }
    return opSizes[index];
}

static void
inferBatchDimensions(TSize* opASizes, unsigned opARank, TSize* opBSizes, unsigned opBRank, TSize* outSizes)
{
    // this function assumes that all dims higher than the rank are 0\1
    std::memset(opASizes + opARank, 0, (tpc_lib_api::MAX_TENSOR_DIM - opARank));
    std::memset(opBSizes + opBRank, 0, (tpc_lib_api::MAX_TENSOR_DIM - opBRank));
    for (size_t i = DIM_GEMM_BATCH; i < std::max(opARank, opBRank); i++)
    {
        if (accessBgemmElement(opASizes, opARank, i) == 0 || accessBgemmElement(opBSizes, opBRank, i) == 0)
        {
            outSizes[i] = 0;
        }
        else
        {
            outSizes[i] = std::max(opASizes[i], opBSizes[i]);
        }
    }
}

static SifReturn gemm(const SifParams* input, SifOutputs* output, bool transposeOutput)
{
    // Verify inputs and output
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2 || input->inputTensorsNr == 3);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto op1               = input->inputTensors[GEMM_OP_1_INDEX];
    auto op2               = input->inputTensors[GEMM_OP_2_INDEX];
    auto op1Sizes          = op1->geometry.maxSizes;
    auto op2Sizes          = op2->geometry.maxSizes;
    auto op1Rank           = op1->geometry.dims;
    auto op2Rank           = op2->geometry.dims;
    auto outputTensorSizes = output->outputTensors[GEMM_OUTPUT_INDEX]->geometry.maxSizes;

    if (op1Rank < DIM_GEMM_BATCH || op2Rank < DIM_GEMM_BATCH)
    {
        LOG_ERR(DYN_SHAPE, "GEMM - both operands must have a rank of at least 2, but have {} & {}", op1Rank, op2Rank);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    // Verify operands' dimensionality and rank
    RETURN_ON_FAILURE(validateGemmCommonDim(input, output, transposeOutput));
    RETURN_ON_FAILURE(validateBatchDim(op1Sizes, op1Rank));
    RETURN_ON_FAILURE(validateBatchDim(op2Sizes, op2Rank));

    // Infer output batch dimensions (should be 0 or 1)
    inferBatchDimensions(op1Sizes, op1Rank, op2Sizes, op2Rank, outputTensorSizes);
    if (input->inputTensorsNr == GEMM_TENSOR_COUNT_WITH_BIAS)
    {
        // Verify bias is broadcastable to output
        RETURN_ON_FAILURE(validateBias(output->outputTensors[GEMM_OUTPUT_INDEX], input->inputTensors[GEMM_BIAS_INDEX]));
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn batchGemm(const SifParams* input, SifOutputs* output, bool enforceDimsEqual, bool transposeOutput)
{
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2 || input->inputTensorsNr == 3);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputRank         = input->inputTensors[GEMM_OP_1_INDEX]->geometry.dims;
    auto weightsRank       = input->inputTensors[GEMM_OP_2_INDEX]->geometry.dims;
    auto inputTensor       = input->inputTensors[GEMM_OP_1_INDEX];
    auto weightsTensor     = input->inputTensors[GEMM_OP_2_INDEX];
    auto inputSizes        = inputTensor->geometry.maxSizes;
    auto weightSizes       = weightsTensor->geometry.maxSizes;
    auto outputTensorSizes = output->outputTensors[GEMM_OUTPUT_INDEX]->geometry.maxSizes;
    auto outputRank        = output->outputTensors[GEMM_OUTPUT_INDEX]->geometry.dims;

    if (inputRank < DIM_GEMM_BATCH || weightsRank < DIM_GEMM_BATCH)
    {
        LOG_ERR(DYN_SHAPE,
                "BGEMM - both operands must have a rank of at least 2, but have {} & {}",
                inputRank,
                weightsRank);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    if (enforceDimsEqual && (inputRank != weightsRank))
    {
        LOG_ERR(DYN_SHAPE,
                "BGEMM - operands have been forced to be of the same rank, but have {} & {}",
                inputRank,
                weightsRank);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // Verify common dimension size and infer output sizes
    RETURN_ON_FAILURE(validateGemmCommonDim(input, output, transposeOutput));
    // Verify inputs are mutually broadcastable
    RETURN_ON_FAILURE(validateBgemmBroadcastShapes(inputTensor, weightsTensor));
    RETURN_ON_FAILURE(validateOutputRank(inputRank, weightsRank, outputRank));

    // Infer output batch dimensions
    inferBatchDimensions(inputSizes, inputRank, weightSizes, weightsRank, outputTensorSizes);
    if (input->inputTensorsNr == GEMM_TENSOR_COUNT_WITH_BIAS)
    {
        // Verify bias is broadcastable to output
        RETURN_ON_FAILURE(validateBias(output->outputTensors[GEMM_OUTPUT_INDEX], input->inputTensors[GEMM_BIAS_INDEX]));
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn tensorViewAccessInput(const SifParams* input, SifOutputs* output, SifTensorViewMetadata* metadata)
{
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr >= 1);

    auto        inputTensor         = input->inputTensors[0];
    const auto& inputTensorDimsSize = inputTensor->geometry.maxSizes;

    for (size_t t = 0; t < input->outputTensorsNr; ++t)
    {
        auto tvd = metadata->data[t];

        VALIDATE_DIMS_EQUAL(output->outputTensors[t]->geometry.dims, tvd.dims);
        auto& outputTensorDimsSize = output->outputTensors[t]->geometry.maxSizes;

        for (size_t d = 0; d < tvd.dims; ++d)
        {
            TOffset begin = tvd.offsets[d];
            TSize end = std::min(static_cast<TSize>(inputTensorDimsSize[d]), tvd.offsets[d] + tvd.sizes[d]);
            outputTensorDimsSize[d] = end > begin ? end - begin : 0;
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

static SifReturn tensorViewAccessOutput(const SifParams* input, SifOutputs* output, SifTensorViewMetadata* metadata)
{
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto  inputTensor          = input->inputTensors[0];
    auto  outputTensor         = output->outputTensors[0];
    auto& outputTensorDimsSize = outputTensor->geometry.maxSizes;

    auto tvd = metadata->data[0];
    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);
    for (size_t d = 0; d < tvd.dims; ++d)
    {
        outputTensorDimsSize[d] = 1;
    }

    for (size_t t = 0; t < input->inputTensorsNr; ++t)
    {
        auto        tvd                 = metadata->data[t];
        const auto& inputTensorDimsSize = input->inputTensors[t]->geometry.maxSizes;

        for (size_t d = 0; d < tvd.dims; ++d)
        {
            if (outputTensorDimsSize[d] < tvd.offsets[d] + inputTensorDimsSize[d] && inputTensorDimsSize[d] > 0)
            {
                outputTensorDimsSize[d] = tvd.offsets[d] + inputTensorDimsSize[d];
            }
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn dmaMemcpyShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return copyInputShape(input, output, true);
}

SifReturn dmaPhysicalConcatSplitDMAShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return copyInputShape(input, output, true);
}

// Temporary
SifReturn dmaPhysicalConcatContainerShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return copyInputShape(input, output, true);
}

SifReturn dmaMemsetShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 0;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    if (input->inputTensorsNr == SHAPE_TENSOR_INDEX)
    {
        invalidateTensors(input, output);
        return tpc_lib_api::GLUE_SUCCESS;
    }
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn concatenateShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifConcatenateMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto outputTensor = output->outputTensors[0];

    if (metadata->withShapeTensor)
    {
        memcpy(outputTensor->geometry.maxSizes,
               input->inputTensors[input->inputTensorsNr - 1]->geometry.maxSizes,
               sizeof(tpc_lib_api::TensorGeometry::maxSizes));
    }
    else
    {
        auto  inputTensor = input->inputTensors[0];
        TSize dimSum      = 0;
        for (size_t t = 0; t < input->inputTensorsNr; ++t)
        {
            VALIDATE_DIM_IN_RANGE(metadata->axis, input->inputTensors[t]->geometry.dims);
            const auto& inputTensorDimsSize = input->inputTensors[t]->geometry.maxSizes;
            TSize       dimSize             = inputTensorDimsSize[metadata->axis];
            dimSum += dimSize;
        }

        VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);
        memcpy(outputTensor->geometry.maxSizes, inputTensor->geometry.maxSizes, sizeof(tpc_lib_api::TensorGeometry::maxSizes));
        outputTensor->geometry.maxSizes[metadata->axis] = dimSum;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn flattenShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifFlattenMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    const auto& inputTensorDimsSize  = inputTensor->geometry.maxSizes;
    auto&       outputTensorDimsSize = outputTensor->geometry.maxSizes;

    for (size_t d = 0; d < outputTensor->geometry.dims; ++d)
    {
        outputTensorDimsSize[d] = 1;
    }

    VALIDATE_DIM_IN_RANGE(metadata->axis, inputTensor->geometry.dims);

    for (size_t d = 0; d < inputTensor->geometry.dims; ++d)
    {
        if (d <= metadata->axis)
        {
            outputTensorDimsSize[0] *= inputTensorDimsSize[d];
        }
        else
        {
            outputTensorDimsSize[1] *= inputTensorDimsSize[d];
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn splitShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifSplitMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr >= 1);

    if (metadata == nullptr)
    {
        LOG_ERR(DYN_SHAPE, "null metadata in SIF {}", HLLOG_FUNC);
        return tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    size_t expectedSize = sizeof(SifSplitHeader) + sizeof(TSize) * metadata->header.splitDimSizesNr;
    if (expectedSize != input->nodeParams.nodeParamsSize)
    {
        LOG_ERR(DYN_SHAPE,
                "incorrect split metadata size, expected: {}, actual: {}",
                expectedSize,
                input->nodeParams.nodeParamsSize);
        return tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE;
    }

    auto        inputTensor         = input->inputTensors[0];
    const auto& inputTensorDimsSize = inputTensor->geometry.maxSizes;

    VALIDATE_DIM_IN_RANGE(metadata->header.axis, inputTensor->geometry.dims);

    TSize splitAxisDimSize          = inputTensorDimsSize[metadata->header.axis];
    TSize allocatedSplitAxisDimSize = 0;

    // If we split for example [10,5,6] on axis 2 to 6 tensors [10,5] (and not [10,5,1])
    bool areOutputAllTensorsReduceOneDim = true;
    for (size_t t = 0; t < input->outputTensorsNr; ++t)
    {
        if (output->outputTensors[t]->geometry.dims != inputTensor->geometry.dims - 1)
        {
            areOutputAllTensorsReduceOneDim = false;
            break;
        }
    }

    if (splitAxisDimSize == input->outputTensorsNr && inputTensor->geometry.dims - 1 == metadata->header.axis &&
        areOutputAllTensorsReduceOneDim)
    {
        for (size_t t = 0; t < input->outputTensorsNr; ++t)
        {
            auto& outputTensorDimsSize = output->outputTensors[t]->geometry.maxSizes;
            for (size_t d = 0; d < inputTensor->geometry.dims - 1; ++d)
            {
                outputTensorDimsSize[d] = inputTensorDimsSize[d];
            }
        }
    }
    else
    {
        for (size_t t = 0; t < input->outputTensorsNr; ++t)
        {
            VALIDATE_DIMS_EQUAL(output->outputTensors[t]->geometry.dims, inputTensor->geometry.dims);
            auto& outputTensorDimsSize = output->outputTensors[t]->geometry.maxSizes;

            for (size_t d = 0; d < inputTensor->geometry.dims; ++d)
            {
                outputTensorDimsSize[d] = inputTensorDimsSize[d];
                if (d == metadata->header.axis)
                {
                    TSize dimSize = std::min(metadata->splitDimSizes[t], splitAxisDimSize - allocatedSplitAxisDimSize);
                    outputTensorDimsSize[d] = dimSize;
                    allocatedSplitAxisDimSize += dimSize;
                }
            }
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn expandDimsShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifExpandDimsMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    const auto& inputTensorDimsSize  = inputTensor->geometry.maxSizes;
    auto&       outputTensorDimsSize = outputTensor->geometry.maxSizes;

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims + 1);
    outputTensorDimsSize[metadata->axis] = 1;

    for (size_t d = 0; d < inputTensor->geometry.dims; ++d)
    {
        unsigned offset                  = d < metadata->axis ? 0 : 1;
        outputTensorDimsSize[d + offset] = inputTensorDimsSize[d];
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn mergeShapesInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifMergeShapesMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto  outputTensor         = output->outputTensors[0];
    auto& outputTensorDimsSize = outputTensor->geometry.maxSizes;

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, metadata->outputDim);

    for (size_t d = 0; d < outputTensor->geometry.dims; ++d)
    {
        if (metadata->dimMap[d].inputIdx != -1)
        {
            VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= metadata->dimMap[d].inputIdx);
            const auto& inputTensor = input->inputTensors[metadata->dimMap[d].inputIdx];
            VALIDATE_DIM_IN_RANGE(metadata->dimMap[d].dimIdx, inputTensor->geometry.dims);
            outputTensorDimsSize[d] = inputTensor->geometry.maxSizes[metadata->dimMap[d].dimIdx];
        }
        else
        {
            outputTensorDimsSize[d] = metadata->fillValue;
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn sliceShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 1;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    if (input->inputTensorsNr == SHAPE_TENSOR_INDEX)
    {
        auto metadata = reinterpret_cast<SifSliceMetadata*>(input->nodeParams.nodeParams);
        VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);

        auto inputTensor  = input->inputTensors[0];
        auto outputTensor = output->outputTensors[0];

        return slice(inputTensor, outputTensor, metadata);
    }
    else
    {
        VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 2);

        return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
    }
}

SifReturn sliceAxisShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifSliceAxisMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    if (input->inputTensorsNr > 1)
    {
        VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2);
    }
    else
    {
        VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    }

    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    const auto& inputTensorDimsSize  = inputTensor->geometry.maxSizes;
    auto&       outputTensorDimsSize = outputTensor->geometry.maxSizes;

    VALIDATE_DIM_IN_RANGE(metadata->axis, inputTensor->geometry.dims);

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);

    for (size_t d = 0; d < inputTensor->geometry.dims; ++d)
    {
        outputTensorDimsSize[d] = inputTensorDimsSize[d];
        if (d == metadata->axis)
        {
            TSize begin = metadata->begin;
            TSize end   = metadata->end;

            if (input->inputTensorsNr > 1)
            {
                end = input->inputTensors[1]->geometry.maxSizes[d];
            }

            end                     = std::min((TSize)inputTensorDimsSize[d], end);
            outputTensorDimsSize[d] = end > begin ? end - begin : 0;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn sliceBackwardShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 1;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn sliceInsertShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned ORIGINAL_TENSOR_INDEX = 0;
    const unsigned OUTPUT_TENSOR_INDEX   = 0;

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    return copyShapeTensorToOutput(input, output, ORIGINAL_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn reshapeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned INPUT_TENSOR_INDEX  = 0;
    const unsigned SHAPE_TENSOR_INDEX  = 1;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    if (input->inputTensorsNr == SHAPE_TENSOR_INDEX)
    {
        invalidateTensors(input, output);
        return tpc_lib_api::GLUE_SUCCESS;
    }
    // Check that the total number of elements is the same

    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor = input->inputTensors[INPUT_TENSOR_INDEX];
    auto shapeTensor = input->inputTensors[SHAPE_TENSOR_INDEX];
    VALIDATE_SAME_NUMBER_OF_ELEMENTS(inputTensor, shapeTensor);

    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn staticReshapeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto                       inputTensor  = input->inputTensors[0];
    auto                       outputTensor = output->outputTensors[0];
    synStaticReshapeSifParams* params       = reinterpret_cast<synStaticReshapeSifParams*>(input->nodeParams.nodeParams);

    // There is the same number of dynamic dimension in the input and output tensors,
    // and there's 1:1 correspondence between them.
    //
    // (input)   S1 S2 D1 S3 D2
    // (output)  S4 D1 D2 S5
    // We need to copy D1 from its place in the input to its place in the output, same with D2.
    // However S4 S5 are taken from params->outputMaxSizes, not from the input tensor (because
    // the dimensions of the input tensor are completely different).
    //
    // The special case where there isn't a 1:1 correspondence between input and output dynamic dimensions is the case
    // where we have only 1 dynamic dimension and it is the most outer dimension.
    // example (FCD on the left):
    // inMax = (x, Y)     inActual = (x, y)
    // outMax = (x*Y).
    // in this case we will get outActual of: (x*y)
    // we can get that by: outActual[outDim] = outMax[outDim] * inActual[inDim] / inMax(inDim) = (x*Y) * y / Y

    // Copy static dimensions from the SIF parameter.
    memcpy(outputTensor->geometry.maxSizes, params->outputMaxSizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));
    unsigned dynamicDimMap[ARRAY_SIZE(inputTensor->geometry.maxSizes)] = {0};
    unsigned dynamicOutIdx                                 = 0;
    for (unsigned inIdx = 0; inIdx < inputTensor->geometry.dims; inIdx++)
    {
        if (params->inputStaticDims[inIdx] == 0)
        {
            dynamicDimMap[dynamicOutIdx++] = inIdx;
        }
    }
    unsigned numDynamicOutputDims =
        std::count(params->outputStaticDims, params->outputStaticDims + outputTensor->geometry.dims, 0);
    VALIDATE_DIMS_EQUAL(numDynamicOutputDims, dynamicOutIdx);

    // Copy dynamic dimensions from the input tensor.
    dynamicOutIdx = 0;
    for (unsigned outIdx = 0; outIdx < outputTensor->geometry.dims; outIdx++)
    {
        if (params->outputStaticDims[outIdx] == 0)  // dynamic dim
        {
            unsigned dynamicInIdx = dynamicDimMap[dynamicOutIdx++];
            if (params->inputMaxSizes[dynamicInIdx] != 0)
            {
                // use calculation that works in case of reshaped dynamic dimension
                uint64_t mul = (uint64_t)outputTensor->geometry.maxSizes[outIdx] * (uint64_t)inputTensor->geometry.maxSizes[dynamicInIdx];
                if (mul % params->inputMaxSizes[dynamicInIdx] != 0)
                {
                    LOG_ERR(DYN_SHAPE, "incorrect static reshape dynamic dimension");
                    return tpc_lib_api::GLUE_FAILED;
                }
                outputTensor->geometry.maxSizes[outIdx] = mul / params->inputMaxSizes[dynamicInIdx];
            }
            else  // just use regular copy - assume that the dynamic dimension is the same in input and output
            {
                outputTensor->geometry.maxSizes[outIdx] = inputTensor->geometry.maxSizes[dynamicInIdx];
            }
        }
    }

    // final validation - make sure all dimensions passed properly
    VALIDATE_SAME_NUMBER_OF_ELEMENTS(inputTensor, outputTensor);

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn broadcastShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 1;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    if (input->inputTensorsNr == SHAPE_TENSOR_INDEX)
    {
        invalidateTensors(input, output);
        return tpc_lib_api::GLUE_SUCCESS;
    }
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn identityShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    if (input->inputTensorsNr == 2)
    {
        VALIDATE_PARAMS(input, output);
        VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);
        return copyShapeTensorToOutput(input, output, 1, 0);
    }
    return copyInputShape(input, output);
}

SifReturn reductionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    memcpy(outputTensor->geometry.maxSizes, inputTensor->geometry.maxSizes, sizeof(tpc_lib_api::TensorGeometry::maxSizes));
    return tpc_lib_api::GLUE_SUCCESS;
}

static bool verifyViewMemoryAccess(const SifParams* input)
{
    // verify that the last strided element does not exceed the original tensor size
    // this is done for minimal sizes only, verification for actual sizes are done during runtime
    const TensorShapeInfo* real    = input->inputTensors[0];
    const TensorShapeInfo* view    = input->inputTensors[1];
    const TensorShapeInfo* strides = input->inputTensors[2];
    const TensorShapeInfo* offset  = input->inputTensors[3];

    uint64_t realTensorElements =
        std::accumulate(real->geometry.maxSizes, real->geometry.maxSizes + real->geometry.dims, 1, std::multiplies<uint64_t>());
    if (realTensorElements == 0) return true;
    uint64_t lastElementOffset = 0;
    for (unsigned d = 0; d < view->geometry.dims; d++)
    {
        if (view->geometry.maxSizes[d] == 0) return true;
        lastElementOffset += static_cast<uint64_t>(strides->geometry.maxSizes[d]) * (view->geometry.maxSizes[d] - 1);
    }
    if (offset->geometry.maxSizes[0] + lastElementOffset >= realTensorElements)
    {
        return false;
    }
    return true;
}

SifReturn stridedViewShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 1;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    if (input->inputTensorsNr == 4 && !verifyViewMemoryAccess(input))
    {
        LOG_WARN(DYN_SHAPE, "Strided View might access memory outside of original tensor range!");
    }

    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn stridedInsertShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr >= 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    if (input->inputTensorsNr == 4 && !verifyViewMemoryAccess(input))
    {
        LOG_WARN(DYN_SHAPE, "Strided Insert might access memory outside of original tensor range!");
    }

    return copyInputShape(input, output, true);
}

SifReturn tensorViewShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifTensorViewMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);

    if (metadata == nullptr)
    {
        LOG_ERR(DYN_SHAPE, "null metadata in SIF {}", HLLOG_FUNC);
        return tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE;
    }
    size_t expectedSize = sizeof(SifTensorViewHeader) + sizeof(SifTensorViewData) * metadata->header.viewsNr;
    if (expectedSize != input->nodeParams.nodeParamsSize)
    {
        LOG_ERR(DYN_SHAPE,
                "incorrect tensor view metadata size, expected: {}, actual: {}",
                expectedSize,
                input->nodeParams.nodeParamsSize);
        return tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE;
    }

    return metadata->header.accessInput ? tensorViewAccessInput(input, output, metadata)
                                        : tensorViewAccessOutput(input, output, metadata);
}

SifReturn transposeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return transpose(input, output);
}

SifReturn convolutionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifConvolutionMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2 || input->inputTensorsNr == 3);  // 3 in case of bias tensor
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensor  = input->inputTensors[0];
    auto weightTensor = input->inputTensors[1];
    auto outputTensor = output->outputTensors[0];

    if (inputTensor->geometry.dims < 4)
    {
        LOG_ERR(DYN_SHAPE, "convolution input dims must be at least 4, actual: {}", inputTensor->geometry.dims);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    auto inputTensorDims  = inputTensor->geometry.maxSizes;
    auto weightTensorDims = weightTensor->geometry.maxSizes;
    auto outputTensorDims = outputTensor->geometry.maxSizes;

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensor->geometry.dims);

    outputTensorDims[DIM_C] = weightTensorDims[DIM_C];
    outputTensorDims[DIM_W] = getConvDimSize(inputTensorDims, metadata, DIM_W);
    outputTensorDims[DIM_H] = getConvDimSize(inputTensorDims, metadata, DIM_H);

    if (outputTensor->geometry.dims == 4)
    {
        outputTensorDims[DIM_B] = inputTensorDims[DIM_B];
    }
    else
    {
        outputTensorDims[DIM_B_FOR_5D_TENSOR] = inputTensorDims[DIM_B_FOR_5D_TENSOR];
        outputTensorDims[DIM_D_FOR_5D_TENSOR] = getConvDimSize(inputTensorDims, metadata, DIM_D_FOR_5D_TENSOR);
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn convDeDwShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifConvolutionMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2 || input->inputTensorsNr == 3);  // 3 in case of bias tensor
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto inputTensorDeDy = input->inputTensors[0];
    auto inputTensorDeDw = input->inputTensors[1];
    auto outputTensor    = output->outputTensors[0];

    if (inputTensorDeDy->geometry.dims < 4)
    {
        LOG_ERR(DYN_SHAPE, "dedy input dims must be at least 4, actual: {}", inputTensorDeDy->geometry.dims);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    if (inputTensorDeDy->geometry.dims != inputTensorDeDw->geometry.dims)
    {
        LOG_ERR(DYN_SHAPE, "dedw input tensor rank must be equal to dedy input tensor rank");
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    auto inputTensorDeDyDims = inputTensorDeDy->geometry.maxSizes;
    auto inputTensorDeDwDims = inputTensorDeDw->geometry.maxSizes;
    auto outputTensorDims    = outputTensor->geometry.maxSizes;

    auto kernel = metadata->params.kernel;

    VALIDATE_DIMS_EQUAL(outputTensor->geometry.dims, inputTensorDeDy->geometry.dims);

    outputTensorDims[WEIGHT_DIM_K] = inputTensorDeDyDims[WEIGHT_DIM_K];
    outputTensorDims[WEIGHT_DIM_C] = inputTensorDeDwDims[DIM_C] / metadata->params.nGroups;
    outputTensorDims[WEIGHT_DIM_S] = kernel[CONV_KERNEL_WIDTH];
    outputTensorDims[WEIGHT_DIM_R] = kernel[CONV_KERNEL_HEIGHT];
    outputTensorDims[WEIGHT_DIM_Q] = kernel[CONV_KERNEL_DEPTH];

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn convDeDxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 2;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 3);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn gemmShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return gemm(input, output, false);
}

SifReturn gemmDeDwShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return gemm(input, output, true);
}

SifReturn gemmDeDxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return gemm(input, output, false);
}

SifReturn gemmFcShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return gemm(input, output, false);
}

SifReturn batchGemmShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return batchGemm(input, output, false, false);
}

SifReturn batchGemmDeDwShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return batchGemm(input, output, true, true);
}

SifReturn batchGemmDeDxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return batchGemm(input, output, true, false);
}

SifReturn squeezeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto metadata = reinterpret_cast<SifSqueezeMetadata*>(input->nodeParams.nodeParams);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);

    auto inputTensor  = input->inputTensors[0];
    auto outputTensor = output->outputTensors[0];

    const auto& inputTensorDimsSize  = inputTensor->geometry.maxSizes;
    auto&       outputTensorDimsSize = outputTensor->geometry.maxSizes;
    size_t      count                = 0;

    for (size_t d = 0; d < outputTensor->geometry.dims; ++d)
    {
        outputTensorDimsSize[d] = 1;
    }
    for (size_t d = 0; d < inputTensor->geometry.dims; ++d)
    {
        if (!metadata->squeezeDim[d])
        {
            outputTensorDimsSize[count] = inputTensorDimsSize[d];
            count++;
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn frobeniusNormShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn momentsShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 2);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);

    unsigned dimsNum = input->inputTensors[0]->geometry.dims;
    if (dimsNum == 0 || dimsNum > MAX_DIMENSIONS_NUM)
    {
        LOG_ERR(DYN_SHAPE, "Invalid input tensor rank: {}", dimsNum);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    int channelsNum = input->inputTensors[0]->geometry.maxSizes[0];
    if (channelsNum == 0)
    {
        LOG_ERR(DYN_SHAPE, "Number of channels cannot be 0");
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    for (int outTesorIndex = 0; outTesorIndex < 2; outTesorIndex++)
    {  // first tensor is the mean calc result and the second is the variance. both should have the same shape.
        output->outputTensors[outTesorIndex]->geometry.dims     = dimsNum;
        output->outputTensors[outTesorIndex]->geometry.maxSizes[0] = channelsNum;
        for (unsigned i = 1; i < dimsNum; i++)
        {
            output->outputTensors[outTesorIndex]->geometry.maxSizes[i] = 1;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn nopShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{  // The function sets the output dims to 0 and returns success indication, to be used in nodes where no shape
   // inference is needed.
    for (unsigned i = 0; i < input->outputTensorsNr; i++)
    {
        output->outputTensors[i]->geometry.dims = 0;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn rotateShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    const unsigned SHAPE_TENSOR_INDEX  = 1;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);
    auto shapeTensor   = input->inputTensors[SHAPE_TENSOR_INDEX];
    auto outputTensor  = output->outputTensors[OUTPUT_TENSOR_INDEX];
    outputTensor->geometry.dims = shapeTensor->geometry.dims;
    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn tfBatchNormShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    const unsigned SHAPE_TENSOR_INDEX  = 0;
    const unsigned OUTPUT_TENSOR_INDEX = 0;

    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 5);

    output->outputTensors[0]->geometry.dims = input->inputTensors[0]->geometry.dims;
    return copyShapeTensorToOutput(input, output, SHAPE_TENSOR_INDEX, OUTPUT_TENSOR_INDEX);
}

SifReturn noSupportShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    LOG_ERR(DYN_SHAPE, "The node does not support dynamic shapes");
    return tpc_lib_api::GLUE_FAILED;
}

SifReturn dynamicSplitShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    const unsigned DATA_TENSOR_INDEX  = 0;
    const unsigned SHAPE_TENSOR_INDEX = 1;

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2);

    const auto* sifData    = reinterpret_cast<const SifDynamicSplitMetadata*>(input->nodeParams.nodeParams);
    TSize       dimSum     = 0;
    unsigned    splitDim   = sifData->axis;
    unsigned*   tensorData = input->inputTensors[SHAPE_TENSOR_INDEX]->hostAddress;
    NULL_CHECK(tensorData, "Shape tensor data is null!");

    for (unsigned i = 0, *curDimPtr = tensorData + splitDim; i < input->outputTensorsNr;
         ++i, curDimPtr += SYN_MAX_TENSOR_DIM)
    {
        dimSum += *curDimPtr;
    }

    // In the wide bucket mode, the input size may be 0, but the H2D tensor data does not sum to 0
    // Skip the check when doing wide bucket and the size is 0
    if (!GCFG_ENABLE_WIDE_BUCKET.value() || input->inputTensors[DATA_TENSOR_INDEX]->geometry.maxSizes[splitDim] != 0)
    {
        VALIDATE_DIMS_EQUAL(dimSum, input->inputTensors[DATA_TENSOR_INDEX]->geometry.maxSizes[splitDim]);
    }

    for (unsigned i = 0, *curDimPtr = tensorData; i < input->outputTensorsNr; ++i, curDimPtr += SYN_MAX_TENSOR_DIM)
    {
        std::copy(curDimPtr, curDimPtr + SYN_MAX_TENSOR_DIM, output->outputTensors[i]->geometry.maxSizes);
    }

    // In the wide bucket mode, if the input size is 0, force the output size to be 0
    // regardless of what H2D tensor data says
    if (GCFG_ENABLE_WIDE_BUCKET.value() && input->inputTensors[DATA_TENSOR_INDEX]->geometry.maxSizes[splitDim] == 0)
    {
        for (unsigned i = 0, *curDimPtr = tensorData; i < input->outputTensorsNr;
             ++i, curDimPtr += SYN_MAX_TENSOR_DIM)
        {
            curDimPtr[splitDim] = 0;
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn cudBnFwdExShapeInferenceFunction3or4Inputs(const SifParams* input, SifOutputs* output)
{  // this means only the must have inputs (X, Scale, Bias)-> there will be only one output (Y)
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);
    const unsigned X_INPUT_TENSOR_INDEX  = 0;
    const unsigned Y_OUTPUT_TENSOR_INDEX = 0;
    return copyShapeTensorToOutput(input, output, X_INPUT_TENSOR_INDEX, Y_OUTPUT_TENSOR_INDEX);
}

SifReturn cudBnFwdExShapeInferenceFunction5or6Inputs(const SifParams* input, SifOutputs* output, int inputsNum)
{
    // the 5 inputs must be: X, Scale, Bias, running mean and running var and the 3 (optional 5) outputs are Y, running
    // mean and running var with optional istd and mean in the output
    auto ret = tpc_lib_api::GLUE_FAILED;

    const unsigned RUNNING_MEAN_INPUT_TENSOR_INDEX = (inputsNum == 5) ? 3 : 4;
    switch (input->outputTensorsNr)
    {
        case 5:
        {
            const unsigned MEAN_OUTPUT_TENSOR_INDEX = 3;
            const unsigned ISTD_OUTPUT_TENSOR_INDEX = 4;
            ret = copyShapeTensorToOutput(input, output, RUNNING_MEAN_INPUT_TENSOR_INDEX, MEAN_OUTPUT_TENSOR_INDEX);
            if (ret != tpc_lib_api::GLUE_SUCCESS) return ret;
            ret = copyShapeTensorToOutput(input, output, RUNNING_MEAN_INPUT_TENSOR_INDEX, ISTD_OUTPUT_TENSOR_INDEX);
            if (ret != tpc_lib_api::GLUE_SUCCESS) return ret;
            // Note - intentional fall thorough
        }
        case 3:
        {
            const unsigned X_INPUT_TENSOR_INDEX             = 0;
            const unsigned Y_OUTPUT_TENSOR_INDEX            = 0;
            const unsigned RUNNING_MEAN_OUTPUT_TENSOR_INDEX = 1;
            const unsigned RUNNING_VAR_OUTPUT_TENSOR_INDEX  = 2;
            ret                                             = copyShapeTensorToOutput(input,
                                          output,
                                          RUNNING_MEAN_INPUT_TENSOR_INDEX,
                                          RUNNING_MEAN_OUTPUT_TENSOR_INDEX);
            if (ret != tpc_lib_api::GLUE_SUCCESS) return ret;
            ret = copyShapeTensorToOutput(input,
                                          output,
                                          RUNNING_MEAN_INPUT_TENSOR_INDEX,
                                          RUNNING_VAR_OUTPUT_TENSOR_INDEX);
            if (ret != tpc_lib_api::GLUE_SUCCESS) return ret;
            ret = copyShapeTensorToOutput(input, output, X_INPUT_TENSOR_INDEX, Y_OUTPUT_TENSOR_INDEX);
            if (ret != tpc_lib_api::GLUE_SUCCESS) return ret;
            break;
        }
        default:
        {
            LOG_ERR(DYN_SHAPE,
                    "Unexpected number of outputs in SIF: {}, actual {}, while only 3 or 5 are expected.",
                    HLLOG_FUNC,
                    input->outputTensorsNr);
            return tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT;
        }
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn cudBnFwdExShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    VALIDATE_PARAMS(input, output);

    switch (input->inputTensorsNr)
    {
        case 3:
        case 4:
            return cudBnFwdExShapeInferenceFunction3or4Inputs(input, output);
        case 5:
        case 6:
            return cudBnFwdExShapeInferenceFunction5or6Inputs(input, output, input->inputTensorsNr);
        default:
            return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT;
    }
}

SifReturn physicalSplitShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    return dynamicSplitShapeInferenceFunction(deviceId, input, output);
}

SifReturn cudBnBwdShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 6);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 3 || input->outputTensorsNr == 4);

    const unsigned X_INPUT_TENSOR_INDEX   = 0;
    const unsigned DX_OUTPUT_TENSOR_INDEX = 0;
    auto successIndication = copyShapeTensorToOutput(input, output, X_INPUT_TENSOR_INDEX, DX_OUTPUT_TENSOR_INDEX);
    if (successIndication != tpc_lib_api::GLUE_SUCCESS)
    {
        return successIndication;
    }
    const unsigned SCALE_INPUT_TENSOR_INDEX   = 2;
    const unsigned DSCALE_OUTPUT_TENSOR_INDEX = 1;
    successIndication = copyShapeTensorToOutput(input, output, SCALE_INPUT_TENSOR_INDEX, DSCALE_OUTPUT_TENSOR_INDEX);
    if (successIndication != tpc_lib_api::GLUE_SUCCESS)
    {
        return successIndication;
    }

    const unsigned BIAS_INPUT_TENSOR_INDEX   = 3;
    const unsigned DBIAS_OUTPUT_TENSOR_INDEX = 2;
    successIndication = copyShapeTensorToOutput(input, output, BIAS_INPUT_TENSOR_INDEX, DBIAS_OUTPUT_TENSOR_INDEX);
    if (successIndication != tpc_lib_api::GLUE_SUCCESS)
    {
        return successIndication;
    }

    if (input->outputTensorsNr == 4)
    {
        const unsigned DZ_OUTPUT_TENSOR_INDEX = 3;
        successIndication = copyShapeTensorToOutput(input, output, X_INPUT_TENSOR_INDEX, DZ_OUTPUT_TENSOR_INDEX);
        if (successIndication != tpc_lib_api::GLUE_SUCCESS)
        {
            return successIndication;
        }
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn einsumShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    auto metadata = reinterpret_cast<SifEinsumMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1 || input->inputTensorsNr == 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    std::array<TSize, tpc_lib_api::MAX_TENSOR_DIM* 2> labels_to_dim_sizes = {0};

    // Recalculate label to dim size mappings using actual tensor sizes
    for (int tensor = 0; tensor < input->inputTensorsNr; tensor++)
    {
        for (int axis = 0; axis < input->inputTensors[tensor]->geometry.dims; axis++)
        {
            const TSize    size  = input->inputTensors[tensor]->geometry.maxSizes[axis];
            const unsigned label = metadata->input_dims_to_labels[tensor][axis];

            labels_to_dim_sizes[label] = (metadata->labels_to_types[label] == DimensionType::kBroadcasting)
                                             ? std::max(labels_to_dim_sizes[label], size)
                                             : size;
        }
    }

    for (unsigned i = 0; i < metadata->output_dims; i++)
    {
        output->outputTensors[0]->geometry.maxSizes[i] = labels_to_dim_sizes[metadata->output_dims_to_labels[i]];
    }
    output->outputTensors[0]->geometry.dims =
        metadata->output_dims == 0 ? (output->outputTensors[0]->geometry.maxSizes[0] = 1, 1) : metadata->output_dims;

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn dynamicReshapeShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    auto metadata = reinterpret_cast<SifDynamicReshapeMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_DIMS_EQUAL(input->inputTensors[0]->geometry.dims, metadata->input_dims);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    // Map einsum labels to split fused SIF labels
    std::unordered_map<char, std::string> input_labels_to_output_labels;
    for (char dim = 0, lbl = 'a'; dim < metadata->input_dims; dim++, lbl++)
    {
        char label = metadata->input_dims_to_labels[unsigned(dim)];
        input_labels_to_output_labels.emplace(std::make_pair(label, std::string({lbl, '0'})));
    }

    std::string output_eq(metadata->output_eq);
    for (size_t pos = 0; pos < output_eq.length(); pos++)
    {
        // convert the equation to one suitable for split fused kernel SIF
        char c = output_eq[pos];
        if (isalpha(c))
        {
            output_eq.replace(pos, 1, input_labels_to_output_labels[c]);
        }
    }

    const_cast<SifParams*>(input)->nodeParams.nodeParams = output_eq.data();
    return splitFusedKernelsShapeInferenceFunction(deviceId, input, output);
}


SifReturn splitFusedKernelsShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    auto metadata = reinterpret_cast<SifSplitFusedKernelMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr > 0);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr > 0);

    // map input dims to labels
    // labels are hardcoded: a0 - dim 0 of tensor 0, b0 - dim 1 of tensor 0, c1 - dim 2 of tensor 1, etc...
    // input_labels_to_size[0]['a'] = input->inputTensors[0]->geometry.maxSizes[0], ...
    int input_labels_to_size[input->inputTensorsNr][26];
#define LBL2IDX(c) ((c) - 'a')
    for (unsigned inp = 0; inp < input->inputTensorsNr; inp++)
    {
        for (unsigned dim = 0, lbl = 'a'; dim < input->inputTensors[inp]->geometry.dims; dim++, lbl++)
        {
            input_labels_to_size[inp][LBL2IDX(lbl)] = input->inputTensors[inp]->geometry.maxSizes[dim];
        }
    }

    // Calculate output size
    // ex: 2 rank 3 input tensors -> 2 rank 2 output tensors
    // [a0,b0,c0,a1,b1,c1->]a0*2,b0+b1;c0/2,b0+b1
    // (output_eq is the part after ->)
    char* output_eq = metadata->sif;
    unsigned length = strlen(output_eq);

    enum ops
    {
        NONE = 0,
        ADD  = '+',
        SUB  = '-',
        MUL  = '*',
        DIV  = '/',
        MAX  = '>',
        MIN  = '<',
        POW  = '^',
        CDIV = '\\'
    };
    auto doOperation = [](uint64_t val1, ops op, uint64_t val2) -> uint64_t {
        switch (op)
        {
            case NONE:
                return val2;
            case ADD:
                return val1 + val2;
            case SUB:
                return val1 - val2;
            case MUL:
                return val1 * val2;
            case DIV:
                return val1 / val2;
            case CDIV:
                return div_round_up(val1, val2);
            case MAX:
                return std::max(val1, val2);
            case MIN:
                return std::min(val1, val2);
            case POW:
                return std::pow(val1, val2);
            default:
                LOG_ERR(DYN_SHAPE, "Unsupported operation {}", op);
                return 0;
        }
    };
    auto getNumber = [&](size_t& pos) -> uint64_t {
        uint64_t number = 0;
        for (; pos < length; pos++)
        {
            char c = output_eq[pos];
            if (isdigit(c))
            {
                number = number * 10 + (c - '0');
            }
            else
            {
                break;
            }
        }
        pos--;
        return number;
    };

    unsigned dim      = 0;     // current dimension
    unsigned out      = 0;     // current output tensor
    TSize    size     = 0;     // dimension size
    ops      curr_op  = NONE;  // parsed operation
    uint64_t constant = 0;     // accumulator for numeric constants
    for (size_t pos = 0; pos < length; pos++)
    {
        char c = output_eq[pos];
        if (isalpha(c))        // got a label
        {
            // retrieve tensor id and get dim size
            uint64_t inp = getNumber(++pos);
            size = doOperation(size, curr_op, input_labels_to_size[inp][LBL2IDX(c)]);
        }
        else if (isdigit(c))   // got a constant
        {
            constant = getNumber(pos);
            size     = doOperation(size, curr_op, constant);
            constant = 0;
        }
        else if (c == ',')     // finish calculating a dim and move to the next
        {
            output->outputTensors[out]->geometry.maxSizes[dim++] = size;
            size = constant = 0;
            curr_op         = NONE;
        }
        else if (c == ';')     // finish calculating output tensor and move to the next
        {
            output->outputTensors[out]->geometry.maxSizes[dim++] = size;
            output->outputTensors[out++]->geometry.dims          = dim;
            size = constant = dim = 0;
            curr_op               = NONE;
        }
        else                   // got an operation
        {
            curr_op = (ops)c;
        }
    }
    output->outputTensors[out]->geometry.maxSizes[dim++] = size;  // last dim
    output->outputTensors[out]->geometry.dims            = dim;

#undef LBL2IDX
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn einsumExpandShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    auto metadata = reinterpret_cast<SifEinsumExpandShapeMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);

    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2 || input->inputTensorsNr == 3);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    tpc_lib_api::TensorShapeInfo** in  = input->inputTensors;
    tpc_lib_api::TensorShapeInfo*  out = output->outputTensors[0];
    unsigned                       dim = 0;
    // Inputs are batch/broadcast, input 0, input 1
    // Output shape is [free labels 1, free labels 0, batch/broadcast]
    if (input->inputTensorsNr > 2)
    {
        for (int i = 0; i < metadata->numOfFreeDimsInSecondInput; i++)
        {
            out->geometry.maxSizes[dim++] = in[2]->geometry.maxSizes[metadata->freeDimsInSecondInput[i]];
        }
    }
    for (int i = 0; i < metadata->numOfFreeDimsInFirstInput; i++)
    {
        out->geometry.maxSizes[dim++] = in[1]->geometry.maxSizes[metadata->freeDimsInFirstInput[i]];
    }
    for (int i = DIM_GEMM_BATCH; i < in[0]->geometry.dims; i++)
    {
        out->geometry.maxSizes[dim++] = in[0]->geometry.maxSizes[i];
    }
    out->geometry.dims = (dim == 0) ? (out->geometry.maxSizes[0] = 1, 1) : dim;

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn reinterpretCastShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    auto metadata = reinterpret_cast<SifReinterpretCastMetadata*>(input->nodeParams.nodeParams);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);

    if (metadata->outputElementSizeInBytes == 0 || metadata->inputElementSizeInBytes == 0)
    {
        LOG_ERR(DYN_SHAPE, "element size in SIF {} is zero", HLLOG_FUNC);
        return tpc_lib_api::GLUE_FAILED;
    }

    const auto* inputTensor         = input->inputTensors[0];
    auto*       outputTensor        = output->outputTensors[0];
    uint64_t    inputFcdSizeInBytes = (uint64_t)inputTensor->geometry.maxSizes[0] * metadata->inputElementSizeInBytes;
    // validate that FCD size * (size of input element) is multiplication of (size of output element)
    // if not (example input FCD size is 3, data type is bf16, and output is f32) we can't infer output FCD size
    if ((inputFcdSizeInBytes) % metadata->outputElementSizeInBytes != 0)
    {
        LOG_ERR(DYN_SHAPE,
                "incorrect fcd size in SIF {}: ({} * {}) % {} == {}, but it must be zero",
                HLLOG_FUNC,
                inputTensor->geometry.maxSizes[0],
                metadata->inputElementSizeInBytes,
                metadata->outputElementSizeInBytes,
                inputFcdSizeInBytes % metadata->outputElementSizeInBytes);
        return tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    // copy all the sizes from input to output
    memcpy(outputTensor->geometry.maxSizes, inputTensor->geometry.maxSizes, sizeof(tpc_lib_api::TensorGeometry::maxSizes));
    // fix the fcd size
    outputTensor->geometry.maxSizes[0] = inputFcdSizeInBytes / metadata->outputElementSizeInBytes;

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn inferMaxShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1 || input->outputTensorsNr == 2);

    auto metadata = reinterpret_cast<SifInferMaxShapeMetadata*>(input->nodeParams.nodeParams);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);

    const auto* inputTensor  = input->inputTensors[0];
    auto*       outputTensor = output->outputTensors[0];

    if (input->outputTensorsNr == 2)  // exists output shape tensor
    {
        auto* shapeTensor = output->outputTensors[1];
        VALIDATE_DIMS_EQUAL(shapeTensor->geometry.dims, inputTensor->geometry.dims);
        memcpy(shapeTensor->geometry.maxSizes, inputTensor->geometry.maxSizes, sizeof(tpc_lib_api::TensorGeometry::maxSizes));
    }

    // copy sizes from params to output, only if it activate
    for (unsigned dim = 0; dim < tpc_lib_api::MAX_TENSOR_DIM; ++dim)
    {
        outputTensor->geometry.maxSizes[dim] =
            metadata->params.shouldInferMax[dim] ? metadata->inputMaxSizes[dim] : inputTensor->geometry.maxSizes[dim];
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn tileShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    const auto* metadata = reinterpret_cast<SifTileShapeMetadata*>(input->nodeParams.nodeParams);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);

    const auto* inputTensor  = input->inputTensors[0];
    auto*       outputTensor = output->outputTensors[0];

    for (unsigned dim = 0; dim < tpc_lib_api::MAX_TENSOR_DIM; ++dim)
    {
        outputTensor->geometry.maxSizes[dim] = metadata->repeat[dim] * inputTensor->geometry.maxSizes[dim];
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn bnBatchSizeToH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    float* outData = reinterpret_cast<float*>(output->outputTensors[0]->hostAddress);
    NULL_CHECK(outData, "output data host address is null");

    outData[0] = 1.0;

    for (int i = 1; i < input->inputTensors[0]->geometry.dims; i++)
    {
        outData[0] *= input->inputTensors[0]->geometry.maxSizes[i];
    }

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn dynamicStridedDmaReinterpretH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifReinterpretH2DMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    synDynamicStridedDmaH2dTensor* inData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(input->inputTensors[0]->hostAddress);
    synDynamicStridedDmaH2dTensor* outData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(output->outputTensors[0]->hostAddress);

    NULL_CHECK(inData, "input data host address is null");
    NULL_CHECK(outData, "output data host address is null");
    VALIDATE_DIM_IN_RANGE(inData->num_strides, ARRAY_SIZE(inData->strides));
    unsigned factor     = *metadata;
    outData->strides[0] = 1;
    for (unsigned i = 1; i < inData->num_strides + 1; ++i)
    {
        outData->strides[i] = inData->strides[i - 1] * factor;
    }

    outData->num_strides = inData->num_strides + 1;
    outData->offset      = inData->offset * factor;
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn dynamicStridedDmaExpandH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifExpandH2DMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    synDynamicStridedDmaH2dTensor* inData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(input->inputTensors[0]->hostAddress);
    synDynamicStridedDmaH2dTensor* outData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(output->outputTensors[0]->hostAddress);

    NULL_CHECK(inData, "input data host address is null");
    NULL_CHECK(outData, "output data host address is null");

    unsigned dim = 0;
    for (; dim < metadata->axis; dim++)
    {
        outData->strides[dim] = inData->strides[dim];
    }
    outData->strides[metadata->axis] = (metadata->axis == 0) ? 1 : inData->strides[metadata->axis - 1];
    for (; dim < inData->num_strides; dim++)
    {
        outData->strides[dim + 1] = inData->strides[dim];
    }

    outData->num_strides = inData->num_strides + 1;
    outData->offset      = inData->offset;
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn stridedOpsConversionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 2);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    synDynamicStridedDmaH2dTensor* outData =
        reinterpret_cast<synDynamicStridedDmaH2dTensor*>(output->outputTensors[0]->hostAddress);

    NULL_CHECK(outData, "output data host address is null");

    outData->num_strides = input->inputTensors[0]->geometry.dims;        // first input is strides tensor
    outData->offset      = input->inputTensors[1]->geometry.maxSizes[0];    // second input is offset tensor
    std::copy(input->inputTensors[0]->geometry.maxSizes, input->inputTensors[0]->geometry.maxSizes + outData->num_strides, outData->strides);

    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn dynamicSliceDmaExpandH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifExpandH2DMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    synDynamicSliceDmaH2dTensor* inData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(input->inputTensors[0]->hostAddress);
    synDynamicSliceDmaH2dTensor* outData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(output->outputTensors[0]->hostAddress);

    NULL_CHECK(inData, "input data host address is null");
    NULL_CHECK(outData, "output data host address is null");

    unsigned dim = 0;
    for (; dim < metadata->axis; dim++)
    {
        outData->starts[dim] = inData->starts[dim];
        outData->steps[dim]  = inData->steps[dim];
    }
    outData->starts[metadata->axis] = (metadata->axis == 0) ? 0 : inData->starts[metadata->axis - 1];
    outData->steps[metadata->axis]  = (metadata->axis == 0) ? 1 : inData->steps[metadata->axis - 1];
    for (; dim < inData->dims; dim++)
    {
        outData->starts[dim + 1] = inData->starts[dim];
        outData->steps[dim + 1]  = inData->steps[dim];
    }

    outData->dims = inData->dims + 1;
    return tpc_lib_api::GLUE_SUCCESS;
}

SifReturn sliceConversionShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();

    VALIDATE_PARAMS(input, output);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 2);

    synDynamicSliceDmaH2dTensor* inData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(input->inputTensors[0]->hostAddress);
    NULL_CHECK(inData, "input data host address is null");

    tpc_lib_api::TensorShapeInfo* stepsTensor  = output->outputTensors[0];
    tpc_lib_api::TensorShapeInfo* startsTensor = output->outputTensors[1];

    stepsTensor->geometry.dims = inData->dims;
    startsTensor->geometry.dims = inData->dims;
    // first output is steps tensor, second output is starts tensor
    std::copy(inData->steps, inData->steps + inData->dims, stepsTensor->geometry.maxSizes);
    std::copy(inData->starts, inData->starts + inData->dims, startsTensor->geometry.maxSizes);

    return tpc_lib_api::GLUE_SUCCESS;
}


SifReturn transposeSliceH2DShapeInferenceFunction(tpc_lib_api::DeviceId deviceId, const SifParams* input, SifOutputs* output)
{
    STAT_FUNCTION();
    auto metadata = reinterpret_cast<SifTransposeMetadata*>(input->nodeParams.nodeParams);

    VALIDATE_PARAMS(input, output);
    VALIDATE_METADATA(metadata, input->nodeParams.nodeParamsSize);
    VALIDATE_INPUT_TENSORS_COUNT(input->inputTensorsNr == 1);
    VALIDATE_OUTPUT_TENSORS_COUNT(input->outputTensorsNr == 1);

    synDynamicSliceDmaH2dTensor* inData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(input->inputTensors[0]->hostAddress);
    synDynamicSliceDmaH2dTensor* outData =
        reinterpret_cast<synDynamicSliceDmaH2dTensor*>(output->outputTensors[0]->hostAddress);

    outData->dims = inData->dims;
    for (size_t d = 0; d < outData->dims; d++)
    {
        unsigned index     = metadata->permutation[d];
        outData->starts[d] = inData->starts[index];
        outData->steps[d]  = inData->steps[index];
    }

    return tpc_lib_api::GLUE_SUCCESS;
}
// TODO: [SW-23614] CD 0.12.0 - Compile U20 - Fail build_synapse
// compilation fails on with gcc 9.3, remove when fixed
#pragma GCC diagnostic pop
