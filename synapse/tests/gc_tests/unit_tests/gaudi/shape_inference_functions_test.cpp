#include "sif/shape_inference_metadata.h"
#include "smf/shape_func_registry.h"
#include "tensor_view_node.h"
#include "tpc_node.h"
#include "types.h"

#include <gtest/gtest.h>
#include <numeric>

struct MatrixDims
{
    unsigned w;
    unsigned h;
    unsigned b;
    unsigned c;
    unsigned e;
};

struct GemmParams
{
    MatrixDims op1;
    MatrixDims op2;

    size_t tensorDim;
};

unsigned getTensorShapeInfoElementsSize(const TensorShapeInfo& shape)
{
    return std::accumulate(shape.geometry.maxSizes, shape.geometry.maxSizes + shape.geometry.dims - 1, 1, std::multiplies<unsigned>());
}
class InferShapeFunctions : public ::testing::Test
{
public:
    InferShapeFunctions(){}
    virtual ~InferShapeFunctions(){}

    virtual void SetUp() override { ShapeFuncRegistry::instance().init(synDeviceType::synDeviceTypeInvalid); }
    virtual void TearDown() override { ShapeFuncRegistry::instance().destroy(); }

};

class DmaInferShapeFunctions : public InferShapeFunctions
{
public:
    DmaInferShapeFunctions(){}
    virtual ~DmaInferShapeFunctions(){}
};

class LogicalInferShapeFunctions : public InferShapeFunctions
{
public:
    LogicalInferShapeFunctions(){}
virtual ~LogicalInferShapeFunctions(){}
};

class MomentsInferShapeFunctions : public InferShapeFunctions
{
public:
    MomentsInferShapeFunctions() {}
    virtual ~MomentsInferShapeFunctions() {}
};

class NOPInferShapeFunctions : public InferShapeFunctions
{
public:
    NOPInferShapeFunctions() {}
    virtual ~NOPInferShapeFunctions() {}
};

class RotateInferShapeFunctions : public InferShapeFunctions
{
public:
    RotateInferShapeFunctions() {}
    virtual ~RotateInferShapeFunctions() {}
};

class CudBnFwdInferShapeFunctions : public InferShapeFunctions
{
public:
    CudBnFwdInferShapeFunctions() {}
    virtual ~CudBnFwdInferShapeFunctions() {}
};

class MmeInferShapeFunctions : public InferShapeFunctions
{
public:
    MmeInferShapeFunctions(){}
    virtual ~MmeInferShapeFunctions(){}
};

class TfBatchNormInferShapeFunctions : public InferShapeFunctions
{
public:
    TfBatchNormInferShapeFunctions() {}
    virtual ~TfBatchNormInferShapeFunctions() {}

    std::vector<TensorShapeInfo*> addInputs(TensorShapeInfo* inputTemplate, int numberOfCopies = -1)
    {
        if (numberOfCopies == -1) numberOfCopies = properInputsNum;
        std::vector<TensorShapeInfo*> inputs;
        for (int i = 0; i < numberOfCopies; i++)
        {
            inputs.push_back(inputTemplate);
        }
        return inputs;
    }

    std::vector<TensorShapeInfo*> addOutputs(TensorShapeInfo* outputTemplate, int numberOfCopies = -1)
    {
        if (numberOfCopies == -1) numberOfCopies = properOutputsNum;
        std::vector<TensorShapeInfo*> outputs;
        for (int i = 0; i < numberOfCopies; i++)
        {
            outputs.push_back(outputTemplate);
        }
        return outputs;
    }

    static const unsigned properInputsNum  = 5;
    static const unsigned properOutputsNum = 1;
};

class CudBnBwdExInferShapeFunctions : public InferShapeFunctions
{
public:
    CudBnBwdExInferShapeFunctions() {}
    virtual ~CudBnBwdExInferShapeFunctions() {}
};

class DynamicReshapeInferShapeFunctions : public InferShapeFunctions
{
public:
    DynamicReshapeInferShapeFunctions(){}
    virtual ~DynamicReshapeInferShapeFunctions(){}
};

class TpcInferShapeFunctions : public InferShapeFunctions
{
public:
    TpcInferShapeFunctions(){}
    virtual ~TpcInferShapeFunctions(){}
};

template<class T = unsigned>
unsigned inferShape(ShapeFuncID sifId,
                    std::vector<TensorShapeInfo*> inputs,
                    std::vector<TensorShapeInfo*> outputs,
                    unsigned metadataSize = 0,
                    T* sifMetadata = nullptr,
                    SifReturn expectedResult = tpc_lib_api::GLUE_SUCCESS)
{
    unsigned invalidMask = 0;
    SifParams input = {};
    input.apiVersion                = 2;
    input.inputTensors              = inputs.data();
    input.inputTensorsNr            = inputs.size();
    input.nodeParams.nodeParams     = sifMetadata;
    input.nodeParams.nodeParamsSize = metadataSize;
    input.outputTensorsNr           = outputs.size();
    input.maxAvailableTpc           = TPCNode::getMaxAvailableTpc(tpc_lib_api::DEVICE_ID_GAUDI);
    SifOutputs output = {outputs.data(), &invalidMask};
    sif_t sif = ShapeFuncRegistry::instance().getSIF(sifId);
    if(sif == nullptr)
    {
        throw std::runtime_error("SIF ID: " + std::to_string(sifId) + " is not registered");
    }
    auto sts = sif(tpc_lib_api::DEVICE_ID_GAUDI,&input, &output);
    EXPECT_EQ(expectedResult, sts);
    return invalidMask;
}

std::vector<TensorShapeInfo*> takeAddresses(std::vector<TensorShapeInfo>& in)
{
    std::vector<TensorShapeInfo*> ret(in.size());
    for (auto i = 0U; i < in.size(); ++i) ret[i] = &in[i];
    return ret;
}

// variant for array of structs instead of array of pointers
template <typename T = unsigned>
unsigned inferShape(ShapeFuncID sifId,
                    std::vector<TensorShapeInfo>& inputs,
                    std::vector<TensorShapeInfo>& outputs,
                    unsigned metadataSize = 0,
                    T* sifMetadata = nullptr,
                    SifReturn expectedResult = tpc_lib_api::GLUE_SUCCESS)
{
    return inferShape<T>(sifId, takeAddresses(inputs), takeAddresses(outputs), metadataSize, sifMetadata, expectedResult);
}

void identityCheck(ShapeFuncID sifId)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    auto rnd = [](){ return rand() % 100; };

    TensorShapeInfo inputTensorShapeInfo = {{4, {rnd(), rnd(), rnd(), rnd(), 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{4, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(sifId, inputs, outputs);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();

    EXPECT_EQ(inputTensorShapeInfo.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        unsigned expectedDimSize = inputTensorShapeInfo.geometry.maxSizes[d];
        EXPECT_EQ(expectedDimSize, output->geometry.maxSizes[d]) << "wrong output shape of identity op";
    }
}

void transposeTest(ShapeFuncID sifId)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;
    auto rnd = [](){ return rand() % 100; };

    TensorShapeInfo inputTensorShapeInfo = {{4, {rnd(), rnd(), rnd(), rnd(), 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{4, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifTransposeMetadata metadata;
    unsigned sizes[] = {3, 2, 1, 0, 0};
    memcpy(&metadata.permutation, &sizes, sizeof(sizes));

    unsigned invalidMask = inferShape(sifId, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        unsigned index = metadata.permutation[d];
        unsigned expectedDimSize = inputTensorShapeInfo.geometry.maxSizes[index];

        EXPECT_EQ(expectedDimSize, output->geometry.maxSizes[d]) << "wrong output shape of transpose op";
    }
}

TensorShapeInfo getGemmExpectedOutput(const SifGemmMetadata& metadata, const GemmParams& params, bool transposeOutput)
{
    unsigned        op1ExpectedSize = metadata.params.transpose_a ? params.op1.w : params.op1.h;
    unsigned        op2ExpectedSize = metadata.params.transpose_b ? params.op2.h : params.op2.w;
    TensorShapeInfo expectedOutput  = {{params.tensorDim, {transposeOutput ? op1ExpectedSize : op2ExpectedSize,
                                       transposeOutput ? op2ExpectedSize : op1ExpectedSize,
                                       std::max(params.op2.b, params.op1.b),
                                       std::max(params.op2.c, params.op1.c),
                                       std::max(params.op2.e, params.op1.e)}}};
    return expectedOutput;
}

void baseGemmTest(ShapeFuncID     sifId,
                  SifGemmMetadata metadata,
                  GemmParams      params,
                  bool            transposeOutput,
                  int             biasSize,
                  SifReturn       expectedResult)
{
    TensorShapeInfo tensor1 = {{params.tensorDim, {params.op1.w, params.op1.h, params.op1.b, params.op1.c, params.op1.e}}};
    TensorShapeInfo tensor2 = {{params.tensorDim, {params.op2.w, params.op2.h, params.op2.b, params.op2.c, params.op2.e}}};
    TensorShapeInfo tensorBias;
    TensorShapeInfo expectedOutput = getGemmExpectedOutput(metadata, params, transposeOutput);

    TensorShapeInfo outputTensorShapeInfo = {{params.tensorDim,{}}};
    std::vector<TensorShapeInfo*> inputs  = {&tensor1, &tensor2};
    std::vector<TensorShapeInfo*> outputs = {&outputTensorShapeInfo};

    if (biasSize != -1)
    {
        tensorBias = {{1,{biasSize}}};
        inputs.push_back(&tensorBias);
    }

    unsigned invalidMask = inferShape(sifId, inputs, outputs, sizeof(metadata), &metadata, expectedResult);

    if (expectedResult == tpc_lib_api::GLUE_SUCCESS)
    {
        EXPECT_EQ(0, invalidMask);

        TensorShapeInfo* output = outputs.front();
        EXPECT_EQ(params.tensorDim, output->geometry.dims);

        for (size_t d = 0; d < params.tensorDim; ++d)
        {
            EXPECT_EQ(expectedOutput.geometry.maxSizes[d], output->geometry.maxSizes[d]) << "wrong output shape of gemm op";
        }
    }
}

void gemmTest(ShapeFuncID     sifId,
              SifGemmMetadata metadata,
              GemmParams      params,
              bool            transposeOutput = false,
              int             biasSize        = -1,
              SifReturn       expectedResult  = tpc_lib_api::GLUE_SUCCESS)
{
    params.tensorDim = 2;
    baseGemmTest(sifId, metadata, params, transposeOutput, biasSize, expectedResult);
}

void batchGemmTest(ShapeFuncID     sifId,
                   SifGemmMetadata metadata,
                   GemmParams      params,
                   bool            transposeOutput = false,
                   int             biasSize        = -1,
                   SifReturn       expectedResult  = tpc_lib_api::GLUE_SUCCESS)
{
    baseGemmTest(sifId, metadata, params, transposeOutput, biasSize, expectedResult);
}

void negativeTestSingleInputSingleOutputNoMetadata(ShapeFuncID sifId)
{
    auto rnd = [](){ return rand() % 100; };
    TensorShapeInfo tsi = {{4, {rnd(), rnd(), rnd(), rnd(), 0}}};

    inferShape<unsigned>(sifId, {&tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi, nullptr}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape<unsigned>(sifId, {}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape<unsigned>(sifId, {&tsi}, {}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    inferShape<unsigned>(sifId, {&tsi}, {&tsi, &tsi}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

void negativeTestMultiInputSingleOutputNoMetadata(ShapeFuncID sifId)
{
    auto rnd = [](){ return rand() % 100; };
    TensorShapeInfo tsi = {{4,{rnd(), rnd(), rnd(), rnd(), 0}}};

    inferShape<unsigned>(sifId, {&tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi, nullptr}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape<unsigned>(sifId, {}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape<unsigned>(sifId, {&tsi, &tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi}, {}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    inferShape<unsigned>(sifId, {&tsi}, {&tsi, &tsi}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

template<class T>
void negativeTestSingleInputMultiOutputWithMetadata(ShapeFuncID sifId, unsigned metadataSize , T* metadata)
{
    auto rnd = [](){ return rand() % 100; };
    TensorShapeInfo tsi = {.geometry={.dims= 4, .maxSizes={rnd(), rnd(), rnd(), rnd(), 0}}};

    inferShape(sifId, {&tsi}, {&tsi}, metadataSize, metadata, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi}, {&tsi}, 0, nullptr,tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE);
    inferShape(sifId, {&tsi, nullptr}, {&tsi}, metadataSize, metadata, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape(sifId, {}, {&tsi}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi, &tsi}, {&tsi}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi}, {}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

template<class T>
void negativeTestSingleInputSingleOutputWithMetadata(ShapeFuncID sifId, unsigned metadataSize , T* metadata,
                                                     unsigned outDim = 4)
{
    auto rnd = [](){ return rand() % 100; };
    TensorShapeInfo tsi = {.geometry={.dims= 4, .maxSizes={rnd(), rnd(), rnd(), rnd(), 0}}};
    TensorShapeInfo outTsi = {{outDim,{}}};

    inferShape(sifId, {&tsi}, {&outTsi}, metadataSize, metadata, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi}, {&outTsi}, 0, nullptr, tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE);
    inferShape(sifId, {&tsi, nullptr}, {&outTsi}, metadataSize, metadata, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape(sifId, {}, {&outTsi}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi, &tsi, &tsi}, {&outTsi}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi}, {}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    inferShape(sifId, {&tsi}, {&outTsi, &outTsi}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

template<class T>
void negativeTestMultiInputSingleOutputWithMetadata(ShapeFuncID sifId, unsigned metadataSize , T* metadata)
{
    auto rnd = [](){ return rand() % 100; };
    TensorShapeInfo tsi = {.geometry={.dims= 4, .maxSizes={rnd(), rnd(), rnd(), rnd(), 0}}};

    inferShape(sifId, {&tsi}, {&tsi}, metadataSize, metadata, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi}, {&tsi}, 0, nullptr,tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE);
    inferShape(sifId, {&tsi, nullptr}, {&tsi}, metadataSize, metadata,tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape(sifId, {}, {&tsi}, metadataSize, metadata,tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi, &tsi}, {&tsi}, metadataSize, metadata, tpc_lib_api::GLUE_SUCCESS);
    inferShape(sifId, {&tsi}, {}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
    inferShape(sifId, {&tsi}, {&tsi, &tsi}, metadataSize, metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

void negativeConv(ShapeFuncID sifId)
{
    SifConvolutionMetadata metadata = { };
    auto rnd = [](){ return rand() % 100; };
    TensorShapeInfo tsi = {.geometry={.dims= 4, .maxSizes={rnd(), rnd(), rnd(), rnd(), 0}}};

    unsigned metadataSize = sizeof(metadata);
    inferShape(sifId, {&tsi, &tsi}, {&tsi}, metadataSize, &metadata, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi, &tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE);
    inferShape(sifId, {&tsi, nullptr}, {&tsi}, metadataSize, &metadata, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape(sifId, {}, {&tsi}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi}, {&tsi}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi, &tsi}, {}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

void testShapeTensorSifWithShapeTensor(ShapeFuncID sifId)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{3, {2, 1, 3, 0, 0}}};
    TensorShapeInfo shapeTensor = {{2, {2, 4, 3, 0, 0}}};
    if (sifId == SIF_RESHAPE) // Must use same total number of elements now!
    {
         shapeTensor = {{2,{2, 3, 0, 0, 0}}};
    }
    inputs.push_back(&inputTensorShapeInfo);
    inputs.push_back(&shapeTensor);

    TensorShapeInfo outputTensorShapeInfo = {{2, {},{}}};
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(sifId, inputs, outputs);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(shapeTensor.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < shapeTensor.geometry.dims; ++d)
    {
        unsigned expectedDimSize = shapeTensor.geometry.maxSizes[d];
        EXPECT_EQ(expectedDimSize, output->geometry.maxSizes[d]) << "wrong output shape of shape tensor op";
    }
}

void testShapeTensorSifWithoutShapeTensor(ShapeFuncID sifId)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{3,{2, 1, 3, 0, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = { };
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(sifId, inputs, outputs);
    EXPECT_EQ(0b1, invalidMask);
}

void testShapeTensorSifWithoutShapeTensorNegative(ShapeFuncID sifId)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{3,{2, 1, 3, 0, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = { };
    outputs.push_back(&outputTensorShapeInfo);

    inferShape<unsigned>(sifId, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

void testShapeTensorSifNegative(ShapeFuncID sifId)
{
    TensorShapeInfo tsi = {{4,{1, 2, 3, 4, 0}}};

    inferShape<unsigned>(sifId, {&tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi}, {nullptr}, 0, nullptr, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape<unsigned>(sifId, {}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape<unsigned>(sifId, {&tsi, &tsi}, {}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

TEST_F(DmaInferShapeFunctions, infer_dma_memcpy_op_shape)
{
    identityCheck(SIF_DMA_MEMCPY);
}

TEST_F(DmaInferShapeFunctions, infer_dma_memcpy_op_shape_negative)
{
    negativeTestSingleInputSingleOutputNoMetadata(SIF_DMA_MEMCPY);
}

TEST_F(DmaInferShapeFunctions, infer_dma_memset_op_shape_with_shape_tensor)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo shapeTensor = {{4, {1, 2, 3, 4, 0}}};
    inputs.push_back(&shapeTensor);

    TensorShapeInfo outputTensorShapeInfo = {{4,{}}};
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(SIF_DMA_MEMSET, inputs, outputs);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(shapeTensor.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < shapeTensor.geometry.dims; ++d)
    {
        unsigned expectedDimSize = shapeTensor.geometry.maxSizes[d];
        EXPECT_EQ(expectedDimSize, output->geometry.maxSizes[d]) << "wrong output shape of memset op";
    }
}

TEST_F(DmaInferShapeFunctions, infer_dma_memset_op_shape_without_shape_tensor)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo outputTensorShapeInfo = { };
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(SIF_DMA_MEMSET, inputs, outputs);
    EXPECT_EQ(0b1, invalidMask);
}

TEST_F(DmaInferShapeFunctions, infer_dma_memset_op_shape_negative)
{
    TensorShapeInfo tsi = {{4, {1, 2, 3, 4, 0}}};

    ShapeFuncID sifId = SIF_DMA_MEMSET;
    inferShape<unsigned>(sifId, {&tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi}, {nullptr}, 0, nullptr, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape<unsigned>(sifId, {&tsi, &tsi}, {&tsi}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape<unsigned>(sifId, {&tsi}, {}, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

TEST_F(DmaInferShapeFunctions, infer_dma_transpose_op_shape)
{
    transposeTest(SIF_TRANSPOSE);
}

TEST_F(DmaInferShapeFunctions, infer_dma_transpose_op_shape_negative)
{
    SifTransposeMetadata metadata;
    unsigned sizes[] = {3, 2, 1, 0, 0};
    memcpy(&metadata.permutation, &sizes, sizeof(sizes));
    negativeTestSingleInputSingleOutputWithMetadata(SIF_TRANSPOSE, sizeof(metadata), &metadata);
}

TEST_F(LogicalInferShapeFunctions, infer_concatenate_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    unsigned concatenateDimIndex = 0;

    TensorShapeInfo inputTensorShapeInfo1 = {{2, {2, 2, 0, 0, 0}}};
    TensorShapeInfo inputTensorShapeInfo2 = {{2, {3, 2, 0, 0, 0}}};
    TensorShapeInfo inputTensorShapeInfo3 = {{2, {3, 2, 0, 0, 0}}};

    inputs.push_back(&inputTensorShapeInfo1);
    inputs.push_back(&inputTensorShapeInfo2);
    inputs.push_back(&inputTensorShapeInfo3);

    TensorShapeInfo outputTensorShapeInfo = {{2, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifConcatenateMetadata metadata = {};
    metadata.axis = concatenateDimIndex;

    unsigned invalidMask = inferShape(SIF_CONCATENATE, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    auto dimSum = 0;
    for(auto& t : inputs)
    {
        dimSum += t->geometry.maxSizes[metadata.axis];
    }

    TensorShapeInfo* input = inputs.front();
    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo1.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < input->geometry.dims; ++d)
    {
        unsigned inputDimSize = input->geometry.maxSizes[d];
        unsigned expectedDimSize = d == metadata.axis ? dimSum : inputDimSize;
        EXPECT_EQ(expectedDimSize, output->geometry.maxSizes[d]) << "wrong output shape of concatenate op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_concatenate_op_shape_negative)
{
    SifConcatenateMetadata metadata;
    metadata.axis = 0;
    negativeTestMultiInputSingleOutputWithMetadata(SIF_CONCATENATE, sizeof(metadata), &metadata);
}

TEST_F(LogicalInferShapeFunctions, infer_flatten_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{4, {1, 2, 3, 4, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{2,{}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifFlattenMetadata metadata;
    metadata.axis = 1;

    unsigned invalidMask = inferShape(SIF_FLATTEN, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();

    std::vector<unsigned> expectedDims = {1, 1};

    for(size_t d = 0; d < inputTensorShapeInfo.geometry.dims; ++d)
    {
        if(d <= metadata.axis)
        {
            expectedDims[0] *= inputTensorShapeInfo.geometry.maxSizes[d];
        }
        else
        {
            expectedDims[1] *= inputTensorShapeInfo.geometry.maxSizes[d];
        }
    }

    for(size_t d = 0; d < outputTensorShapeInfo.geometry.dims; ++d)
    {
        EXPECT_EQ(expectedDims[d], output->geometry.maxSizes[d]) << "wrong output shape of flatten op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_flatten_op_shape_negative)
{
    SifFlattenMetadata metadata;
    metadata.axis = 1;
    negativeTestSingleInputSingleOutputWithMetadata(SIF_FLATTEN, sizeof(metadata), &metadata);
}

TEST_F(LogicalInferShapeFunctions, infer_split_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    unsigned splitDimIndex = 2;

    TensorShapeInfo inputTensorShapeInfo = {{3, {1, 2, 4, 0, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    std::vector<unsigned> outputSplitDimSizes = {3, 2, 1};

    std::vector<TensorShapeInfo> expectedOutput;
    expectedOutput.push_back({{3, {1, 2, 3, 0, 0}}});
    expectedOutput.push_back({{3, {1, 2, 1, 0, 0}}});
    expectedOutput.push_back({{3, {1, 2, 0, 0, 0}}});

    std::vector<TensorShapeInfo> buffer(outputSplitDimSizes.size());
    for(auto& t : buffer)
    {
        t.geometry.dims = 3;
        outputs.push_back(&t);
    }

    unsigned splitSize = outputSplitDimSizes.size();
    unsigned headerSize = sizeof(SifSplitHeader);
    unsigned dataSize = splitSize * sizeof(TSize);
    unsigned bufferSize = headerSize + dataSize;

    std::vector<uint8_t> metadataBuffer(bufferSize);
    auto   metadata      = reinterpret_cast<SifSplitMetadata*>(metadataBuffer.data());
    TSize* splitDimSizes = metadata->splitDimSizes;

    metadata->header = {splitDimIndex, splitSize};

    for(size_t i = 0; i < outputSplitDimSizes.size(); ++i)
    {
        splitDimSizes[i] = outputSplitDimSizes[i];
    }

    unsigned invalidMask = inferShape(SIF_SPLIT, inputs, outputs, bufferSize, metadata);
    EXPECT_EQ(0, invalidMask);

    for(size_t t = 0; t < outputs.size(); ++t)
    {
        EXPECT_EQ(inputTensorShapeInfo.geometry.dims, outputs[t]->geometry.dims);
        for(size_t d = 0; d < inputTensorShapeInfo.geometry.dims; ++d)
        {
            EXPECT_EQ(expectedOutput[t].geometry.maxSizes[d], outputs[t]->geometry.maxSizes[d]) << "wrong output shape of split op";
        }
    }
}

TEST_F(LogicalInferShapeFunctions, infer_split_op_shape_negative)
{
    unsigned splitDimIndex = 2;
    std::vector<unsigned> outputSplitDimSizes = {3, 2, 1};
    unsigned splitSize = outputSplitDimSizes.size();
    unsigned headerSize = sizeof(SifSplitHeader);
    unsigned dataSize = splitSize * sizeof(TSize);
    unsigned bufferSize = headerSize + dataSize;

    std::vector<uint8_t> metadataBuffer(bufferSize);
    auto                 metadata      = reinterpret_cast<SifSplitMetadata*>(metadataBuffer.data());
    TSize* splitDimSizes = metadata->splitDimSizes;

    metadata->header = {splitDimIndex, splitSize};

    for(size_t i = 0; i < outputSplitDimSizes.size(); ++i)
    {
        splitDimSizes[i] = outputSplitDimSizes[i];
    }

    negativeTestSingleInputMultiOutputWithMetadata(SIF_SPLIT, bufferSize, metadata);
}

TEST_F(LogicalInferShapeFunctions, infer_expand_dims_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{4, {1, 2, 3, 4, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{5, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifExpandDimsMetadata metadata;
    metadata.axis = 3;

    unsigned invalidMask = inferShape(SIF_EXPAND_DIMS, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo.geometry.dims + 1, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        unsigned offset = d < metadata.axis ? 0 : 1;
        unsigned expectedDimSize = d == metadata.axis ? 1 : inputTensorShapeInfo.geometry.maxSizes[d - offset];
        EXPECT_EQ(expectedDimSize, output->geometry.maxSizes[d]) << "wrong output shape of expand dims op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_expand_dims_op_shape_negative)
{
    SifExpandDimsMetadata metadata;
    metadata.axis = 3;
    negativeTestSingleInputSingleOutputWithMetadata(SIF_EXPAND_DIMS, sizeof(metadata), &metadata, 5);
}

TEST_F(LogicalInferShapeFunctions, infer_slice_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{4, {4, 2, 2, 4, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{4,{}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifSliceMetadata metadata = {.starts = {0, 0, 0, 0, 0}, .ends = {4, 2, 2, 4, 0}, .steps = {2, 2, 1, 1, 0}};

    unsigned invalidMask = inferShape(SIF_SLICE, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    unsigned expectedOutput[] = {2, 1, 2, 4};
    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        EXPECT_EQ(expectedOutput[d], output->geometry.maxSizes[d]) << "wrong output shape of slice common op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_slice_op_shape_negative)
{
    SifSliceMetadata metadata = {.starts = {0, 0, 0, 0, 0}, .ends = {4, 2, 2, 4, 0}, .steps = {2, 2, 1, 1, 0}};

    negativeTestMultiInputSingleOutputWithMetadata(SIF_SLICE, sizeof(metadata), &metadata);
}

TEST_F(LogicalInferShapeFunctions, infer_slice_axis_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{2, {3, 4, 0, 0, 0}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{2, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifSliceAxisMetadata metadata;
    metadata.axis = 1;
    metadata.begin = 0;
    metadata.end = 2;

    unsigned invalidMask = inferShape(SIF_SLICE_AXIS, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    unsigned expectedOutput[] = {3, 2};
    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        EXPECT_EQ(expectedOutput[d], output->geometry.maxSizes[d]) << "wrong output shape of slice axis op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_slice_axis_op_shape_negative)
{
    SifSliceAxisMetadata metadata;
    metadata.axis = 1;
    metadata.begin = 0;
    metadata.end = 2;

    negativeTestSingleInputSingleOutputWithMetadata(SIF_SLICE_AXIS, sizeof(metadata), &metadata);
}

TEST_F(LogicalInferShapeFunctions, infer_slice_backward_op_shape_with_shape_tensor)
{
    testShapeTensorSifWithShapeTensor(SIF_SLICE_BACKWARD);
}

TEST_F(LogicalInferShapeFunctions, infer_slice_backward_op_shape_without_shape_tensor_negative)
{
    testShapeTensorSifWithoutShapeTensorNegative(SIF_SLICE_BACKWARD);
}

TEST_F(DmaInferShapeFunctions, infer_reshape_op_shape_with_shape_tensor)
{
    testShapeTensorSifWithShapeTensor(SIF_RESHAPE);
}

TEST_F(DmaInferShapeFunctions, infer_reshape_op_shape_without_shape_tensor)
{
    testShapeTensorSifWithoutShapeTensor(SIF_RESHAPE);
}

TEST_F(DmaInferShapeFunctions, infer_reshape_op_shape_negative)
{
    testShapeTensorSifNegative(SIF_RESHAPE);
}

TEST_F(LogicalInferShapeFunctions, infer_packing_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{4, {2, 4, 8, 10, 0}}}; // indices 2 and 3 are dynamic
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{3, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    synStaticReshapeSifParams metadata = {{2, 4, 10, 12},  // inputTensorShapeInfoMax
                                          {8, 10, 12},     // packingTensorShapeInfoMax
                                          {1, 1, 0, 0},
                                          {1, 0, 0},
                                          0,
                                          3};  // indices 1 and 2 are dynamic

    unsigned invalidMask = inferShape(SIF_STATIC_RESHAPE, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(3, output->geometry.dims);

    // output shape should be with the same dims as the output bucket
    // the static sizes should be equal to the output bucket
    // the dynamic sizes should be eqaul to the actual input sizes
    TensorShapeInfo expectedShapeInfo = {{3, {8, 8, 10, 0, 0}}};

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        EXPECT_EQ(expectedShapeInfo.geometry.maxSizes[d], output->geometry.maxSizes[d]) << "wrong output shape of packing op";
    }
}

TEST_F(DmaInferShapeFunctions, infer_broadcast_op_shape_with_shape_tensor)
{
    testShapeTensorSifWithShapeTensor(SIF_BROADCAST);
}

TEST_F(DmaInferShapeFunctions, infer_broadcast_op_shape_without_shape_tensor)
{
    testShapeTensorSifWithoutShapeTensor(SIF_BROADCAST);
}

TEST_F(DmaInferShapeFunctions, infer_broadcast_op_shape_negative)
{
    testShapeTensorSifNegative(SIF_BROADCAST);
}

TEST_F(LogicalInferShapeFunctions, infer_identity_op_shape)
{
    identityCheck(SIF_IDENTITY);
}

TEST_F(DmaInferShapeFunctions, infer_identity_op_shape_negative)
{
    negativeTestSingleInputSingleOutputNoMetadata(SIF_IDENTITY);
}

TEST_F(LogicalInferShapeFunctions, infer_reduction_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo1 = {{4, {1, 2, 3, 4, 0}}};
    TensorShapeInfo inputTensorShapeInfo2 = {{4, {1, 2, 3, 4, 0}}};
    TensorShapeInfo inputTensorShapeInfo3 = {{4, {1, 2, 3, 4, 0}}};

    inputs.push_back(&inputTensorShapeInfo1);
    inputs.push_back(&inputTensorShapeInfo2);
    inputs.push_back(&inputTensorShapeInfo3);

    TensorShapeInfo outputTensorShapeInfo = {{4, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(SIF_REDUCTION, inputs, outputs);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* input = inputs.front();
    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo1.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        EXPECT_EQ(input->geometry.maxSizes[d], output->geometry.maxSizes[d]) << "wrong output shape of reduction op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_reduction_op_shape_negative)
{
    negativeTestMultiInputSingleOutputNoMetadata(SIF_REDUCTION);
}

TEST_F(LogicalInferShapeFunctions, tensor_view_metadata)
{
    bool accessInput = true;
    const unsigned inTensorDim = 2;
    const TSize    inSizes[inTensorDim] = {4 ,4};
    const unsigned outTensorDim = 2;
    const TSize    outSizes[outTensorDim] = {2, 2};

    pTensor inTensor = std::make_shared<Tensor>(inTensorDim, inSizes, syn_type_single);
    TensorVector outTensors(3);
    for(auto& t : outTensors)
    {
        t = std::make_shared<Tensor>(outTensorDim, outSizes, syn_type_single);
    }

    std::shared_ptr<TensorViewNode> tvn(std::make_shared<TensorViewNode>(inTensor, accessInput, "inTensor"));

    std::vector<SizeVector> elementOffsets;
    elementOffsets.push_back({1, 1});
    elementOffsets.push_back({3, 1});
    elementOffsets.push_back({3, 3});

    for(size_t t = 0; t < outTensors.size(); ++t)
    {
        tvn->addView(outTensors[t], elementOffsets[t]);
    }
    auto metadata = reinterpret_cast<SifTensorViewMetadata*>(tvn->getShapeInferenceFunctionUserParams());
    size_t metadataSize = tvn->getShapeInferenceFunctionUserParamsSize();
    ASSERT_NE(nullptr, metadata);
    size_t expectedSize = sizeof(SifTensorViewHeader) + sizeof(SifTensorViewData) * metadata->header.viewsNr;
    EXPECT_EQ(expectedSize, metadataSize);
    EXPECT_EQ(outTensors.size(), metadata->header.viewsNr);
    EXPECT_EQ(accessInput, metadata->header.accessInput);
    for(size_t t = 0; t < outTensors.size(); ++t)
    {
        auto tvd = metadata->data[t];
        auto expectedOffsets = elementOffsets[t];
        auto expectedSizes = outSizes;

        EXPECT_EQ(outTensors[t]->getDim(), tvd.dims);

        for(size_t d = 0; d < tvd.dims; ++d)
        {
            EXPECT_EQ(expectedOffsets[d], tvd.offsets[d]);
            EXPECT_EQ(expectedSizes[d], tvd.sizes[d]);
        }
    }
}

TEST_F(LogicalInferShapeFunctions, infer_tensor_view_op_shape_access_input)
{
    const unsigned inTensorDim = 2;
    const TSize    inSizes[inTensorDim] = {4 ,4};
    pTensor inTensor = std::make_shared<Tensor>(inTensorDim, inSizes, syn_type_single);
    const unsigned outTensorDim = 2;
    const TSize    outSizes[outTensorDim] = {2, 2};

    TensorVector outTensors(3);
    for(auto& t : outTensors)
    {
        t = std::make_shared<Tensor>(outTensorDim, outSizes, syn_type_single);
    }

    std::shared_ptr<TensorViewNode> tvn(std::make_shared<TensorViewNode>(inTensor, true, "inTensor"));

    std::vector<SizeVector> elementOffsets;
    elementOffsets.push_back({0, 0});
    elementOffsets.push_back({1, 1});
    elementOffsets.push_back({2, 2});

    std::vector<SizeVector> expectedOutput;
    expectedOutput.push_back({2, 2});
    expectedOutput.push_back({2, 2});
    expectedOutput.push_back({2, 2});

    for(size_t t = 0; t < outTensors.size(); ++t)
    {
        tvn->addView(outTensors[t], elementOffsets[t]);
    }

    SifNodeParams metadata = tvn->getShapeInferenceFunctionUserParams();
    size_t metadataSize = tvn->getShapeInferenceFunctionUserParamsSize();

    TensorShapeInfo input;
    TensorShapeInfo outputs[3] = {};

    input.geometry.dims = inTensor->getDim();
    for(size_t t = 0; t < outTensors.size(); ++t)
    {
        outputs[t].geometry.dims = input.geometry.dims;
    }

    TSize sizes[tpc_lib_api::MAX_TENSOR_DIM];
    inTensor->getAllSizesInElements(sizes, tpc_lib_api::MAX_TENSOR_DIM);
    memcpy(input.geometry.maxSizes, sizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));

    std::vector<TensorShapeInfo*> inputs = {&input};
    std::vector<TensorShapeInfo*> outputsPtr = {&outputs[0], &outputs[1], &outputs[2]};

    unsigned invalidMask = inferShape(SIF_TENSOR_VIEW, inputs, outputsPtr, metadataSize, metadata);
    EXPECT_EQ(0, invalidMask);

    for(size_t t = 0; t < outputsPtr.size(); ++t)
    {
        EXPECT_EQ(input.geometry.dims, outputsPtr[t]->geometry.dims);
        for(size_t d = 0; d < input.geometry.dims; ++d)
        {
            EXPECT_EQ(expectedOutput[t][d], outputsPtr[t]->geometry.maxSizes[d]) << "wrong output shape of tensor view op";
        }
    }
}

TEST_F(LogicalInferShapeFunctions, infer_tensor_view_op_shape_access_input_cropped_output)
{
    const unsigned inTensorDim = 2;
    const TSize    inSizes[inTensorDim] = {4 ,4};
    pTensor inTensor = std::make_shared<Tensor>(inTensorDim, inSizes, syn_type_single);
    const unsigned outTensorDim = 2;
    const TSize    outSizes[outTensorDim] = {2, 2};

    TensorVector outTensors(3);
    for(auto& t : outTensors)
    {
        t = std::make_shared<Tensor>(outTensorDim, outSizes, syn_type_single);
    }

    std::shared_ptr<TensorViewNode> tvn(std::make_shared<TensorViewNode>(inTensor, true, "inTensor"));

    std::vector<SizeVector> elementOffsets;
    elementOffsets.push_back({1, 1});
    elementOffsets.push_back({3, 1});
    elementOffsets.push_back({3, 3});

    std::vector<std::vector<unsigned>> expectedOutput;
    expectedOutput.push_back({2, 2});
    expectedOutput.push_back({1, 2});
    expectedOutput.push_back({1, 1});

    for(size_t t = 0; t < outTensors.size(); ++t)
    {
        tvn->addView(outTensors[t], elementOffsets[t]);
    }

    SifNodeParams metadata = tvn->getShapeInferenceFunctionUserParams();
    size_t metadataSize = tvn->getShapeInferenceFunctionUserParamsSize();

    TensorShapeInfo input;
    TensorShapeInfo outputs[3] = {};

    input.geometry.dims = inTensor->getDim();
    for(size_t t = 0; t < outTensors.size(); ++t)
    {
        outputs[t].geometry.dims = input.geometry.dims;
    }

    TSize sizes[tpc_lib_api::MAX_TENSOR_DIM];
    inTensor->getAllSizesInElements(sizes, tpc_lib_api::MAX_TENSOR_DIM);
    memcpy(input.geometry.maxSizes, sizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));

    std::vector<TensorShapeInfo*> inputs = {&input};
    std::vector<TensorShapeInfo*> outputsPtr = {&outputs[0], &outputs[1], &outputs[2]};

    unsigned invalidMask = inferShape(SIF_TENSOR_VIEW, inputs, outputsPtr, metadataSize, metadata);
    EXPECT_EQ(0, invalidMask);

    for(size_t t = 0; t < outputsPtr.size(); ++t)
    {
        EXPECT_EQ(input.geometry.dims, outputsPtr[t]->geometry.dims);
        for(size_t d = 0; d < input.geometry.dims; ++d)
        {
            EXPECT_EQ(expectedOutput[t][d], outputsPtr[t]->geometry.maxSizes[d]) << "wrong output shape of tensor view op";
        }
    }
}

TEST_F(LogicalInferShapeFunctions, infer_tensor_view_op_shape_access_output)
{
    const unsigned outTensorDim = 2;
    const TSize    outSizes[outTensorDim] = {4 ,4};
    pTensor outTensor = std::make_shared<Tensor>(outTensorDim, outSizes, syn_type_single);


    TensorShapeInfo inputs[3] = {};
    const unsigned inTensorDim = 2;
    const TSize    inSizes[outTensorDim] = {2, 2};

    TensorVector inTensors(3);

    std::shared_ptr<TensorViewNode> tvn(std::make_shared<TensorViewNode>(outTensor, false, "outTensor"));

    std::vector<SizeVector> elementOffsets;
    elementOffsets.push_back({0, 0});
    elementOffsets.push_back({2, 0});
    elementOffsets.push_back({0, 2});

    std::vector<unsigned> expectedOutput = {4, 4};

    std::vector<std::vector<unsigned>> byteOffsets;
    for(size_t t = 0; t < inTensors.size(); ++t)
    {
        auto& tin = inTensors[t];
        tin = std::make_shared<Tensor>(inTensorDim, inSizes, syn_type_single);

        tvn->addView(tin, elementOffsets[t]);
        inputs[t].geometry.dims = tin->getDim();

        TSize sizes[tpc_lib_api::MAX_TENSOR_DIM];
        tin->getAllSizesInElements(sizes, tpc_lib_api::MAX_TENSOR_DIM);
        memcpy(inputs[t].geometry.maxSizes, sizes, tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));
    }

    SifNodeParams metadata = tvn->getShapeInferenceFunctionUserParams();
    size_t metadataSize = tvn->getShapeInferenceFunctionUserParamsSize();

    TensorShapeInfo output = {{2, {}}};
    std::vector<TensorShapeInfo*> inputsPtr = {&inputs[0], &inputs[1], &inputs[2]};
    std::vector<TensorShapeInfo*> outputsPtr = {&output};

    unsigned invalidMask = inferShape(SIF_TENSOR_VIEW, inputsPtr, outputsPtr, metadataSize, metadata);
    EXPECT_EQ(0, invalidMask);

    for(size_t d = 0; d < output.geometry.dims; ++d)
    {
        EXPECT_EQ(expectedOutput[d], output.geometry.maxSizes[d]) << "wrong output shape of tensor view op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_logical_transpose_op_shape)
{
    transposeTest(SIF_TRANSPOSE);
}

TEST_F(LogicalInferShapeFunctions, infer_logical_transpose_op_shape_negative)
{
    SifTransposeMetadata metadata;
    unsigned sizes[] = {3, 2, 1, 0, 0};
    memcpy(&metadata.permutation, &sizes, sizeof(sizes));
    negativeTestSingleInputSingleOutputWithMetadata(SIF_TRANSPOSE, sizeof(metadata), &metadata);
}

TEST_F(MmeInferShapeFunctions, infer_convolution_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    size_t B = 1;
    size_t H = 128;
    size_t W = 128;
    size_t C = 16;

    SifConvolutionMetadata metadata = { };
    auto& kernel = metadata.params.kernel;
    auto& stride = metadata.params.stride;
    auto& padding = metadata.params.padding;
    auto& dilation = metadata.params.dilation;

    kernel[CONV_KERNEL_WIDTH]   = 3;
    kernel[CONV_KERNEL_HEIGHT]  = 3;
    stride[CONV_STRIDE_WIDTH]   = 1;
    stride[CONV_STRIDE_HEIGHT]  = 1;
    padding[CONV_PAD_LEFT]      = 0;
    padding[CONV_PAD_RIGHT]     = 0;
    padding[CONV_PAD_TOP]       = 0;
    padding[CONV_PAD_BOTTOM]    = 0;
    dilation[CONV_DIL_WIDTH]    = 1;
    dilation[CONV_DIL_HEIGHT]   = 1;

    const unsigned convW = convOutputDimSize(W,
                                             kernel[CONV_KERNEL_WIDTH],
                                             stride[CONV_STRIDE_WIDTH],
                                             padding[CONV_PAD_LEFT] + padding[CONV_PAD_RIGHT],
                                             dilation[CONV_DIL_WIDTH]);

    const unsigned convH = convOutputDimSize(H,
                                             kernel[CONV_KERNEL_HEIGHT],
                                             stride[CONV_STRIDE_HEIGHT],
                                             padding[CONV_PAD_TOP] + padding[CONV_PAD_BOTTOM],
                                             dilation[CONV_DIL_HEIGHT]);


    TensorShapeInfo inputTensorShapeInfo = {{4, {C, W, H, B, 0}}};
    TensorShapeInfo weightsTensorShapeInfo = {{4, {C, C, kernel[CONV_KERNEL_WIDTH], kernel[CONV_KERNEL_HEIGHT], 0}}};
    TensorShapeInfo expectedOutput = {{4, {C, convW, convH, B, 0}}};
    TensorShapeInfo outputTensorShapeInfo = {{4, {}}};
    memcpy(metadata.maxOutputSizes, expectedOutput.geometry.maxSizes , tpc_lib_api::MAX_TENSOR_DIM * sizeof(TSize));

    inputs.push_back(&inputTensorShapeInfo);
    inputs.push_back(&weightsTensorShapeInfo);
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(SIF_CONVOLUTION, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        EXPECT_EQ(expectedOutput.geometry.maxSizes[d], output->geometry.maxSizes[d]) << "wrong output shape of convolution op";
    }
}

TEST_F(MmeInferShapeFunctions, infer_convolution_op_shape_negative)
{
    negativeConv(SIF_CONVOLUTION);
}

TEST_F(MmeInferShapeFunctions, infer_conv_dedw_op_shape)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    SifConvolutionMetadata metadata = { };
    auto& kernel = metadata.params.kernel;
    auto& stride = metadata.params.stride;
    auto& padding = metadata.params.padding;
    auto& dilation = metadata.params.dilation;

    kernel[CONV_KERNEL_WIDTH]   = 3;
    kernel[CONV_KERNEL_HEIGHT]  = 3;
    stride[CONV_STRIDE_WIDTH]   = 1;
    stride[CONV_STRIDE_HEIGHT]  = 1;
    padding[CONV_PAD_LEFT]      = 0;
    padding[CONV_PAD_RIGHT]     = 0;
    padding[CONV_PAD_TOP]       = 0;
    padding[CONV_PAD_BOTTOM]    = 0;
    dilation[CONV_DIL_WIDTH]    = 1;
    dilation[CONV_DIL_HEIGHT]   = 1;

    const unsigned tensorDim = 4;
    const unsigned B = 1;
    const unsigned xH = 128;
    const unsigned xW = 128;
    const unsigned xC = 16;
    const unsigned yW = convOutputDimSize(xW,
                                          kernel[CONV_KERNEL_WIDTH],
                                          stride[CONV_STRIDE_WIDTH],
                                          padding[CONV_PAD_LEFT] + padding[CONV_PAD_RIGHT],
                                          dilation[CONV_DIL_WIDTH]);
    const unsigned yH = convOutputDimSize(xH,
                                          kernel[CONV_KERNEL_HEIGHT],
                                          stride[CONV_STRIDE_HEIGHT],
                                          padding[CONV_PAD_TOP] + padding[CONV_PAD_BOTTOM],
                                          dilation[CONV_DIL_HEIGHT]);
    const unsigned yC = xC;

    TensorShapeInfo dedyTensorShapeInfo = {{tensorDim,{yC, yW, yH, B, 0}}};
    TensorShapeInfo inputTensorShapeInfo = {{tensorDim, {xC, xW, xH, B, 0}}};
    TensorShapeInfo expectedOutput = {{tensorDim, {yC, xC, kernel[CONV_KERNEL_WIDTH], kernel[CONV_KERNEL_HEIGHT], 0}}};
    TensorShapeInfo outputTensorShapeInfo = {{tensorDim, {}}};

    inputs.push_back(&dedyTensorShapeInfo);
    inputs.push_back(&inputTensorShapeInfo);
    outputs.push_back(&outputTensorShapeInfo);

    unsigned invalidMask = inferShape(SIF_CONV_DEDW, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();
    EXPECT_EQ(inputTensorShapeInfo.geometry.dims, output->geometry.dims);

    for(size_t d = 0; d < output->geometry.dims; ++d)
    {
        EXPECT_EQ(expectedOutput.geometry.maxSizes[d], output->geometry.maxSizes[d]) << "wrong output shape of convolution op";
    }
}

TEST_F(MmeInferShapeFunctions, infer_conv_dedw_op_shape_negative)
{
    negativeConv(SIF_CONV_DEDW);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_op_shape)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 0}, {256, 256, 0}};
    gemmTest(SIF_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_op_shape_with_bias)
{
    SifGemmMetadata metadata       = {};
    GemmParams      params         = {{256, 128, 0}, {256, 256, 0}, 2};
    TensorShapeInfo expectedOutput = getGemmExpectedOutput(metadata, params, false);
    unsigned        biasSize       = expectedOutput.geometry.maxSizes[0];
    gemmTest(SIF_GEMM, metadata, params, false, biasSize);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_op_shape_transposed_a)
{
    SifGemmMetadata metadata = { };
    metadata.params.transpose_a = true;
    GemmParams params = {{128, 256, 0}, {256, 256, 0}};
    gemmTest(SIF_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_op_shape_transposed_b)
{
    SifGemmMetadata metadata = { };
    metadata.params.transpose_b = true;
    GemmParams params = {{256, 256, 0}, {256, 128, 0}};
    gemmTest(SIF_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_dedw_op_shape)
{
    SifGemmMetadata metadata = { };
    GemmParams params = {{256, 128, 0}, {256, 256, 0}};
    gemmTest(SIF_GEMM_DEDW, metadata, params, true /* transpose result */);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_dedx_op_shape)
{
    SifGemmMetadata metadata = { };
    GemmParams params = {{256, 128, 0}, {256, 256, 0}};
    gemmTest(SIF_GEMM_DEDX, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_fc_op_shape)
{
    SifGemmMetadata metadata = { };
    GemmParams params = {{256, 128, 0}, {256, 256, 0}};
    gemmTest(SIF_GEMM_FC, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_gemm_op_shape_negative)
{
    SifGemmMetadata metadata     = {};
    unsigned        metadataSize = sizeof(metadata);

    TensorShapeInfo tsi2 = {{2, {2, 2, 0, 0, 0}}};
    TensorShapeInfo tsi4 = {{4, {2, 2, 2, 2, 0}}};

    ShapeFuncID sifId = SIF_GEMM;

    inferShape(sifId, {&tsi2, &tsi2}, {&tsi2}, metadataSize, &metadata, tpc_lib_api::GLUE_SUCCESS);
    inferShape<unsigned>(sifId, {&tsi2, &tsi2}, {&tsi2}, 0, nullptr, tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE);
    inferShape(sifId, {&tsi4, &tsi2}, {&tsi2}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
    inferShape(sifId, {&tsi2, nullptr}, {&tsi2}, metadataSize, &metadata, tpc_lib_api::GLUE_SIF_NULL_PTR);
    inferShape(sifId, {}, {&tsi2}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi2}, {&tsi2}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
    inferShape(sifId, {&tsi2, &tsi2}, {}, metadataSize, &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);

    // wrong transpose
    gemmTest(SIF_GEMM, {{true, false}}, {{256, 128, 0}, {256, 256, 0}}, false, -1, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
    gemmTest(SIF_GEMM, {{false, true}}, {{256, 256, 0}, {128, 256, 0}}, false, -1, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);

    // wrong bias
    TensorShapeInfo expectedOutput = getGemmExpectedOutput(metadata, {{256, 128, 0}, {256, 256, 0}, 2}, false);
    unsigned        wrongBiasSize  = expectedOutput.geometry.maxSizes[0] + 1;
    gemmTest(SIF_GEMM,
             metadata,
             {{256, 128, 0}, {256, 256, 0}},
             false,
             wrongBiasSize,
             tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_3d)
{
    SifGemmMetadata metadata = { };
    GemmParams params = {{256, 128, 2, 0, 0}, {256, 256, 2, 0, 0}, 3};
    batchGemmTest(SIF_BATCH_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_3d_with_bias)
{
    SifGemmMetadata metadata       = {};
    GemmParams      params         = {{256, 128, 2, 0, 0}, {256, 256, 2, 0, 0}, 3};
    TensorShapeInfo expectedOutput = getGemmExpectedOutput(metadata, params, false);
    unsigned        biasSize       = expectedOutput.geometry.maxSizes[0];
    batchGemmTest(SIF_BATCH_GEMM, metadata, params, false, biasSize);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_3d_with_bias_negative)
{
    SifGemmMetadata metadata       = {};
    GemmParams      params         = {{256, 128, 2, 0, 0}, {256, 256, 2, 0, 0}, 3};
    TensorShapeInfo expectedOutput = getGemmExpectedOutput(metadata, params, false);
    unsigned        wrongBiasSize  = expectedOutput.geometry.maxSizes[0] + 1;
    batchGemmTest(SIF_BATCH_GEMM, metadata, params, false, wrongBiasSize, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_5d)
{
    SifGemmMetadata metadata = { };
    GemmParams params = {{256, 128, 2, 4, 5}, {256, 256, 2, 4, 5}, 5};
    batchGemmTest(SIF_BATCH_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_5d_asimetric)
{
    SifGemmMetadata metadata = { };
    GemmParams params = {{256, 128, 2, 4, 5}, {256, 256, 1, 1, 1}, 5};
    batchGemmTest(SIF_BATCH_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_dim_input_broadcast)
{
    SifGemmMetadata metadata = { };
    TensorShapeInfo tensor1 = {{4, {256, 128, 1, 1}}};
    TensorShapeInfo tensor2 = {{5, {256, 256, 2, 4, 5}}};

    TensorShapeInfo outputTensorShapeInfo = {{5, {}}};

    std::vector<TensorShapeInfo*> inputs = {&tensor1, &tensor2};
    std::vector<TensorShapeInfo*> outputs = {&outputTensorShapeInfo};

    inferShape(SIF_BATCH_GEMM, inputs, outputs, sizeof(metadata), &metadata);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dw_op_shape_3d)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 0, 0}, {256, 256, 2, 0, 0}, 3};
    batchGemmTest(SIF_BATCH_GEMM_DEDW, metadata, params, true /* transpose output */);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dw_op_shape_5d)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 4, 5}, {256, 256, 2, 4, 5}, 5};
    batchGemmTest(SIF_BATCH_GEMM_DEDW, metadata, params, true /* transpose output */);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dw_op_shape_5d_asimetric)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 4, 5}, {256, 256, 1, 1, 1}, 5};
    batchGemmTest(SIF_BATCH_GEMM_DEDW, metadata, params, true /* transpose output */);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dw_op_shape_dim_inputs_mismatch)
{
    SifGemmMetadata metadata = {};
    TensorShapeInfo tensor1  = {{4, {256, 128, 1, 1}}};
    TensorShapeInfo tensor2  = {{5, {256, 256, 2, 4, 5}}};

    TensorShapeInfo outputTensorShapeInfo = {{5, {}}};

    std::vector<TensorShapeInfo*> inputs  = {&tensor1, &tensor2};
    std::vector<TensorShapeInfo*> outputs = {&outputTensorShapeInfo};

    inferShape(SIF_BATCH_GEMM_DEDW, inputs, outputs, sizeof(metadata), &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_op_shape_dim_input_mismatch)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 3, 1, 1}, {256, 256, 2, 4, 5}, 5};
    batchGemmTest(SIF_BATCH_GEMM, metadata, params, false, -1, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dx_op_shape_3d)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 0, 0}, {256, 256, 2, 0, 0}, 3};
    batchGemmTest(SIF_BATCH_GEMM_DEDX, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_shape_partial_broadcast)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 1, 0}, {256, 256, 1, 3, 0}, 4};
    batchGemmTest(SIF_BATCH_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_shape_asymmetric_weights)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 1, 1, 0}, {256, 256, 4, 3, 0}, 4};
    batchGemmTest(SIF_BATCH_GEMM, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dx_op_shape_5d)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 4, 5}, {256, 256, 2, 4, 5}, 5};
    batchGemmTest(SIF_BATCH_GEMM_DEDX, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dx_op_shape_5d_asimetric)
{
    SifGemmMetadata metadata = {};
    GemmParams      params   = {{256, 128, 2, 4, 5}, {256, 256, 1, 1, 1}, 5};
    batchGemmTest(SIF_BATCH_GEMM_DEDX, metadata, params);
}

TEST_F(MmeInferShapeFunctions, infer_batch_gemm_de_dx_op_shape_dim_inputs_mismatch)
{
    SifGemmMetadata metadata = {};
    TensorShapeInfo tensor1  = {{4, {256, 128, 1, 1}}};
    TensorShapeInfo tensor2  = {{5, {256, 256, 2, 4, 5}}};

    TensorShapeInfo outputTensorShapeInfo = {{5, {}}};

    std::vector<TensorShapeInfo*> inputs  = {&tensor1, &tensor2};
    std::vector<TensorShapeInfo*> outputs = {&outputTensorShapeInfo};

    inferShape(SIF_BATCH_GEMM_DEDX, inputs, outputs, sizeof(metadata), &metadata, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE);
}

TEST_F(TpcInferShapeFunctions, DynamicShapeInfer)
{
    synStaticReshapeSifParams metadata = {{3, 3, 3, 3}, {3, 3, 3, 3}, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, 4};
    metadata.dimsNum = 4;
    unsigned metadataSize = sizeof(metadata);
    TensorShapeInfo tsiIn = {{4, {2, 2, 2, 2, 0}}};
    TensorShapeInfo tsiOut = {{4, {0, 0, 0, 0, 0}}};
    inferShape(SIF_STATIC_RESHAPE, {&tsiIn}, {&tsiOut}, metadataSize, &metadata, tpc_lib_api::GLUE_SUCCESS);
    for(int i = 0; i < tsiIn.geometry.dims; i++)
        EXPECT_EQ(tsiIn.geometry.maxSizes[i], tsiOut.geometry.maxSizes[i]);
}

TEST_F(TpcInferShapeFunctions, StaticShapeInfer)
{
    synStaticReshapeSifParams metadata     = {{2, 2, 2, 2}, {2, 2, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, 0, 4};
    unsigned metadataSize = sizeof(metadata);
    TensorShapeInfo tsiIn = {{4, {2, 2, 2, 2, 0}}};
    TensorShapeInfo tsiOut = {{4, {0, 0, 0, 0, 0}}};
    inferShape(SIF_STATIC_RESHAPE, {&tsiIn}, {&tsiOut}, metadataSize, &metadata, tpc_lib_api::GLUE_SUCCESS);
    for(int i = 0; i < tsiIn.geometry.dims; i++)
    {
        EXPECT_EQ(metadata.outputMaxSizes[i], tsiOut.geometry.maxSizes[i]);
    }
}

TEST_F(TpcInferShapeFunctions, MixedShapeInfer)
{
    synStaticReshapeSifParams metadata       = {{0, 0, 3, 0}, {0, 0, 3, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}, 0, 4};
    unsigned metadataSize = sizeof(metadata);
    TensorShapeInfo           tsiIn          = {{4, {2, 2, 3, 2, 0}}};
    TensorShapeInfo tsiOut = {{4, {0, 0, 0, 0, 0}}};
    int dynamicDimsNum = 0;
    int staticDimsNum = 0;
    inferShape(SIF_STATIC_RESHAPE, {&tsiIn}, {&tsiOut}, metadataSize, &metadata, tpc_lib_api::GLUE_SUCCESS);
    for(int i = 0; i < tsiIn.geometry.dims; i++)
    {
        if (metadata.outputStaticDims[i] == false)
        {
            dynamicDimsNum++;
            EXPECT_EQ(tsiIn.geometry.maxSizes[i], tsiOut.geometry.maxSizes[i]);
        }
        else
        {
            staticDimsNum++;
            EXPECT_EQ(metadata.outputMaxSizes[i], tsiOut.geometry.maxSizes[i]);
        }
    }
    int expectedStatisDimsNum =
        std::accumulate(&metadata.outputStaticDims[0], &metadata.outputStaticDims[metadata.dimsNum], 0);
    int expectedDynamicDimsNum = metadata.dimsNum - expectedStatisDimsNum;
    EXPECT_EQ(expectedStatisDimsNum, staticDimsNum);
    EXPECT_EQ(expectedDynamicDimsNum, dynamicDimsNum);
}

TEST_F(LogicalInferShapeFunctions, infer_squeeze_op_shape_no_params)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo =  {{5, {1, 2, 3, 1, 4}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{3, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifSqueezeMetadata params = {1, 0, 0, 1, 0};

    unsigned invalidMask = inferShape(SIF_SQUEEZE, inputs, outputs, sizeof(params), &params);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();

    std::vector<unsigned> expectedDims = {2, 3 ,4};

    for(size_t d = 0; d < outputTensorShapeInfo.geometry.dims; ++d)
    {
        EXPECT_EQ(expectedDims[d], output->geometry.maxSizes[d]) << "wrong output shape of flatten op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_squeeze_op_shape_with_params)
{
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;

    TensorShapeInfo inputTensorShapeInfo = {{5, {2, 1, 1, 3, 4}}};
    inputs.push_back(&inputTensorShapeInfo);

    TensorShapeInfo outputTensorShapeInfo = {{4, {}}};
    outputs.push_back(&outputTensorShapeInfo);

    SifSqueezeMetadata metadata = {0, 1, 0, 0, 0};

    unsigned invalidMask = inferShape(SIF_SQUEEZE, inputs, outputs, sizeof(metadata), &metadata);
    EXPECT_EQ(0, invalidMask);

    TensorShapeInfo* output = outputs.front();

    std::vector<unsigned> expectedDims = {2, 1, 3, 4};

    for(size_t d = 0; d < outputTensorShapeInfo.geometry.dims; ++d)
    {
        EXPECT_EQ(expectedDims[d], output->geometry.maxSizes[d]) << "wrong output shape of flatten op";
    }
}

TEST_F(LogicalInferShapeFunctions, infer_squeeze_op_shape_negative)
{
    SifSqueezeMetadata params = {0, 1, 0, 0};
    negativeTestSingleInputSingleOutputWithMetadata(SIF_SQUEEZE, sizeof(params), &params);
}

std::vector<TensorShapeInfo> createMomentsShapeInferInput(unsigned dims, std::vector<unsigned> sizes)
{
    std::vector<TensorShapeInfo> inputs(1);

    inputs[0].geometry.dims = dims;
    for (int i = 0; i < sizes.size(); i++)
    {
        inputs[0].geometry.maxSizes[i] = sizes[i];
    }

    return inputs;
}

std::vector<TensorShapeInfo> createMomentsShapeInferOutputs()
{
    std::vector<TensorShapeInfo> outputs(2);
    return outputs;
}

void verifyMomentsOutputSizes(unsigned dims, unsigned channelsNum, const std::vector<TensorShapeInfo>& outputs)
{
    EXPECT_EQ(dims, outputs[0].geometry.dims);
    EXPECT_EQ(dims, outputs[1].geometry.dims);

    EXPECT_EQ(channelsNum, outputs[0].geometry.maxSizes[0]);
    EXPECT_EQ(channelsNum, outputs[1].geometry.maxSizes[0]);

    for (int i = 1; i < dims; i++)
    {
        EXPECT_EQ(1, outputs[0].geometry.maxSizes[i]);
        EXPECT_EQ(1, outputs[1].geometry.maxSizes[i]);
    }
}

TEST_F(MomentsInferShapeFunctions, infer_moments_shape)
{
    unsigned inputDims   = 4;
    unsigned channelsNum = 3;

    std::vector<TensorShapeInfo> inputs  = createMomentsShapeInferInput(inputDims, {channelsNum, 4, 3, 2});
    std::vector<TensorShapeInfo> outputs = createMomentsShapeInferOutputs();

    inferShape(SIF_MOMENTS, inputs, outputs);
    verifyMomentsOutputSizes(inputDims, channelsNum, outputs);
}

TEST_F(MomentsInferShapeFunctions, infer_moments_shape_one_dim_size)
{
    unsigned inputDims   = 4;
    unsigned channelsNum = 1;

    std::vector<TensorShapeInfo> inputs  = createMomentsShapeInferInput(inputDims, {channelsNum, 1, 1, 1});
    std::vector<TensorShapeInfo> outputs = createMomentsShapeInferOutputs();

    inferShape(SIF_MOMENTS, inputs, outputs);
    verifyMomentsOutputSizes(inputDims, channelsNum, outputs);
}

TEST_F(MomentsInferShapeFunctions, infer_moments_shape_missing_input)
{
    std::vector<TensorShapeInfo> inputs;
    std::vector<TensorShapeInfo> outputs = createMomentsShapeInferOutputs();

    inferShape<void>(SIF_MOMENTS, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

TEST_F(MomentsInferShapeFunctions, infer_moments_shape_missing_output)
{
    unsigned inputDims   = 4;
    unsigned channelsNum = 1;

    std::vector<TensorShapeInfo> inputs  = createMomentsShapeInferInput(inputDims, {channelsNum, 1, 1, 1});
    std::vector<TensorShapeInfo> outputs = createMomentsShapeInferOutputs();
    outputs.pop_back();

    inferShape<void>(SIF_MOMENTS, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

TEST_F(NOPInferShapeFunctions, infer_wait_op_shape)
{  // the wait is not using inputs or outputs so the SIF does nothing but returns OK status
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;
    inferShape(SIF_WAIT, inputs, outputs);
}

TEST_F(NOPInferShapeFunctions, infer_debug_op_shape)
{  // the debug does not validate inputs or outputs sizes so the SIF does nothing but returns OK status
    std::vector<TensorShapeInfo*> inputs;
    std::vector<TensorShapeInfo*> outputs;
    inferShape(SIF_DEBUG, inputs, outputs);
}

TEST_F(RotateInferShapeFunctions, infer_rotate_shape)
{
    TensorShapeInfo               input  = {{4, {1, 4, 2, 5}}};
    TensorShapeInfo               shape  = {{4, {2, 5, 7, 5}}};
    std::vector<TensorShapeInfo*> inputs = {&input, &shape};

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = {&output};

    inferShape(SIF_ROTATE, inputs, outputs);

    ASSERT_EQ(input.geometry.dims, output.geometry.dims);
    for (unsigned i = 0; i < shape.geometry.dims; i++)
    {
        ASSERT_EQ(shape.geometry.maxSizes[i], output.geometry.maxSizes[i]);
    }
}

TEST_F(RotateInferShapeFunctions, infer_rotate_shape_no_inputs)
{
    std::vector<TensorShapeInfo*> inputs;

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = {&output};

    inferShape<void>(SIF_ROTATE, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

TEST_F(RotateInferShapeFunctions, infer_rotate_shape_no_shape_input)
{
    TensorShapeInfo               input  = {{4, {1, 4, 2, 5}}};
    std::vector<TensorShapeInfo*> inputs = {&input};

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = {&output};

    inferShape<void>(SIF_ROTATE, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

TEST_F(RotateInferShapeFunctions, infer_rotate_shape_no_output)
{
    TensorShapeInfo               input  = {{4, {1, 4, 2, 5}}};
    TensorShapeInfo               shape  = {{4, {2, 5, 7, 5}}};
    std::vector<TensorShapeInfo*> inputs = {&input, &shape};

    std::vector<TensorShapeInfo*> outputs;

    inferShape<void>(SIF_ROTATE, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

TEST_F(RotateInferShapeFunctions, infer_rotate_shape_too_many_inputs)
{
    TensorShapeInfo               input  = {{4, {1, 4, 2, 5}}};
    TensorShapeInfo               shape  = {{4, {2, 5, 7, 5}}};
    std::vector<TensorShapeInfo*> inputs = {&input, &input, &shape};

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = {&output};

    inferShape<void>(SIF_ROTATE, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

TEST_F(RotateInferShapeFunctions, infer_rotate_shape_too_many_outputs)
{
    TensorShapeInfo               input  = {{4, {1, 4, 2, 5}}};
    TensorShapeInfo               shape  = {{4, {2, 5, 7, 5}}};
    std::vector<TensorShapeInfo*> inputs = {&input, &shape};

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = {&output, &output};

    inferShape<void>(SIF_ROTATE, inputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

TEST_F(TfBatchNormInferShapeFunctions, infer_moments_shape)
{
    TensorShapeInfo               input  = {{3, {1, 4, 2}}};
    std::vector<TensorShapeInfo*> inputs = addInputs(&input);

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = addOutputs(&output);

    inferShape(SIF_TF_BATCH_NORM, inputs, outputs);

    ASSERT_EQ(input.geometry.dims, output.geometry.dims);
    for (unsigned i = 0; i < input.geometry.dims; i++)
    {
        ASSERT_EQ(input.geometry.maxSizes[i], output.geometry.maxSizes[i]);
    }
}

TEST_F(TfBatchNormInferShapeFunctions, infer_shape_no_input)
{
    std::vector<TensorShapeInfo*> noInputs;
    TensorShapeInfo               output;
    auto                          outputs = addOutputs(&output);
    inferShape<void>(SIF_TF_BATCH_NORM, noInputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

TEST_F(TfBatchNormInferShapeFunctions, infer_shape_missing_input)
{
    TensorShapeInfo               input         = {{3, {1, 4, 2}}};
    std::vector<TensorShapeInfo*> missingInputs = addInputs(&input, properInputsNum - 1);

    TensorShapeInfo               output;
    std::vector<TensorShapeInfo*> outputs = addOutputs(&output);

    inferShape<void>(SIF_TF_BATCH_NORM, missingInputs, outputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT);
}

TEST_F(TfBatchNormInferShapeFunctions, infer_shape_missing_output)
{
    TensorShapeInfo               input  = {{3, {1, 4, 2}}};
    std::vector<TensorShapeInfo*> inputs = addInputs(&input);
    std::vector<TensorShapeInfo*> noOutputs;

    inferShape<void>(SIF_MOMENTS, inputs, noOutputs, 0, nullptr, tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT);
}

TEST_F(DynamicReshapeInferShapeFunctions, dynamic_reshape)
{
    /* Test 1 - constants */
    TensorShapeInfo input  = {{4, {10, 10, 10, 10}}};
    TensorShapeInfo output = {{4, {5, 20, 5, 20}}};
    SifDynamicReshapeMetadata metadata = {4, {'a','b','c','d'}, "a/2,b*2,c-5,d+10"};
    std::vector<TensorShapeInfo*> inputs  = {&input};
    std::vector<TensorShapeInfo*> outputs = {&output};

    inferShape(SIF_DYNAMIC_RESHAPE, inputs, outputs, sizeof(metadata), &metadata);

    /* Test 2 - labels */
    output.geometry.dims     = 1;
    output.geometry.maxSizes[0] = 10000;
    strncpy(metadata.output_eq, "a*b*c*d", sizeof(metadata.output_eq));

    inferShape(SIF_DYNAMIC_RESHAPE, inputs, outputs, sizeof(metadata), &metadata);
}
