#include "gaudi_tests/dynamic_shapes_types.h"
#include "gc_dynamic_shapes_infra.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include <cstdint>
#include <vector>

// using TripleArray = std::array<unsigned, 3>;

typedef std::vector<unsigned> inputMaxSize;
typedef std::vector<unsigned> inputMinSize;
typedef std::vector<unsigned> inputActualSize;
typedef std::vector<unsigned> intermediateMaxSizes;
typedef std::vector<unsigned> intermediateMinSizes;
typedef std::vector<unsigned> shapeSizes;
typedef std::vector<unsigned> minMaxShapeData;
typedef std::vector<unsigned> actualShapeData;


class SynGaudiDynamicSplitTest
: public SynGaudiDynamicShapesTestsInfra
, public testing::WithParamInterface<std::vector<std::vector<unsigned>>>
{
    void afterSynInitialize() override
    {
        if (m_deviceType == synDeviceGaudi3)
        {
            synConfigurationSet("ENABLE_EXPERIMENTAL_FLAGS", "true");
        }
        SynGaudiDynamicShapesTestsInfra::afterSynInitialize();
    }
};

// add description what is min what is max what is actual .....
// the number of intermediate tensors is shapesize[1] (write it better later)******
INSTANTIATE_TEST_SUITE_P(
    ,
    SynGaudiDynamicSplitTest,
    ::testing::Values(std::vector({inputMaxSize {12, 5, 5},
                                   inputMinSize {3, 1, 1},
                                   inputActualSize {6, 5, 5},
                                   intermediateMaxSizes {4, 5, 5},
                                   intermediateMinSizes {1, 1, 1},
                                   shapeSizes {5, 3, 2},
                                   minMaxShapeData {4, 5, 5, 1, 1, 4, 5, 5, 1, 1, 4, 5, 5, 1, 1,
                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   actualShapeData {2, 5, 5, 1, 1, 3, 5, 5, 1, 1, 1, 5, 5, 1, 1}}),
                      std::vector({inputMaxSize {12, 15, 5},
                                   inputMinSize {1, 3, 1},
                                   inputActualSize {6, 6, 5},
                                   intermediateMaxSizes {12, 5, 5},
                                   intermediateMinSizes {1, 1, 1},
                                   shapeSizes {5, 3, 2},
                                   minMaxShapeData {12, 5, 5, 1, 1, 12, 5, 5, 1, 1, 12, 5, 5, 1, 1,
                                                    1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1},
                                   actualShapeData {6, 2, 5, 1, 1, 6, 2, 5, 1, 1, 6, 2, 5, 1, 1}}),
                      std::vector({inputMaxSize {12, 15, 5},
                                   inputMinSize {4, 1, 1},
                                   inputActualSize {8, 6, 5},
                                   intermediateMaxSizes {3, 15, 5},
                                   intermediateMinSizes {1, 1, 1},
                                   shapeSizes {5, 4, 2},
                                   minMaxShapeData {3, 15, 5, 1, 1, 3, 15, 5, 1, 1, 3, 15, 5, 1, 1, 3, 15, 5, 1, 1,
                                                    1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1},
                                   actualShapeData {2, 6, 5, 1, 1, 2, 6, 5, 1, 1, 2, 6, 5, 1, 1, 2, 6, 5, 1, 1}}),
                      std::vector({inputMaxSize {12, 16, 5},
                                   inputMinSize {1, 4, 1},
                                   inputActualSize {8, 16, 5},
                                   intermediateMaxSizes {12, 4, 5},
                                   intermediateMinSizes {1, 1, 1},
                                   shapeSizes {5, 4, 2},
                                   minMaxShapeData {12, 4, 5, 1, 1, 12, 4, 5, 1, 1, 12, 4, 5, 1, 1, 12, 4, 5, 1, 1,
                                                    1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1},
                                   actualShapeData {8, 4, 5, 1, 1, 8, 4, 5, 1, 1, 8, 4, 5, 1, 1, 8, 4, 5, 1, 1}}),
                      std::vector({inputMaxSize {1, 1, 1, 8},
                                   inputMinSize {1, 1, 1, 4},
                                   inputActualSize {1, 1, 1, 8},
                                   intermediateMaxSizes {1, 1, 1, 2},
                                   intermediateMinSizes {1, 1, 1, 1},
                                   shapeSizes {5, 4, 2},
                                   minMaxShapeData {1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   actualShapeData {1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1}}),
                      std::vector({inputMaxSize {1, 1, 1, 1, 8},
                                   inputMinSize {1, 1, 1, 1, 4},
                                   inputActualSize {1, 1, 1, 1, 8},
                                   intermediateMaxSizes {1, 1, 1, 1, 2},
                                   intermediateMinSizes {1, 1, 1, 1, 1},
                                   shapeSizes {5, 4, 2},
                                   minMaxShapeData {1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2,
                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                   actualShapeData {1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2}})));

TEST_P_GC(SynGaudiDynamicSplitTest, dynamicSplitBasic, {synDeviceGaudi, synDeviceGaudi2, synDeviceGaudi3})
{
    auto lists = GetParam();
    auto tensorDims = lists[0].size();

    unsigned* inMaxSizes = &lists[0][0];
    unsigned* inMinSizes = &lists[1][0];
    unsigned* inActSizes = &lists[2][0];

    unsigned* outMaxSizes = &lists[0][0];
    unsigned* outMinSizes = &lists[1][0];
    unsigned* outActSizes = &lists[2][0];

    unsigned* intermediateMaxSizes = &lists[3][0];
    unsigned* intermediateMinSizes = &lists[4][0];

    unsigned* shapeSizes = &lists[5][0];
    unsigned* minMaxShapeData = &lists[6][0];
    unsigned* actualShapeData = &lists[7][0];
    unsigned opDim = 0;
    for  (int i=0; i<lists[0].size(); i++) {
        if (inMaxSizes[i] != intermediateMaxSizes[i]) {
            opDim = i;
            break;
        }
    }
    unsigned inTensor = createPersistTensor(INPUT_TENSOR,
                                            MEM_INIT_RANDOM_WITH_NEGATIVE,
                                            nullptr,
                                            inMaxSizes,
                                            tensorDims,
                                            syn_type_float,
                                            nullptr,
                                            nullptr,
                                            0,
                                            0,
                                            nullptr,
                                            inMinSizes);

    std::vector<unsigned> intermediateTensors;
    intermediateTensors.reserve(shapeSizes[1]);
    for (int i = 0; i < shapeSizes[1]; i++)
    {
        intermediateTensors.push_back(createTensor(OUTPUT_TENSOR,
                                                MEM_INIT_RANDOM_WITH_NEGATIVE,
                                                nullptr,
                                                intermediateMaxSizes,
                                                tensorDims,
                                                syn_type_float,
                                                nullptr,
                                                intermediateMinSizes));
    }
    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             outMaxSizes,
                                             tensorDims,
                                             syn_type_float,
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                             nullptr,
                                             outMinSizes);

    unsigned shapeTensor = createHostShapeTensor(INPUT_TENSOR, shapeSizes, minMaxShapeData);

    synSplitParams splitParams = {opDim};

    addNodeToGraph(NodeFactory::splitNodeTypeName,
                   {inTensor, shapeTensor},
                   intermediateTensors,
                   &splitParams,
                   sizeof(splitParams));

    synConcatenateParams concatenateParams = {opDim};

    addNodeToGraph(NodeFactory::concatenateNodeTypeName,
                   intermediateTensors,
                   {outTensor},
                   &concatenateParams,
                   sizeof(concatenateParams));

    compileTopology();
    ASSERT_FALSE(HasFailure()) << "Compilation failed";

    setActualSizes(inTensor, inActSizes);
    setActualSizes(outTensor, outActSizes);
    setRuntimeHostBuffer(shapeTensor, actualShapeData);

    auto* inData = castHostInBuffer<float>(inTensor);
    runTopology(0, true);
    ASSERT_FALSE(HasFailure()) << "Launch failed";

    float* outData         = castHostOutBuffer<float>(outTensor);
    auto   totalActualSize = multiplyElements(inActSizes, inActSizes + tensorDims);

    for (int i = 0; i < totalActualSize; i++)
    {
        ASSERT_EQ(inData[i], outData[i]);
    }
}