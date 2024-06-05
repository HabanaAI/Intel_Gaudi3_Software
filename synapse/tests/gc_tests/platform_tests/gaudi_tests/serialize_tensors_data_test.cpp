#include "gc_gaudi_test_infra.h"
#include "data_serializer/data_serializer.h"
#include "node_factory.h"
#include <fstream>

// this test can't run in CI since it requires GraphCompilerPlugin to be loaded
// to run it localy:
// synrec -m -t -p /tmp/_tensorsDataDumpToFileTest -- run_synapse_test -d -s *serialize_tensors_data_to_file
TEST_F_GC(SynGaudiTestInfra, DISABLED_serialize_tensors_data_to_file)
{
    const std::string graphId  = "tensors_data_dump_test";
    const std::string fileName = "/tmp/_tensorsDataDumpToFileTest.db";

    // clear the data in case the file exists
    {
        std::ofstream(fileName, std::ios::binary | std::ios::trunc);
    }

    unsigned tensorDim = 4;
    unsigned shape[]   = {2, 4, 3, 5};

    const std::string inTensor1Name = "input_1";
    const std::string inTensor2Name = "input_2";
    const std::string outTensorName = "output";

    unsigned inTensor1 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_ALL_ONES,
                                             nullptr,
                                             shape,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             inTensor1Name.c_str(),
                                             0,
                                             0,
                                             nullptr);

    unsigned inTensor2 = createPersistTensor(INPUT_TENSOR,
                                             MEM_INIT_RANDOM_WITH_NEGATIVE,
                                             nullptr,
                                             shape,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             inTensor2Name.c_str(),
                                             0,
                                             0,
                                             nullptr);

    unsigned outTensor = createPersistTensor(OUTPUT_TENSOR,
                                             MEM_INIT_ALL_ZERO,
                                             nullptr,
                                             shape,
                                             tensorDim,
                                             syn_type_single,
                                             nullptr,
                                             outTensorName.c_str(),
                                             0,
                                             0,
                                             nullptr);

    addNodeToGraph(NodeFactory::addNodeTypeName, {inTensor1, inTensor2}, {outTensor});

    compileTopology(graphId);

    runTopology(0, true);

    float* expectedInBuffer1 = castHostInBuffer<float>(inTensor1);
    float* expectedInBuffer2 = castHostInBuffer<float>(inTensor2);
    float* expectedOutBuffer = castHostOutBuffer<float>(outTensor);

    auto           dds       = data_serialize::DataDeserializer::create(fileName)->getGraph({graphId, -1, 0});
    const uint64_t iteration = 0;

    // check data type
    ASSERT_EQ(syn_type_single, dds->getDataType(inTensor1Name, iteration));
    ASSERT_EQ(syn_type_single, dds->getDataType(inTensor2Name, iteration));
    ASSERT_EQ(syn_type_single, dds->getDataType(outTensorName, iteration));

    // check shape
    auto actualInput1Shape = dds->getShape(inTensor1Name, iteration);
    auto actualInput2Shape = dds->getShape(inTensor2Name, iteration);
    auto actualOutputShape = dds->getShape(outTensorName, iteration);

    for (size_t i = 0; i < tensorDim; ++i)
    {
        ASSERT_EQ(shape[i], actualInput1Shape[i]);
        ASSERT_EQ(shape[i], actualInput2Shape[i]);
        ASSERT_EQ(shape[i], actualOutputShape[i]);
    }

    // check data
    uint64_t dataSize = dds->getDataSize(inTensor1Name, iteration);
    uint64_t elements = dataSize / sizeof(float);

    std::vector<uint8_t> actualInBuffer1 = dds->getData(inTensor1Name, iteration);
    std::vector<uint8_t> actualInBuffer2 = dds->getData(inTensor2Name, iteration);
    std::vector<uint8_t> actualOutBuffer = dds->getData(outTensorName, iteration);

    for (size_t i = 0; i < elements; ++i)
    {
        ASSERT_EQ(expectedInBuffer1[i], reinterpret_cast<float*>(actualInBuffer1.data())[i]);
        ASSERT_EQ(expectedInBuffer2[i], reinterpret_cast<float*>(actualInBuffer2.data())[i]);
        ASSERT_EQ(expectedOutBuffer[i], reinterpret_cast<float*>(actualOutBuffer.data())[i]);
    }
}