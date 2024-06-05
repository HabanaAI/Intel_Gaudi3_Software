#include "gc_gaudi_test_infra.h"
#include "infra/gc_synapse_test.h"
#include "node_factory.h"
#include "syn_singleton.hpp"
#include "scoped_configuration_change.h"

TEST_F_GC(SynGaudiTestInfra, printf_vpu_demo_f32, {synDeviceGaudi, synDeviceGaudi2})
{
    const char* kernelName = "printf_vpu_demo_f32";

    unsigned inputTensorSize[2]  = {1024, 1};
    unsigned outputTensorSize[2] = {1024, 1};
    unsigned inputBufferSize = inputTensorSize[0] * inputTensorSize[1];
    unsigned outputBufferSize = outputTensorSize[0] * outputTensorSize[1];

    float inputBuffer[inputBufferSize];
    float outputBuffer[outputBufferSize];

    // populate input buffer
    inputBuffer[0] = 1.234;
    inputBuffer[1] = 2.567;
    inputBuffer[2] = 3.891;

    createPersistTensor(INPUT_TENSOR, MEM_INIT_FROM_INITIALIZER, inputBuffer, inputTensorSize, 1, syn_type_float);
    createPersistTensor(OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, outputBuffer, outputTensorSize, 1, syn_type_float);

    ns_PrintfVpuDemoKernel::Params params;
    params.pos_0 = 0;
    params.pos_1 = 1;
    params.pos_2 = 2;

    // for deubg of gaudi2 scal -> let's the code later clear the ws buffer before use
    // if we don't have the stream, buffer is not cleared
    synStreamHandle streamHandle;
    ASSERT_EQ(synStreamCreateGeneric(&streamHandle, 0, 0), synSuccess) << "Failed to create stream";

    addNodeToGraph(kernelName, &params, sizeof(ns_PrintfVpuDemoKernel::Params));
    compileAndRun();

    char* buff              = new char[GCFG_TPC_PRINTF_TENSOR_SIZE.value()];
    uint32_t* buff32        = reinterpret_cast<uint32_t*>(buff);

    synSingleton* singleton = _SYN_SINGLETON_INTERNAL;
    GraphData& graphData  = m_graphs[0];
    synStatus     status     = singleton->kernelsPrintf(graphData.recipeHandle, graphData.hbmAddr, buff);
    ASSERT_EQ(status, synSuccess);

    std::vector<std::string> prints;
    parsePrintf(buff32, prints);

    HabanaGraph* graph = synSingleton::getInstanceInternal()->getGraph(graphData.graphHandle);

    unsigned numTpcEngines = graph->getNumTpcEng();

    unsigned numROIs = 0;

    auto nodes = graph->getExeSortedNodes();
    for (auto& node : nodes)
    {
        if (graph->runsOnTPC(node))
        {
            numROIs = graph->GetNodeROIs(node)->size();
            break;
        }
    }

    unsigned totalRows = numROIs * numTpcEngines * 5;

    ASSERT_GE(prints.size(), totalRows);

    delete[] buff;
}
