#include "syn_gaudi_two_run_compare_test.h"

void SynTrainingTwoRunCompareTest::SetUpTest()
{
    SynTrainingTestInfra::SetUpTest();
    // Create the second graph;
    createGraph();
}

void SynTrainingTwoRunCompareTest::addConfigurationToRun(RunIndex runIdx, std::string&& globalConf, std::string&& value)
{
    m_runsConfig[runIdx].push_back(std::make_pair(globalConf, value));
}

void SynTrainingTwoRunCompareTest::compareRunsResults(const std::vector<unsigned>& firstIdxs)
{
    if (!m_executed)
    {
        compileAndRun();
    }

    for(unsigned firstIdx : firstIdxs)
    {
        unsigned secondIdx = m_firstToSecondTensorIdx[firstIdx];
        uint64_t length;
        auto     it = m_actualSizesForValidation.find(firstIdx);
        if (it != m_actualSizesForValidation.end())
        {
            length = it->second;
        }
        else
        {
            length = multiplyElements(m_tensorDescs[firstIdx].m_sizes,
                                      m_tensorDescs[firstIdx].m_sizes + m_tensorDescs[firstIdx].m_dims);
        }

        switch (m_tensorDescs[firstIdx].m_dataType)
        {
        case syn_type_float:
            validateResult(castHostBuffer<float>(firstIdx), castHostBuffer<float>(secondIdx), length, m_tensorDescs[firstIdx].m_name);
            break;
        case syn_type_bf16:
            validateResult(castHostBuffer<bfloat16>(firstIdx), castHostBuffer<bfloat16>(secondIdx), length, m_tensorDescs[firstIdx].m_name);
            break;
        case syn_type_int8:
            validateResult(castHostBuffer<int8_t>(firstIdx),
                           castHostBuffer<int8_t>(secondIdx),
                           length,
                           m_tensorDescs[firstIdx].m_name);
            break;
        case syn_type_int16:
            validateResult(castHostBuffer<int16_t>(firstIdx),
                           castHostBuffer<int16_t>(secondIdx),
                           length,
                           m_tensorDescs[firstIdx].m_name);
            break;
        case syn_type_fp8_152:
            validateResult(castHostBuffer<fp8_152_t>(firstIdx),
                           castHostBuffer<fp8_152_t>(secondIdx),
                           length,
                           m_tensorDescs[firstIdx].m_name);
            break;
        default:
            HB_ASSERT(0, "Unsupported data type {} for test check", m_tensorDescs[firstIdx].m_dataType);
        }
    }
}

unsigned SynTrainingTwoRunCompareTest::createSection(uint64_t size)
{
    unsigned firstRunSectionIdx                   = SynTrainingTestInfra::createSection(size, FIRST_RUN);
    unsigned secondRunSectionIdx                  = SynTrainingTestInfra::createSection(size, SECOND_RUN);
    m_firstToSecondSectionIdx[firstRunSectionIdx] = secondRunSectionIdx;
    return firstRunSectionIdx;
}

unsigned SynTrainingTwoRunCompareTest::createConstSection(uint64_t graphIndex)
{
    unsigned firstRunConstSectionIdx             = SynTrainingTestInfra::createConstSection(FIRST_RUN);
    unsigned secondRunConstSectionIdx            = SynTrainingTestInfra::createConstSection(SECOND_RUN);
    m_firstToSecondSectionIdx[firstRunConstSectionIdx] = secondRunConstSectionIdx;
    return firstRunConstSectionIdx;
}

SynTrainingTwoRunCompareTest::TensorIndices SynTrainingTwoRunCompareTest::createTensors(unsigned        numTensors,
                                                                                        TensorUsage     usage,
                                                                                        bool            isPersistent,
                                                                                        const char*     name,
                                                                                        MemInitType     initSelect,
                                                                                        const float*    initializer,
                                                                                        unsigned*       sizes,
                                                                                        unsigned        dims,
                                                                                        synDataType     dataType,
                                                                                        unsigned*       strides,
                                                                                        unsigned        graphIndex,
                                                                                        unsigned        offsetInSection,
                                                                                        const unsigned* sectionIndex,
                                                                                        bool            isConst,
                                                                                        unsigned*       minSizes,
                                                                                        synTensorType   tensorType,
                                                                                        synTensor       existingTensor)
{
    auto ret = SynTrainingTestInfra::createTensors(numTensors,
                                                   usage,
                                                   isPersistent,
                                                   name,
                                                   initSelect,
                                                   initializer,
                                                   sizes,
                                                   dims,
                                                   dataType,
                                                   strides,
                                                   FIRST_RUN,
                                                   offsetInSection,
                                                   sectionIndex,
                                                   isConst,
                                                   minSizes,
                                                   tensorType);
    for (unsigned firstRunIdx : ret)
    {
        if (sectionIndex)
        {
            sectionIndex = &m_firstToSecondSectionIdx[*sectionIndex];
        }
        unsigned secondRunIdx = SynTrainingTestInfra::createTensors(
            1,
            usage,
            isPersistent,
            nullptr,  // Create unique name for the second run
            initSelect == MEM_INIT_COMPILATION_ONLY ? MEM_INIT_COMPILATION_ONLY : MEM_INIT_FROM_INITIALIZER_NO_CAST,
            (float*)m_hostBuffers[firstRunIdx],
            sizes,
            dims,
            dataType,
            strides,
            SECOND_RUN,
            offsetInSection,
            sectionIndex,
            isConst,
            minSizes,
            tensorType)[0];
        m_firstToSecondTensorIdx[firstRunIdx] = secondRunIdx;
    }
    return ret;
}

void SynTrainingTwoRunCompareTest::addNodeToGraph(const char*          guid,
                                                  const TensorIndices& inputTensorIndices,
                                                  const TensorIndices& outputTensorIndices,
                                                  void*                userParams,
                                                  unsigned             paramSize,
                                                  const char*          nodeName,
                                                  unsigned             graphIndex,
                                                  synNodeId*           nodeId,
                                                  const char**         inputLayouts,
                                                  const char**         outputLayouts)
{
    // Add first run node
    std::string nodeNameStr = std::string(nodeName ? nodeName : guid) + "_first_run";
    SynTrainingTestInfra::addNodeToGraph(guid,
                                         inputTensorIndices,
                                         outputTensorIndices,
                                         userParams,
                                         paramSize,
                                         nodeNameStr.c_str(),
                                         FIRST_RUN,
                                         nodeId,
                                         inputLayouts,
                                         outputLayouts);

    // Add second run node
    TensorIndices secondRunIn(inputTensorIndices.size());
    auto          transformFunc = [this](unsigned firstIdx) {
        return firstIdx != INVALID_TENSOR_INDEX ? m_firstToSecondTensorIdx[firstIdx] : INVALID_TENSOR_INDEX;
    };
    std::transform(inputTensorIndices.begin(), inputTensorIndices.end(), secondRunIn.begin(), transformFunc);
    TensorIndices secondRunOut(outputTensorIndices.size());
    std::transform(outputTensorIndices.begin(),
                   outputTensorIndices.end(),
                   secondRunOut.begin(),
                   transformFunc);
    synNodeId secondRunId = 0;
    nodeNameStr           = std::string(nodeName ? nodeName : guid) + "_second_run";
    SynTrainingTestInfra::addNodeToGraph(guid,
                                         secondRunIn,
                                         secondRunOut,
                                         userParams,
                                         paramSize,
                                         nodeNameStr.c_str(),
                                         SECOND_RUN,
                                         nodeId == nullptr ? nodeId : &secondRunId,
                                         inputLayouts,
                                         outputLayouts);
    if (nodeId)
    {
        m_firstToSecondNodeId[*nodeId] = secondRunId;
    }
}

void SynTrainingTwoRunCompareTest::setNodeDependency(const synNodeId* pBlockingNodesIdList,
                                                     const synNodeId* pBlockedNodesIdList,
                                                     const uint32_t   numberblocking,
                                                     const uint32_t   numberblocked,
                                                     unsigned         graphIndex)
{
    SynTrainingTestInfra::setNodeDependency(pBlockingNodesIdList,
                                            pBlockedNodesIdList,
                                            numberblocking,
                                            numberblocked,
                                            FIRST_RUN);

    std::vector<synNodeId> secondRunBlocking(numberblocking);
    auto transformFunc = [this](synNodeId firstId) { return m_firstToSecondNodeId[firstId]; };
    std::transform(pBlockingNodesIdList,
                   pBlockingNodesIdList + numberblocking,
                   secondRunBlocking.begin(),
                   transformFunc);
    std::vector<synNodeId> secondRunBlocked(numberblocked);
    std::transform(pBlockedNodesIdList,
                   pBlockedNodesIdList + numberblocked,
                   secondRunBlocked.begin(),
                   transformFunc);

    SynTrainingTestInfra::setNodeDependency(secondRunBlocking.data(),
                                            secondRunBlocked.data(),
                                            numberblocking,
                                            numberblocked,
                                            SECOND_RUN);
}

void SynTrainingTwoRunCompareTest::setActualSizes(unsigned int tensorIndex, const unsigned int* tensorSizes)
{
    m_actualSizes[FIRST_RUN].emplace_back(tensorIndex, tensorSizes);
    m_actualSizes[SECOND_RUN].emplace_back(m_firstToSecondTensorIdx[tensorIndex], tensorSizes);
    m_actualSizesForValidation[tensorIndex] = std::accumulate(tensorSizes,
                                                              tensorSizes + getTensorDescriptor(tensorIndex).m_dims,
                                                              1,
                                                              std::multiplies<unsigned>());
}

void SynTrainingTwoRunCompareTest::compileAndRun()
{
    compileAndRun(FIRST_RUN);
    compileAndRun(SECOND_RUN);
    m_executed = true;
}

void SynTrainingTwoRunCompareTest::compileAndRun(RunIndex runIdx)
{
    std::list<GlobalConfTestSetter> runConf;
    for (auto conf : m_runsConfig[runIdx])
    {
        runConf.emplace_back(conf.first, conf.second);
    }
    SynTrainingTestInfra::compileTopology(m_testName.value() + "_" + std::to_string(runIdx), runIdx);
    for (auto p : m_actualSizes[runIdx])
    {
        SynTrainingTestInfra::setActualSizes(p.first, p.second, runIdx);
    }
    SynTrainingTestInfra::runTopology(runIdx);
}

void SynTrainingTwoRunCompareTest::addNodeToGraph(const char*  guid,
                                                  void*        userParams,
                                                  unsigned     paramSize,
                                                  const char*  nodeName,
                                                  unsigned     graphIndex,
                                                  const char** inputLayouts,
                                                  const char** outputLayouts)
{
    HB_ASSERT(0, "Not supported method for this test");
}
void SynTrainingTwoRunCompareTest::setGraphsAttributes(synGraphAttribute* attributes, synGraphAttributeVal* values, uint32_t size)
{
    SynTrainingTestInfra::setGraphAttributesV2(attributes, values, size, FIRST_RUN);
    SynTrainingTestInfra::setGraphAttributesV2(attributes, values, size, SECOND_RUN);
}
void SynTrainingTwoRunCompareTest::setGraphsInferenceMode()
{
    SynTrainingTestInfra::setGraphInferenceMode(FIRST_RUN);
    SynTrainingTestInfra::setGraphInferenceMode(SECOND_RUN);
}

void SynTrainingTwoRunCompareTest::setGraphsInferenceModeAndQuantizationEnabled()
{
    SynTrainingTestInfra::setGraphInferenceModeAndQuantizationEnabled(FIRST_RUN);
    SynTrainingTestInfra::setGraphInferenceModeAndQuantizationEnabled(SECOND_RUN);
}

void SynTrainingTwoRunCompareTest::setNodeDeterminstic(const synNodeId& nodeId)
{
    synNodeSetDeterministic(getGraph(FIRST_RUN).graphHandle, nodeId, true);
    synNodeSetDeterministic(getGraph(SECOND_RUN).graphHandle, m_firstToSecondNodeId[nodeId], true);
}

void SynTrainingTwoRunCompareTest::setPermutationForTensor(unsigned                    tensorIndex,
                                                           const std::vector<uint8_t>& permutation)
{
    synTensorPermutation synPermutation;
    synPermutation.dims = permutation.size();
    for (auto i = 0; i < permutation.size(); i++)
    {
        synPermutation.permutation[i] = permutation.at(i);
    }
    synTensorSetPermutation(m_tensors.at(tensorIndex), &synPermutation);
    synTensorSetPermutation(m_tensors.at(m_firstToSecondTensorIdx.at(tensorIndex)), &synPermutation);
}

SynTrainingTwoRunCompareTest::SynTrainingTwoRunCompareTest()
{
    setTestPackage(TEST_PACKAGE_COMPARE_TEST);
}
