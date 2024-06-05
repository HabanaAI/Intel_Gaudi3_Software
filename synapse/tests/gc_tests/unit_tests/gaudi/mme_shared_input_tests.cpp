#include "graph_compiler/passes/sram_management/solution_generator.h"
#include "graph_compiler/passes/sram_management/bundlizer.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "graph_compiler/passes/sram_management/bundle_slicer.h"
#include "graph_compiler/passes/sram_management/mme_shared_input.h"
#include "mme_shared_input_tests.h"

bool sliceGraphToSRAMCapacity(HabanaGraph& graph);
namespace gaudi
{

void MMEInterleaveTest::createTensors(const SizeArray& inputXSize,
                                      const SizeArray& inputYSize,
                                      const SizeArray& outputPSize,
                                      const SizeArray& inputZSize,
                                      const SizeArray& outputQSize)
{
    m_inputX = std::make_shared<Tensor>(4, inputXSize.data(), getType());
    m_inputX->setName("inputX");
    m_inputY = std::make_shared<Tensor>(4, inputYSize.data(), getType());
    m_inputY->setName("inputY");
    m_outputP = std::make_shared<Tensor>(4, outputPSize.data(), getType());
    m_outputP->setName("outputP");
    m_inputZ = std::make_shared<Tensor>(4, inputZSize.data(), getType());
    m_inputZ->setName("inputZ");
    m_outputQ = std::make_shared<Tensor>(4, outputQSize.data(), getType());
    m_outputQ->setName("outputQ");
}

void MMEInterleaveTest::addNodesToGraph(GaudiGraph& graph)
{
    TensorVector firstOperation = firstOperationTensors();
    TensorVector secondOperation  = secondOperationTensors();
    HB_ASSERT(firstOperation[2] != nullptr && secondOperation[2] != nullptr, "Call createTensors first");

    pNode node1 = NodeFactory::createNode({firstOperation[0], firstOperation[1]}, {firstOperation[2]},
                                          getParams(0), getParamsSize(), getFirstNodeGuid(), getFirstNodeName());
    pNode node2 = NodeFactory::createNode({secondOperation[0], secondOperation[1]}, {secondOperation[2]},
                                          getParams(1), getParamsSize(), getSecondNodeGuid(), getSecondNodeName());
    GraphEditor::addNode(graph, node1);
    GraphEditor::addNode(graph, node2);
}

void MMEInterleaveTest::findAndAddCandidate(pBundle& bundle, pMmeSlicingStrategy& strategy)
{
    SharedMMEInputCandidateHandler handler;
    MMESlaveBrain slaveBrain(getGraph());
    pBundleExpansion candidate = handler.findSharedMMEInputCandidate(strategy, getGraph()).front();
    pBundleExpansion adjustedCandidate = slaveBrain.adjustCandidateToStrategy(candidate, strategy);
    ASSERT_TRUE(handler.isCandidateValidForStrategy(adjustedCandidate, strategy));
    strategy->getMmeSlicingData().addValidCandidate(adjustedCandidate, false);
    ASSERT_LE(strategy->calculateMetrics().SRAMCapacity, SlicingBrain::knobs.maxSRAMCapInBytes);
    slaveBrain.addSharedOperandMme(adjustedCandidate, strategy);
    bundle->addNode(adjustedCandidate->nodeToStitch);
    SolutionGenerator generator(getGraph(), bundle, strategy);
    ASSERT_TRUE(generator.fillSolution());

}

 pMmeSlicingStrategy GEMMInterleaveTest::makeStrategy(unsigned commonSize, unsigned inputXChunk,
                                                  unsigned inputYChunk, pBundle& bundle)
{
    std::list<pBundle> bundles;
    for(auto& node: getGraph().getNodes())
    {
        auto tmpBundle = std::make_shared<Bundle>(BundleType::MME);
        tmpBundle->addNode(node);
        bundles.push_back(tmpBundle);
    }
    bundle = bundles.front();
    const pNode& node = bundle->getNodes().front();
    pMmeSlicingStrategy strategy = MmeSlicingStrategy::createStrategyForMMENode(*getGraph().getHALReader(), node);
    for (auto& operand : strategy->getMmeSlicingData().getSlicedOperands())
    {
        if (operand->originalTensor == m_inputX)
        {
            operand->chunkDimensions = {inputXChunk, commonSize, 1 ,1, 1};
            operand->numOfBuffers = 2;
        }
        else if (operand->originalTensor == m_inputY)
        {
            bool isTransposed = false;
            if (node->getNodeType() == Node::TYPE_GEMM)
            {
                auto params = (synGEMMParams*)getParams(0);
                unsigned inputIdx = m_sharedOperandPosition_firstGEMM;
                isTransposed = (inputIdx == 0) ? params->transpose_a : params->transpose_b;
            }
            if (isTransposed)
            {
                operand->chunkDimensions = {inputYChunk, commonSize, 1, 1, 1};
            }
            else
            {
                operand->chunkDimensions = {commonSize, inputYChunk, 1, 1, 1};
            }
            operand->numOfBuffers = 2;
        }
        else if (operand->originalTensor == m_outputP)
        {
            operand->chunkDimensions = {inputXChunk, inputYChunk, 1, 1, 1};
        }
        else
            assert(0 && "unkown operand");
    }
    strategy->setInputIsInSRAM(0, true).setInputIsInSRAM(1, true).calculateMetrics();
    strategy->printLog();
    return strategy;
}

void DedxDedwInterleaveTest::createTensors(const SizeArray& xSize,
                                           uint32_t yChannel,
                                           const synConvolutionParams& convParams)
{
    SizeArray dedySize;
    SizeArray dedwSize;
    getConvolutionSize(xSize, 512, convParams, dedwSize, dedySize);
    MMEInterleaveTest::createTensors(xSize, dedySize, dedwSize, dedwSize, xSize);
    TensorVector dedwOperation = firstOperationTensors();
    TensorVector dedxOperation = secondOperationTensors();
    dedwOperation[0]->setName("xTensor");
    dedwOperation[1]->setName("dedyTensor");
    dedwOperation[2]->setName("dedwTensor");
    dedxOperation[1]->setName("wTensor");
    dedxOperation[2]->setName("dedxTensor");
    m_convParams = convParams;
}

TensorVector GEMMInterleaveTest::secondOperationTensors()
{
    assert(m_sharedOperandPosition_secondGEMM == 0 || m_sharedOperandPosition_secondGEMM == 1);
    if (m_sharedOperandPosition_secondGEMM == 0) return {m_inputY, m_inputZ, m_outputQ};
    else return {m_inputZ, m_inputY, m_outputQ};
}

TensorVector GEMMInterleaveTest::firstOperationTensors()
{
    assert(m_sharedOperandPosition_firstGEMM == 0 || m_sharedOperandPosition_firstGEMM == 1);
    if (m_sharedOperandPosition_firstGEMM == 0) return {m_inputY, m_inputX, m_outputP};
    else return {m_inputX, m_inputY, m_outputP};
}

void GEMMInterleaveTest::setSharedOperandPosition(unsigned firstGemm, unsigned secondGemm)
{
    m_sharedOperandPosition_firstGEMM = firstGemm;
    m_sharedOperandPosition_secondGEMM = secondGemm;
}

void GEMMInterleaveTest::transposeSharedInput(unsigned nodeIdx, unsigned operandIdx)
{
    assert (nodeIdx < 2 && operandIdx < 2 && "op index and position must be 0\\1");
    // reset params
    m_paramsA = {false, false};
    m_paramsB = {false, false};
    if (nodeIdx == 0 && operandIdx == 0) m_paramsA.transpose_a = true;
    else if (nodeIdx == 0 && operandIdx == 1) m_paramsA.transpose_b = true;
    else if (nodeIdx == 1 && operandIdx == 0) m_paramsB.transpose_a = true;
    else if (nodeIdx == 1 && operandIdx == 1) m_paramsB.transpose_b = true;
}

void* GEMMInterleaveTest::getParams(unsigned opIdx)
{
    assert (opIdx < 2 && "opIdx must be 0\\1");
    if (opIdx == 0) return &m_paramsA;
    else return &m_paramsB;
}

Solution GEMMInterleaveTest::runTest(unsigned commonSize, unsigned xChunk, unsigned yChunk, unsigned zChunk)
{
    SizeArray inputX  = {commonSize, commonSize, 1, 1};
    SizeArray inputY  = {commonSize, commonSize, 1, 1}; // shared
    SizeArray outputP = {commonSize, commonSize, 1, 1};
    SizeArray inputZ  = {commonSize, commonSize, 1, 1};
    SizeArray outputQ = {commonSize, commonSize, 1, 1};
    createTensors(inputX, inputY, outputP, inputZ, outputQ);
    addNodesToGraph(getGraph());
    pBundle bundle;
    pMmeSlicingStrategy strategy = makeStrategy(commonSize, xChunk, yChunk, bundle);
    findAndAddCandidate(bundle, strategy);
    return bundle->getSolution();
}

/**** Tests     ****/
TEST_F(DedxDedwInterleaveTest, check_interleave_slices)
{
    synConvolutionParams convParams;

    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(18 * 1024 * 1024);
    //Since no flattening pass is assumed we need 2X2 kernel
    convParams.kH = 2;
    convParams.kW = 2;
    SizeArray xSize = {256, 7, 7, 256};
    createTensors(xSize, 256, convParams);

    GaudiGraph graph;

    addNodesToGraph(graph);
    sliceGraphToSRAMCapacity(graph);
    const NodeVector& sortedNodes = graph.getExeSortedNodes();

    Node::eNodeType prevNodeType = Node::TYPE_DEBUG;

    uint32_t numOfinterleavedNodes = 0;

    for (auto node : sortedNodes)
    {
        // Skip not interesting slicer product nodes
        if (node->getNodeType() != Node::TYPE_DEDW && node->getNodeType() != Node::TYPE_DEDX) continue;

        ASSERT_NE(node->getNodeType(), prevNodeType);
        prevNodeType = node->getNodeType();
        ++numOfinterleavedNodes;
    }
    // Validate the nodes is sliced
    ASSERT_TRUE(numOfinterleavedNodes > 2);
}

TEST_F(GEMMInterleaveTest, identification_test)
{
    SizeArray inputX  = {512 * 64, 1024 * 64, 1, 1};
    SizeArray inputY  = {128 * 64, 512 * 64, 1, 1}; // shared
    SizeArray outputP = {128 * 64, 1024 * 64, 1, 1};
    SizeArray inputZ  = {512 * 64, 512 * 64, 1, 1};
    SizeArray outputQ = {128 * 64, 1024 * 64, 1, 1};
    createTensors(inputX, inputY, outputP, inputZ, outputQ);

    std::vector<GaudiGraph> graphVector(2);
    for (unsigned position : {0, 1})
    {
        GaudiGraph& graph = graphVector[position];
        setSharedOperandPosition(position, position);
        addNodesToGraph(graph);
        SharedMMEInputCandidateHandler         handler;
        Bundlizer                              bundlizer(graph);
        MMESlicingBrain                        brain(graph);
        std::map<pBundle, SlicingStrategyList> solvingDataPerBundle;
        // create bundles and strategies
        BundleList bundles = bundlizer.getMMEBundles();
        for (pBundle& bundle : bundles)
        {
            solvingDataPerBundle[bundle] = brain.getSolutionStrategies(bundle);
        }
        ASSERT_EQ(bundles.size(), 2);
        // find expansion candidates.
        for (auto& bundle : bundles)
        {
            ASSERT_FALSE(solvingDataPerBundle[bundle].empty());
            for (auto& strategy : solvingDataPerBundle[bundle])
            {
                pBundleExpansion candidate = handler.findSharedMMEInputCandidate(std::static_pointer_cast<MmeSlicingStrategy>(strategy), graph).front();
                ASSERT_TRUE(candidate && candidate->nodeToStitch);
                ASSERT_EQ(candidate->stitchedOperand->originalTensor, m_inputY);
                ASSERT_NE(candidate->nodeToStitch, candidate->bundleNode);
            }
        }
    }

}

TEST_F(GEMMInterleaveTest, shared_operand_on_same_position)
{
    // the slave operation should be the same then the master.
    unsigned size = 1.6*1024;
    unsigned xChunk = 64, yChunk = 256, zChunk = 64;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk)*2*sizeof(float));
    setSharedOperandPosition(0, 0);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = (div_round_up(size, xChunk) + div_round_up(size, zChunk)) * div_round_up(size, yChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {zChunk, size, 1, 1});
}

TEST_F(GEMMInterleaveTest, shared_operand_on_same_position_slave_smaller_than_master)
{
    // the slave operation should be smaller then the master ,
    // expectation is that the slave will finish a line of operations and wait for the master to finish his line.
    unsigned size = 1.6*1024;
    unsigned xChunk = 32, yChunk = 256, zChunk = 64;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk)*2*sizeof(float));
    setSharedOperandPosition(0, 0);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = (div_round_up(size, xChunk) + div_round_up(size, zChunk)) * div_round_up(size, yChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {zChunk, size, 1, 1});
}

TEST_F(GEMMInterleaveTest, shared_operand_on_same_position_slave_larger_than_master)
{
    // the slave operation should be larger then the master ,
    // expectation is that the master will finish a line of operations and wait for the slave to finish his line.
    unsigned size = 1.6*1024;
    unsigned xChunk = 96, yChunk = 256, zChunk = 64;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk)*2*sizeof(float));
    setSharedOperandPosition(0, 0);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = (div_round_up(size, xChunk) + div_round_up(size, zChunk)) * div_round_up(size, yChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {zChunk, size, 1, 1});
}

TEST_F(GEMMInterleaveTest, shared_operand_on_different_position)
{
    unsigned size = 1.6*1024;
    unsigned xChunk = 96, yChunk = 256, zChunk = 256;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk+size)*2*sizeof(float));
    setSharedOperandPosition(0, 1);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = div_round_up(size, xChunk)  * div_round_up(size, yChunk) + div_round_up(size, zChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {zChunk, size, 1, 1});
}

TEST_F(GEMMInterleaveTest, shared_operand_on_same_position_and_transposed)
{
    unsigned size = 1.6*1024;
    unsigned xChunk = 64, yChunk = 128, zChunk = 128;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk+size)*2*sizeof(float));
    setSharedOperandPosition(0, 0);
    transposeSharedInput(0, 0);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = div_round_up(size, xChunk)  * div_round_up(size, yChunk) + div_round_up(size, zChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {size, zChunk, 1, 1});
}

TEST_F(GEMMInterleaveTest, shared_operand_on_same_position_trivial_non_shared_operands)
{
    // this test mimix the first resnet50 downsample-conv1 stitching ,
    // the non-shared operands on master and slave are trivially sliced, and the shared operand is sliced on the non-common dim
    // expectation is that the execution order will be interleaved - gemm1,gemm2,gemm1,gemm2 etc..
    unsigned commonDimSize = 64;
    unsigned sharedOperandW = 200704;
    SizeArray inputX  = {commonDimSize, commonDimSize, 1, 1};
    SizeArray inputY  = {commonDimSize, sharedOperandW, 1, 1}; // shared
    SizeArray outputP = {commonDimSize, sharedOperandW, 1, 1};
    SizeArray inputZ  = {2 * commonDimSize, commonDimSize, 1, 1};
    SizeArray outputQ = {2 * commonDimSize, sharedOperandW, 1, 1};
    setSharedOperandPosition(0, 0);

    createTensors(inputX, inputY, outputP, inputZ, outputQ);
    addNodesToGraph(getGraph());
    pBundle bundle;
    pMmeSlicingStrategy strategy = makeStrategy(commonDimSize, commonDimSize, sharedOperandW / 7, bundle);
    findAndAddCandidate(bundle, strategy);
    BundleSlicer::sliceBundle(*bundle, getGraph());
    unsigned lastNodeIndex = 0; // gemm 1 or 2, first gemm will determine the start index
    for (auto& node : getGraph().getExeSortedNodes())
    {
        if (node->getNodeType() == Node::TYPE_GEMM)
        {
            unsigned currentNodeIndex = stoi(node->getNodeName().substr(4,5));
            ASSERT_NE(lastNodeIndex, currentNodeIndex) << "Incorrect execution order in node - " << node->getNodeName();
            lastNodeIndex = currentNodeIndex;
        }
    }
}

TEST_F(GEMMInterleaveTest, shared_operand_on_different_position_and_transposed)
{
    unsigned size = 1.6*1024;
    unsigned xChunk = 64, yChunk = 256, zChunk = 64;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk)*2*sizeof(float));
    setSharedOperandPosition(0, 1);
    transposeSharedInput(0, 0);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = (div_round_up(size, xChunk) + div_round_up(size, zChunk)) * div_round_up(size, yChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {size, zChunk, 1, 1});
}



TEST_F(ConvInterleaveTest, shared_operand_on_same_position)
{
    // the slave operation should be the same then the master.
    unsigned size = 1.6*1024;
    unsigned xChunk = 64, yChunk = 256, zChunk = 64;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk)*2*sizeof(float));
    setSharedOperandPosition(0, 0);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = (div_round_up(size, xChunk) + div_round_up(size, zChunk)) * div_round_up(size, yChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {zChunk, size, 1, 1});
}

TEST_F(ConvInterleaveTest, shared_operand_on_different_position)
{
    unsigned size = 1.6*1024;
    unsigned xChunk = 96, yChunk = 256, zChunk = 256;
    GCFG_SRAM_SLICER_MAX_CAPACITY_BYTES.setValue(size*(xChunk+yChunk+zChunk+size)*2*sizeof(float));
    setSharedOperandPosition(0, 1);
    Solution solution = runTest(size, xChunk, yChunk, zChunk);
    unsigned numOfOps = div_round_up(size, xChunk)  * div_round_up(size, yChunk) + div_round_up(size, zChunk);
    checkSolutionSize(solution,5, numOfOps);
    checkChunkSize(solution, m_inputZ, {zChunk, size, 1, 1});
}

TEST_F(GEMMInterleaveTest, two_gemms_with_identical_inputs)
{
    // Scenario is 2 gemms with the same inputs inputX and input Y are both shared between the 2 nodes.
    // Expectation is that the nodes will be stitched and the master will have less slices than the slave.
    // This means that the execution order will be master,slave.master.slave until the master ran out of operations ,
    // and then the slave will continue;
    SizeArray inputX  = {4096, 4096, 1, 1};
    SizeArray inputY  = {512, 4096, 1, 1};
    SizeArray output = {512, 4096, 1, 1};
    setType(syn_type_bf16);
    createTensors(inputX, inputY, output, inputX, output);

    pNode gemm1 = NodeFactory::createNode({m_inputX, m_inputY}, {m_outputP},
                                          getParams(0), getParamsSize(),
                                          getFirstNodeGuid(), getFirstNodeName());
    pNode gemm2 = NodeFactory::createNode({m_inputX, m_inputY}, {m_outputQ},
                                           getParams(1), getParamsSize(),
                                           getSecondNodeGuid(), getSecondNodeName());
    GraphEditor::addNode(getGraph(), gemm1);
    GraphEditor::addNode(getGraph(), gemm2);
    sliceGraphToSRAMCapacity(getGraph());
    ASSERT_EQ(gemm1->getNodeAnnotation().bundleInfo->bundleIndex, gemm2->getNodeAnnotation().bundleInfo->bundleIndex);
    // check execution order
    uint32_t masterCount = 0;
    uint32_t slaveCount = 0;
    for (auto& node : getGraph().getExeSortedNodes())
    {
        if(node->getNodeType() == Node::TYPE_GEMM)
        {
            std::string origNodeName = node->getNodeName().substr(0, std::strlen(getFirstNodeName()));
            bool isMasterNode = (origNodeName == getSecondNodeName()); // Expanding from last to first, so expect master to be the second GEMM.
            bool isSlaveNode = (origNodeName == getFirstNodeName());
            ASSERT_NE(isMasterNode, isSlaveNode) << "Expect GEMM node to be either the master or the slave";
            masterCount += isMasterNode ? 1 : 0;
            slaveCount += isSlaveNode? 1 : 0;
        }
    }
    ASSERT_LT(masterCount, slaveCount) << "Expected less master nodes than slave nodes.";
}

} //namespace gaudi
