#include "brain_data.h"
#include "bundle_scheduler_interfaces.h"
#include "bundle_view.h"
#include "slicer/bundle_views_collector.h"
#include "gaudi2_graph.h"
#include "habana_graph.h"
#include "math_utils.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "tensor_view_node.h"
#include "platform/gaudi2/graph_compiler/passes.h"
#include "tpc_slicing_test_infra.h"
#include "bundle_scheduler.h"
#include "bundle_scheduler_routes_creator.h"
#include "bundle_scheduler_bvd_traversal_sorter.h"
#include "bundle_scheduler_slices_sequencer.h"
#include "bundle_scheduler_threads_creator.h"
#include "bundle_scheduler_threads_scheduler.h"
#include "graph_visualization.h"
#include "brain_conf.h"
#include <iterator>

using namespace gc::layered_brain;

const unsigned c_bundleIndex = 0;

class SlicedGraphCreator : public GraphOptimizerTest
{
protected:
    using TensorSize = std::vector<TSize>;
    // The internal vector is a single slice nodes route, the external vector holds multiple sliced chains
    using SlicedChains = std::vector<NodesRoute>;

    NodePtr createGemmNode(const std::array<TensorPtr, 2>& inputs,
                           const TensorPtr&                out,
                           unsigned                        nodeIdx    = 0,
                           bool                            slicedNode = false)
    {
        DimVector slicedDims;
        if (slicedNode)
        {
            slicedDims = {m_sliceDim};
        }
        const TensorPtr& input0 = getGemmOperand(inputs[0], 0, true, slicedDims);
        const TensorPtr& input1 = getGemmOperand(inputs[1], 1, true, slicedDims);
        const TensorPtr& output = getGemmOperand(out, 0, false, slicedDims);

        synGEMMParams gemmParams {};
        std::string   nodeName = slicedNode ? "gemm_tile_" : "gemm_";
        nodeName               = nodeName + std::to_string(nodeIdx);
        NodePtr gemm =
            NodeFactory::createNode({input0, input1}, {output}, &gemmParams, NodeFactory::gemmNodeTypeName, nodeName);
        return gemm;
    }

    TensorPtr getGemmOperand(const TensorPtr& preallocatedTensor, unsigned index, bool isInput, DimVector slicedDims)
    {
        if (preallocatedTensor) return preallocatedTensor;

        // if t is not given, create it
        unsigned           numSlices = slicedDims.empty() ? m_numSlices : 1;
        std::vector<TSize> sliceSize = isInput ? m_gemmInputSize[index] : m_gemmOutputSize;
        for (auto dim : slicedDims)
        {
            // assume all slices are the same size
            HB_ASSERT(sliceSize[dim] % numSlices == 0, "invalid num slices");
            // set the tensors sizes to num slices
            sliceSize[dim] = div_round_up(sliceSize[dim], numSlices);
        }
        return TensorPtr(new Tensor(sliceSize.size(), sliceSize.data(), syn_type_float));
    }

    NodeVector addGemmNodeTiles(const NodePtr&      bigGemm,
                                const TensorVector& input0Slices,
                                const TensorVector& input1Slices,
                                const TensorVector& outputSlices)
    {
        NodeVector ret;
        // create gemm slices
        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            std::array<TensorPtr, 2> inputs;
            inputs[0]                    = input0Slices.empty() ? nullptr : input0Slices.at(sliceIdx);
            inputs[1]                    = input1Slices.empty() ? nullptr : input1Slices.at(sliceIdx);
            const TensorPtr& outputSlice = outputSlices.empty() ? nullptr : outputSlices.at(sliceIdx);

            NodePtr slicedGemm                          = createGemmNode(inputs, outputSlice, sliceIdx, true);
            slicedGemm->getNodeAnnotation().origBigNode = bigGemm;
            HB_ASSERT(GraphEditor::addNode(m_graph, slicedGemm), "failed to add node to graph");
            ret.push_back(slicedGemm);
            addSlicedNodeToBundle(slicedGemm);
        }
        return ret;
    }

    NodePtr createTPCNode(const TensorPtr& input, TensorSize sizes, std::string name)
    {
        TPCCustomIndexSpaceNode::Params nodeParams {};
        for (Dim dim = 0; dim < sizes.size(); dim++)
        {
            unsigned granularity  = 1;
            int      inputOverlap = 0;
            nodeParams.dims.emplace_back(sizes[dim], granularity, inputOverlap);
        }
        nodeParams.transpose = false;
        nodeParams.name      = name;

        NodePtr node = TPCCustomIndexSpaceNode::create(nodeParams, input, nullptr);
        return node;
    }

    NodeVector
    createTpcChain(unsigned chainLength, TensorSize sizes, std::string chainName, const TensorPtr& firstInput = nullptr)
    {
        NodeVector tpcChain;
        TensorPtr  connectingTensor = firstInput;
        for (auto i = 0; i < chainLength; i++)
        {
            std::string name = chainName + "_t" + std::to_string(i);
            NodePtr     node = createTPCNode(connectingTensor, sizes, name);
            connectingTensor = node->getOutput(0);
            tpcChain.push_back(node);
            addUnslicedNodeToBundle(node);
        }
        return tpcChain;
    }

    NodeVector addTpcNodeTiles(const NodePtr& bigTpc, const TensorVector& inputSlices, TensorSize sizes)
    {
        NodeVector ret;
        // create tpc node slices
        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            const TensorPtr& inputSlice                = inputSlices.empty() ? nullptr : inputSlices.at(sliceIdx);
            std::string      name                      = bigTpc->getNodeName() + "_tile_" + std::to_string(sliceIdx);
            NodePtr          slicedTpc                 = createTPCNode(inputSlice, sizes, name);
            slicedTpc->getNodeAnnotation().origBigNode = bigTpc;
            HB_ASSERT(GraphEditor::addNode(m_graph, slicedTpc), "failed to add node to graph");
            ret.push_back(slicedTpc);
            addSlicedNodeToBundle(slicedTpc);
        }
        return ret;
    }

    // The internal vector is a single slice nodes route, the external vector holds multiple sliced chains
    SlicedChains addTpcChainSlicedNodes(const NodeVector&   tpcChain,
                                        TensorSize          bigTensorSizes,
                                        const TensorVector& firstInputSlices = {})
    {
        SlicedChains ret(m_numSlices);
        TensorSize   sliceSizes = bigTensorSizes;
        HB_ASSERT(sliceSizes[m_sliceDim] % m_numSlices == 0,
                  "num slices should divide the dim size: dim size {}, num slices {}",
                  sliceSizes[m_sliceDim],
                  m_numSlices);
        sliceSizes[m_sliceDim] = div_round_up(sliceSizes[m_sliceDim], m_numSlices);
        TensorVector connectingTensors = firstInputSlices;
        for (auto chainNodeIdx = 0; chainNodeIdx < tpcChain.size(); chainNodeIdx++)
        {
            NodeVector nodeTiles = addTpcNodeTiles(tpcChain[chainNodeIdx], connectingTensors, sliceSizes);
            connectingTensors.clear();
            for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
            {
                connectingTensors.push_back(nodeTiles.at(sliceIdx)->getOutput(0));
                ret.at(sliceIdx).push_back(nodeTiles.at(sliceIdx));
            }
        }
        return ret;
    }

    NodePtr addForkNode(const TensorPtr& input, const TensorVector& outputs, std::string name)
    {
        auto tensorView = std::make_shared<TensorViewNode>(input, true, name + "_fork");
        for (const TensorPtr& t : outputs)
        {
            tensorView->addView(t, SizeVector(t->getDim(), 0));
        }
        HB_ASSERT(GraphEditor::addNode(m_graph, tensorView), "failed to add node to graph");
        addSlicedNodeToBundle(tensorView);
        return tensorView;
    }

    NodePtr addJoinNode(const TensorVector& inputs, const TensorPtr& output, std::string name)
    {
        auto tensorView = std::make_shared<TensorViewNode>(output, false, name + "_join");
        for (const TensorPtr& t : inputs)
        {
            tensorView->addView(t, SizeVector(t->getDim(), 0));
        }
        HB_ASSERT(GraphEditor::addNode(m_graph, tensorView), "failed to add node to graph");
        addSlicedNodeToBundle(tensorView);
        return tensorView;
    }

    NodePtr addReductionNode(const TensorVector& inputs, const TensorPtr& output)
    {
        ReductionOperation reductionOperation = ReductionOperation::REDUCTION_ADD;

        NodePtr reduction = NodeFactory::createNode(inputs,
                                                    {output},
                                                    &reductionOperation,
                                                    NodeFactory::reductionNodeTypeName,
                                                    "reduction");
        HB_ASSERT(GraphEditor::addNode(m_graph, reduction), "failed to add node to graph");
        addSlicedNodeToBundle(reduction);
        m_reductionNodes.insert(reduction);
        return reduction;
    }

    void addSlicedNodeToBundle(NodePtr node)  // by value to accept TensorViewNode without shared_from_this()
    {
        BundleInfo bundleInfo {c_bundleIndex, BundleType::UNDEFINED};
        node->getNodeAnnotation().bundleInfo = bundleInfo;
        m_bundleNodes.push_back(node);
    }

    void addUnslicedNodeToBundle(const NodePtr& node) { m_bundleUnslicedNodes.push_back(node); }

    std::vector<TensorPtr> getOutputBPTs()
    {
        std::vector<TensorPtr> outputBPTs;
        for (const auto& join : getJoins())
        {
            outputBPTs.push_back(join->getOutput(0));
        }
        return outputBPTs;
    }

    std::vector<TensorPtr> getReductionsOutputs()
    {
        std::vector<TensorPtr> outputs;
        for (const auto& reduction : getReductions())
        {
            outputs.push_back(reduction->getOutput(0));
        }
        return outputs;
    }

    std::vector<NodePtr> getRouteEnds()
    {
        std::vector<NodePtr> routeEnds;
        const auto&          joins(getJoins());
        const auto&          reductions(getReductions());
        routeEnds.reserve(joins.size() + reductions.size());
        std::copy(reductions.begin(), reductions.end(), std::back_inserter(routeEnds));
        std::copy(joins.begin(), joins.end(), std::back_inserter(routeEnds));
        return routeEnds;
    }

    std::map<TensorPtr, DimVector> getSlicedDimsPerOutputBPT(DimVector outputSlicedDims)
    {
        // assuming all bundle output BPTs are sliced on the given dims
        std::map<TensorPtr, DimVector> dimPerOutputBPT;
        for (auto bpt : getOutputBPTs())
        {
            dimPerOutputBPT.emplace(bpt, outputSlicedDims);
        }
        return dimPerOutputBPT;
    }

    TensorSet getOutputSlices()
    {
        TensorSet outputSlices;
        for (const auto& join : getJoins())
        {
            const auto& inputs = join->getInputs();
            outputSlices.insert(inputs.begin(), inputs.end());
        }
        return outputSlices;
    }

    NodeSet getJoins()
    {
        NodeSet joins;
        for (const auto& node : m_bundleNodes)
        {
            if (Node::isJoinNode(node))
            {
                joins.insert(node);
            }
        }
        return joins;
    }

    NodeSet getReductions()
    {
        // todo - can we detect them instead of saving?
        return m_reductionNodes;
    }

    // Returns:
    // TensorPtr - the chain output, which is the input to the big MME node
    // TensorVector - the sliced chains outputs, which are the inputs to the sliced MME nodes
    std::tuple<TensorPtr, TensorVector> createProducerChain(unsigned inputIdx, unsigned length)
    {
        // Create the big unsliced nodes, not adding to graph
        std::string name            = "producer" + std::to_string(inputIdx);
        m_producersChains[inputIdx] = createTpcChain(length, m_gemmInputSize[inputIdx], name);
        const TensorPtr& mmeInput   = m_producersChains[inputIdx].back()->getOutput(0);
        EXPECT_TRUE(mmeInput);
        // Create the sliced nodes and add them to the graph
        m_slicedProducersChains[inputIdx] =
            addTpcChainSlicedNodes(m_producersChains[inputIdx], m_gemmInputSize[inputIdx]);
        TensorVector inputSlices;
        TensorVector forkOutputs;
        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            inputSlices.push_back(m_slicedProducersChains[inputIdx].at(sliceIdx).back()->getOutput(0));
            forkOutputs.push_back(m_slicedProducersChains[inputIdx].at(sliceIdx).front()->getInput(0));
        }
        // Create the fork for the input BPT
        addForkNode(m_producersChains[inputIdx].front()->getInput(0), forkOutputs, name);

        return std::make_tuple(mmeInput, inputSlices);
    }

    // Returns:
    // TensorPtr - the chain output, which is the input to the big MME node
    // TensorVector - the sliced chains outputs, which are the inputs to the sliced MME nodes
    std::tuple<TensorPtr, TensorVector>
    createConsumerChain(unsigned length, const TensorPtr& firstInput, const TensorVector& firstInputSlices)
    {
        // Create the big unsliced nodes, not adding to graph
        unsigned    consumerIdx   = m_consumerChains.size();
        std::string name          = "consumer" + std::to_string(consumerIdx);
        NodeVector  consumerChain = createTpcChain(length, m_gemmOutputSize, name, firstInput);
        m_consumerChains.push_back(consumerChain);
        const TensorPtr& mmeOutput = consumerChain.front()->getInput(0);
        EXPECT_TRUE(mmeOutput);
        if (firstInput)
        {
            EXPECT_EQ(mmeOutput, firstInput);
        }
        // Create the sliced nodes and add them to the graph
        SlicedChains slicedConsumerChain = addTpcChainSlicedNodes(consumerChain, m_gemmOutputSize, firstInputSlices);
        m_slicedConsumerChains.push_back(slicedConsumerChain);
        TensorVector outputSlices;
        TensorVector joinInputs;
        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            outputSlices.push_back(slicedConsumerChain.at(sliceIdx).front()->getInput(0));
            joinInputs.push_back(slicedConsumerChain.at(sliceIdx).back()->getOutput(0));
        }

        // Create the join for the output BPT
        addJoinNode(joinInputs, consumerChain.back()->getOutput(0), name);

        return std::make_tuple(mmeOutput, outputSlices);
    }

protected:
    Gaudi2Graph m_graph;

    Dim                       m_sliceDim {3};
    uint64_t                  m_numSlices {4};
    std::array<TensorSize, 2> m_gemmInputSize;
    TensorSize                m_gemmOutputSize;

    NodeVector                     m_bundleNodes;
    NodeVector                     m_bundleUnslicedNodes;
    std::array<NodeVector, 2>      m_producersChains;
    std::vector<NodeVector>        m_consumerChains;
    std::array<SlicedChains, 2>    m_slicedProducersChains;
    std::vector<SlicedChains>      m_slicedConsumerChains;
    NodeSet                        m_reductionNodes;
};

class SingleMmeWithChainsSlicedGraph : public SlicedGraphCreator
{
protected:
    NodePtr    m_gemm;
    NodeVector m_slicedGemms;
    TSize      m_slicedDimSize = m_numSlices * 16;

    void createSlicedGraph(const std::array<unsigned, 2>& tpcProdChainLength,
                           const std::vector<unsigned>&   tpcConsChainLength)
    {
        TensorSize tensorSizes {128, 128, 5, 16};
        tensorSizes[m_sliceDim] = m_slicedDimSize;
        m_gemmInputSize         = {tensorSizes, tensorSizes};
        m_gemmOutputSize        = tensorSizes;

        std::array<TensorPtr, 2>    mmeInputs;
        std::array<TensorVector, 2> inputsSlices;
        TensorPtr                   mmeOutput;
        TensorVector                outputSlices;
        for (unsigned inputIdx = 0; inputIdx < tpcProdChainLength.size(); inputIdx++)
        {
            if (tpcProdChainLength[inputIdx] > 0)
            {
                std::tie(mmeInputs[inputIdx], inputsSlices[inputIdx]) =
                    createProducerChain(inputIdx, tpcProdChainLength[inputIdx]);
            }
        }
        for (unsigned consumerIdx = 0; consumerIdx < tpcConsChainLength.size(); consumerIdx++)
        {
            if (tpcConsChainLength[consumerIdx] == 0) continue;
            // mmeOutput and outputSlices are created by the first consumer chain, and same value is returne by the
            // subsequent calls
            std::tie(mmeOutput, outputSlices) =
                createConsumerChain(tpcConsChainLength[consumerIdx], mmeOutput, outputSlices);
        }

        // Create the big unsliced node, not adding to graph
        m_gemm = createGemmNode(mmeInputs, mmeOutput);
        addUnslicedNodeToBundle(m_gemm);

        // Create the sliced nodes and add them to the graph
        m_slicedGemms = addGemmNodeTiles(m_gemm, inputsSlices[0], inputsSlices[1], outputSlices);

        // Create a fork for MME inputs if there's no producers chain
        for (unsigned inputIdx = 0; inputIdx < tpcProdChainLength.size(); inputIdx++)
        {
            if (tpcProdChainLength[inputIdx] == 0)
            {
                addMmeInputAsBpt(inputIdx);
            }
        }
        // Add the MME output to BPTs to have multiple output BPTs
        addMmeOutputAsBpt();
        GraphVisualization::graphVisualizationOnDemand(m_graph, "before_test");
    }

    void addMmeInputAsBpt(unsigned inputIdx)
    {
        TensorVector inputSlices;
        for (const auto& n : m_slicedGemms)
        {
            inputSlices.push_back(n->getInput(inputIdx));
        }
        // Create the fork for the input BPT
        addForkNode(m_gemm->getInput(inputIdx), inputSlices, "gemm" + std::to_string(inputIdx));
    }

    void addMmeOutputAsBpt()
    {
        TensorPtr    mmeOutput = m_gemm->getOutput(0);
        TensorVector outputSlices;
        for (const auto& n : m_slicedGemms)
        {
            outputSlices.push_back(n->getOutput(0));
        }
        addJoinNode(outputSlices, mmeOutput, "gemm");
    }

    RoutesTable collectRoutes()
    {
        RoutesTable routesPerSlice;

        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            NodesRoute route;

            // producers chains
            for (unsigned prodIdx = 0; prodIdx < m_slicedProducersChains.size(); prodIdx++)
            {
                if (!m_slicedProducersChains[prodIdx].empty())
                {
                    const auto& prodChain = m_slicedProducersChains[prodIdx].at(sliceIdx);
                    route.insert(route.end(), prodChain.begin(), prodChain.end());
                }
            }
            // gemm
            route.push_back(m_slicedGemms.at(sliceIdx));
            const auto& gemmOutputBptSlice = m_slicedGemms.at(sliceIdx)->getOutput(0);
            // add gemm output BPT route
            NodesRoute gemmRoute(route.begin(), route.end());
            routesPerSlice[gemmOutputBptSlice].push_back(gemmRoute);

            // consumers chains
            for (unsigned consumerIdx = 0; consumerIdx < m_slicedConsumerChains.size(); consumerIdx++)
            {
                NodesRoute  routePerConsumer = route;
                const auto& consChain        = m_slicedConsumerChains[consumerIdx].at(sliceIdx);
                routePerConsumer.insert(routePerConsumer.end(), consChain.begin(), consChain.end());
                const auto& outputBptSlice = consChain.back()->getOutput(0);
                routesPerSlice[outputBptSlice].push_back(routePerConsumer);
            }
        }
        return routesPerSlice;
    }

    ThreadsSequence collectThreads()
    {
        ThreadsSequence threads;
        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            NodesThread thread;

            // producers chains
            for (unsigned prodIdx = 0; prodIdx < m_slicedProducersChains.size(); prodIdx++)
            {
                if (!m_slicedProducersChains[prodIdx].empty())
                {
                    const auto& prodChain = m_slicedProducersChains[prodIdx].at(sliceIdx);
                    thread.insert(thread.end(), prodChain.begin(), prodChain.end());
                }
            }

            // gemm
            thread.push_back(m_slicedGemms.at(sliceIdx));
            // consumers chains
            for (unsigned consumerIdx = 0; consumerIdx < m_slicedConsumerChains.size(); consumerIdx++)
            {
                const auto& consChain = m_slicedConsumerChains[consumerIdx].at(sliceIdx);
                thread.insert(thread.end(), consChain.begin(), consChain.end());
            }
            threads.push_back(thread);
        }
        return threads;
    }

    NodePtr getForkNode(unsigned mmeInputIdx)
    {
        TensorPtr forkOutput;  // find any of the fork outputs
        if (!m_slicedProducersChains[mmeInputIdx].empty())
        {
            // the fork node is the same for all slices - pick 0 randomly
            const auto& prodChain = m_slicedProducersChains[mmeInputIdx].at(0);
            forkOutput            = prodChain.front()->getInput(0);
        }
        else
        {
            // the fork is the gemm producer
            forkOutput = m_slicedGemms.at(0)->getInput(mmeInputIdx);
        }
        auto fork = m_graph.getTensorProducer(forkOutput);
        HB_ASSERT_PTR(fork);
        return fork;
    }

    std::vector<TensorVector> collectSliceSets()
    {
        std::vector<TensorVector> sliceSets;
        for (unsigned sliceIdx = 0; sliceIdx < m_numSlices; sliceIdx++)
        {
            TensorVector set;
            const auto&  gemmOutputBptSlice = m_slicedGemms.at(sliceIdx)->getOutput(0);
            set.push_back(gemmOutputBptSlice);
            for (unsigned consumerIdx = 0; consumerIdx < m_slicedConsumerChains.size(); consumerIdx++)
            {
                const auto& outputBptSlice = m_slicedConsumerChains[consumerIdx].at(sliceIdx).back()->getOutput(0);
                set.push_back(outputBptSlice);
            }
            sliceSets.push_back(set);
        }
        return sliceSets;
    }

    // return the last non join node in the thread
    NodePtr getThreadLastProducer(const NodesThread& thread)
    {
        unsigned nodeIndex = thread.size() - 1;
        NodePtr  lastNode  = thread[nodeIndex];
        if (Node::isJoinNode(lastNode))
        {
            // If the last node in the thread is join, skip it, return the prev node
            nodeIndex--;
            lastNode = thread[nodeIndex];
        }
        return lastNode;
    }

    // return the first non fork node in the thread
    NodePtr getThreadFirstConsumer(const NodesThread& thread)
    {
        unsigned nodeIndex = 0;
        NodePtr  firstNode = thread[nodeIndex];
        while (nodeIndex < thread.size() && Node::isForkNode(firstNode))
        {
            // If the last node in the thread is join, skip it, return the prev node
            nodeIndex++;
            firstNode = thread[nodeIndex];
        }
        HB_ASSERT(nodeIndex < thread.size(), "thread must start with a fork");
        return firstNode;
    }
};

// Subgraph that only fills the graph end points, for testing components which don't need the full graph
class OutputBPTsSubGraph : public SlicedGraphCreator
{
protected:
    std::vector<NodePtr> m_prodGemms;
    std::vector<NodePtr> m_consGemms;

    // Create the unsliced graph, only to get the BVDs related between outputs
    // Create a graph of numGemmsInBundleIn, whose outputs are consumed by numGemmsInBundleOut.
    // Mix the outputs consumers input index to get more BVDs in the bundle for more interesting testing.
    void createUnslicedGraph(unsigned numGemmsInBundleOut, unsigned numGemmsInBundleIn = 2)
    {
        TensorSize tensorSizes {128, 128, 5, 16};
        tensorSizes[m_sliceDim] *= m_numSlices;  // make sure the sizes can split to even slices
        m_gemmInputSize  = {tensorSizes, tensorSizes};
        m_gemmOutputSize = tensorSizes;

        for (unsigned gemmProdIdx = 0; gemmProdIdx < numGemmsInBundleIn; gemmProdIdx++)
        {
            NodePtr gemmProducer = createGemmNode(std::array<TensorPtr, 2>(), nullptr, gemmProdIdx);
            addUnslicedNodeToBundle(gemmProducer);
            m_prodGemms.push_back(gemmProducer);
        }

        for (unsigned gemmConsIdx = 0; gemmConsIdx < numGemmsInBundleOut; gemmConsIdx++)
        {
            std::array<TensorPtr, 2> mmeInputs;
            // Connect all the even consumer MME nodes to the first prod MME out through input 0
            // Connect all the odd consumer MME nodes to the second prod MME out through input 1
            // Create the same BVDs across MMEs which share input
            unsigned inputIdx   = gemmConsIdx % numGemmsInBundleIn;
            mmeInputs[inputIdx] = m_prodGemms[inputIdx]->getOutput(0);
            // set gemm index to be distinct in the unsliced graph - only influences the name
            NodePtr gemmConsumer = createGemmNode(mmeInputs, nullptr, numGemmsInBundleIn + gemmConsIdx);
            addUnslicedNodeToBundle(gemmConsumer);
            m_consGemms.push_back(gemmConsumer);
        }
    }

    // Create only the required parts of the sliced graph, which are required by the tested component -
    // The inputs which it requires, and the data it needs for initializing the layered brain bundle data.
    // This function creates the output slices and the join nodes which aggregate them.
    void createSlicedGraphJoins(const DimVector& slicedDims)
    {
        // create output slices for each MME output, and add the join node
        for (unsigned gemmIdx = 0; gemmIdx < m_consGemms.size(); gemmIdx++)
        {
            TensorVector slices;
            unsigned     numSlices = pow(m_numSlices, slicedDims.size());
            for (unsigned i = 0; i < numSlices; i++)
            {
                // create the gemm output slice
                TensorPtr outputSlice = getGemmOperand(nullptr, 0, false /*isInput*/, slicedDims);
                slices.push_back(outputSlice);
            }
            addJoinNode(slices, m_consGemms[gemmIdx]->getOutput(0), m_consGemms[gemmIdx]->getNodeName());
        }
    }

    // Create only the required parts of the sliced graph, which are required by the tested component -
    // The inputs which it requires, and the data it needs for initializing the layered brain bundle data.
    // This function creates the reduction input slices and the reduction nodes which aggregate them.
    void createSlicedGraphReductions(const DimVector& reducedMmeOutSlicedDims)
    {
        // Immitate slicing producer MME 0 on the common dim, in addition to its output slicing.
        // Create reduction of the producer MME 0 output slices.
        // All slices relate to the same big tensor
        const TensorPtr& bigTensor   = m_prodGemms.at(0)->getOutput(0);
        unsigned         numCDSlices = m_numSlices;
        // there are m_numSlices^numSlicedDims slices of the producer MME 0 output, each is an output of a reduction
        for (unsigned reductionIdx = 0; reductionIdx < pow(m_numSlices, reducedMmeOutSlicedDims.size()); reductionIdx++)
        {
            TensorVector reductionInputs;
            for (unsigned i = 0; i < numCDSlices; i++)
            {
                // create the reduction inputs - the gemm producer output operand slices
                TensorPtr reductionInput = getGemmOperand(nullptr, 0, false /*isInput*/, reducedMmeOutSlicedDims);
                reductionInput->getTensorAnnotation().origBigTensor = bigTensor;
                reductionInputs.push_back(reductionInput);
            }
            // create the reduction output - the gemm producer output 0
            TensorPtr reductionOutput = getGemmOperand(nullptr, 0, false /*isInput*/, reducedMmeOutSlicedDims);
            reductionOutput->getTensorAnnotation().origBigTensor = bigTensor;
            addReductionNode(reductionInputs, reductionOutput);
        }
    }
};

class LayeredBrainDataTestInfra
{
public:
    // slicedDimsPerBpt define the sliced BVDs before they have an index, as the BVDs are initialized by
    // initLayeredBrainData
    LayeredBrainDataTestInfra(HabanaGraph&                          graph,
                              NodeVector                            bundleUnslicedNodes,
                              const NodeSet&                        joinNodes,
                              const NodeSet&                        reductionNodes,
                              const std::map<TensorPtr, DimVector>& slicedDimsPerBigTensor,
                              unsigned                              numSlices)
    : m_graph(graph),
      m_bundleUnslicedNodes(bundleUnslicedNodes),
      m_joinNodes(joinNodes),
      m_reductionNodes(reductionNodes),
      m_slicedDimsPerBigTensor(slicedDimsPerBigTensor),
      m_numSlices(numSlices)
    {
        initLayeredBrainData();
    }

    LayeredBrainDataTestInfra(HabanaGraph&                          graph,
                              NodeVector                            bundleUnslicedNodes,
                              const std::map<TensorPtr, DimVector>& slicedDimsPerBigTensor,
                              unsigned                              numSlices)
    : LayeredBrainDataTestInfra(graph, bundleUnslicedNodes, {}, {}, slicedDimsPerBigTensor, numSlices)
    {
    }

    void addWalkPattern(const NodePtr& mmeNode, std::vector<BundleViewId> walkPattern)
    {
        auto*       lbData     = m_graph.getLayeredBrainData();
        BundleData& bundleData = lbData->m_bundleData.at(c_bundleIndex);
        StrategyPtr s          = bundleData.getFinalStrategy();

        s->getMmeSolution()->QORs.at(mmeNode)->solutionRequirements.walkDims = walkPattern;
        for (const auto& bvd : walkPattern)
        {
            s->setBVDMultiplier(bvd, BVDMultiplier(2 /* arbitrary multiplier to indicate bvd is sliced */));
        }
    }

    std::set<BundleViewId> getSlicedBvds(const TensorPtr& queriedTensor = nullptr) const
    {
        std::set<BundleViewId> slicedBvds;
        BundleData&            bundleData = m_graph.getLayeredBrainData()->m_bundleData.at(c_bundleIndex);
        for (const auto& [t, slicedDims] : m_slicedDimsPerBigTensor)
        {
            if (!queriedTensor || t == queriedTensor)
            {
                for (Dim dim : slicedDims)
                {
                    slicedBvds.insert(bundleData.getBundleViews()->getBVDForTensorDim(t, dim));
                }
            }
        }
        return slicedBvds;
    }

    BundleViewId getBvd(const TensorPtr& t, Dim dim) const
    {
        BundleData& bundleData = m_graph.getLayeredBrainData()->m_bundleData.at(c_bundleIndex);
        BundleViewId bvd        = bundleData.getBundleViews()->getBVDForTensorDim(t, dim);
        return bvd;
    }

    unsigned getNumBVDs() const
    {
        BundleData& bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];
        return bundleData.getBundleViews()->getNumOfBundleViews();
    }

    BVDCoord getSliceBvdCoord(const TensorPtr& slice) const { return m_bvdCoordPerSlice.at(slice); }

    std::vector<BVDCoord> getSlicedBvdsSubCoordsByOrder(BvdTraversalPattern slicedBvdsOrder) const
    {
        std::vector<BVDCoord> slicedBvdsSubCoords;
        BVDCoord              zeroCoord(slicedBvdsOrder.size(), 0);
        unsigned              numCoords = pow(m_numSlices, slicedBvdsOrder.size());
        // init all coords to 0
        for (unsigned coordIdx = 0; coordIdx < numCoords; coordIdx++)
        {
            slicedBvdsSubCoords.push_back(zeroCoord);
        }
        // fill 1 coord element at a time
        for (unsigned i = 0; i < slicedBvdsOrder.size(); i++)
        {
            unsigned counter = 0;
            unsigned jump    = pow(m_numSlices, i);
            for (unsigned coordIdx = 0; coordIdx < numCoords; coordIdx++)
            {
                if (coordIdx % jump == 0 && coordIdx != 0)
                {
                    counter++;
                }
                if (counter == m_numSlices)
                {
                    counter = 0;
                }
                slicedBvdsSubCoords[coordIdx][i] = counter;
            }
        }
        return slicedBvdsSubCoords;
    }

protected:
    void initLayeredBrainData()
    {
        m_graph.setLayeredBrainData(std::make_unique<gc::layered_brain::LayeredBrainData>());
        initBundleData();
    }

    void initBundleData()
    {
        initBundleViews();
        initBundleTensorsData();
        initBundleStrategy();
        initSlicedBVDs();
    }

    void initBundleViews()
    {
        BundleData& bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];

        BundleViewsCollector bvdCollector(m_bundleUnslicedNodes);
        TileSizePerTensor    granularityPerTensor;
        TileSizePerNode      granularityPerNode;
        for (const auto& node : m_bundleUnslicedNodes)
        {
            // Set dummy elementwise granularity for all nodes and tensors
            HB_ASSERT_PTR(node->getNodeAccessPattern());
            auto nodeTileRank        = node->getNodeAccessPattern()->getNodeResolution().size();
            granularityPerNode[node] = NodeTile::Geometry(nodeTileRank, 1);
            for (const auto& t : node->getOperands())
            {
                granularityPerTensor[t] = TensorTile::Geometry(t->getDim(), 1);
            }
        }
        BundleViewContainerPtr bvdContainer = bvdCollector.getAllBundleViews(granularityPerTensor, granularityPerNode);
        bundleData.setBundleViews(bvdContainer);
    }

    void initBundleTensorsData()
    {
        BundleData&  bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];

        for (const auto& join : m_joinNodes)
        {
            auto                   bigTensor = join->getOutput(0);
            std::vector<TensorPtr> slices {join->getInputs().begin(), join->getInputs().end()};

            auto joinInputsCoords = getBundleEndInputsCoordinates(slices, bigTensor, {});
            bundleData.addRouteEndInputsCoords(join, joinInputsCoords);
        }

        // Collect reductions big tensors - there might be a few reductions which create the same big tensor, if it's
        // sliced on additional dims. Append reductions inputs to a single long vector per same big output.
        std::map<TensorPtr, std::vector<TensorPtr>> slicesPerBigTensor;
        std::map<TensorPtr, std::vector<NodePtr>>   reductionsPerBigTensor;
        for (const auto& reduction : m_reductionNodes)
        {
            auto bigTensor = reduction->getOutput(0)->getTensorAnnotation().origBigTensor;
            HB_ASSERT_PTR(bigTensor);
            auto  retEmplaceSlices = slicesPerBigTensor.emplace(bigTensor, std::vector<TensorPtr>());
            auto& slices           = retEmplaceSlices.first->second;
            slices.insert(slices.end(), reduction->getInputs().begin(), reduction->getInputs().end());
            auto  retEmplaceNode = reductionsPerBigTensor.emplace(bigTensor, std::vector<NodePtr>());
            auto& nodes          = retEmplaceNode.first->second;
            nodes.push_back(reduction);
        }
        // Create the reductions big tensors slices coordinates together to make it simpler
        // add the reduced BVD from the GEMM input
        for (const auto& [bigTensor, slices] : slicesPerBigTensor)
        {
            // Create this big tensor slices coordinates together to make it simpler
            // add the reduced BVD from the GEMM input
            auto reducedBVD      = getReducedBVD(bigTensor);
            auto bigTensorCoords = getBundleEndInputsCoordinates(slices, bigTensor, {reducedBVD});

            // split the coordinates list between the different reductions, each  num reduction inputs
            // coordinates belong to a single reduction. since the reduced BVD is the first to be iterated, they coords
            // are ordered correctly to split by order
            unsigned coordIdx = 0;
            for (const auto& reduction : reductionsPerBigTensor.at(bigTensor))
            {
                unsigned numInputs = reduction->getNumInputs();
                HB_ASSERT(numInputs == m_numSlices, "unexpected num inputs for reduction");
                std::vector<BVDCoord> reductionInputsCoords {std::next(bigTensorCoords.begin(), coordIdx),
                                                             std::next(bigTensorCoords.begin(), coordIdx + numInputs)};
                bundleData.addRouteEndInputsCoords(reduction, reductionInputsCoords);
                coordIdx += numInputs;
            }
        }
    }

    std::vector<BVDCoord> getBundleEndInputsCoordinates(const std::vector<TensorPtr>& slices,
                                                        const TensorPtr&              bigTensor,
                                                        const BVDSet&                 reducedBVDs)
    {
        // build bvd coord per slice according to the sliced BVD
        std::vector<BVDCoord> inputsCoords;
        BundleData&           bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];
        // add the reduced BVDs in the beginning of the traversal pattern
        BvdTraversalPattern bvdsOrder {reducedBVDs.begin(), reducedBVDs.end()};
        // add the rest of the sliced BVDs by arbitrary order
        auto slicedBvds = getSlicedBvds(bigTensor);
        bvdsOrder.insert(bvdsOrder.end(), slicedBvds.begin(), slicedBvds.end());
        std::vector<BVDCoord> subCoords = getSlicedBvdsSubCoordsByOrder(bvdsOrder);
        HB_ASSERT(subCoords.size() == slices.size(),
                  "expected to get distinct coord per slice {} != {}",
                  subCoords.size(),
                  slices.size());
        unsigned numBvds = getNumBVDs();
        BVDCoord coord(numBvds, 0);
        unsigned subCoordIdx = 0;
        for (const auto& slice : slices)
        {
            // update the tensor related BVDs in the coordinate
            for (Dim tensorDim = 0; tensorDim < slice->getDim(); tensorDim++)
            {
                // get the BVD ID by the big tensor, as the BVD is created before slicing
                auto bvdId           = bundleData.getBundleViews()->getBVDForTensorDim(bigTensor, tensorDim);
                auto bvdIndexInOrder = index_of(bvdsOrder, bvdId);
                if (bvdIndexInOrder != -1)  // the BVD is found
                {
                    coord[bvdId] = subCoords[subCoordIdx][bvdIndexInOrder];
                }
            }
            // update the reduced BVDs in the coordinate
            for (auto bvdId : reducedBVDs)
            {
                auto bvdIndexInOrder = index_of(bvdsOrder, bvdId);
                HB_ASSERT(bvdIndexInOrder != -1, "reduced BVD must be found in the traversal order");
                {
                    coord[bvdId] = subCoords[subCoordIdx][bvdIndexInOrder];
                }
            }
            inputsCoords.push_back(coord);
            m_bvdCoordPerSlice.emplace(slice, coord);
            subCoordIdx++;
        }
        return inputsCoords;
    }

    BundleViewId getReducedBVD(const TensorPtr& bigMmeOutput)
    {
        BundleData& bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];
        // find the reduction MME big node producer
        for (const auto& n : m_bundleUnslicedNodes)
        {
            if (n->getOutput(0) == bigMmeOutput)
            {
                // get input 0 of this MME, to get the common dim BVD
                return bundleData.getBundleViews()->getBVDForTensorDim(n->getInput(0), 0);
            }
        }
        HB_ASSERT(false, "unexpectedly couldn't find reduction reduced common dim");
    }

    void initBundleStrategy()
    {
        BundleData&  bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];
        auto         mmeSolution = std::make_shared<MmeSolution>();
        for (const auto& node : m_bundleUnslicedNodes)
        {
            if (HabanaGraph::runsOnMME(node))
            {
                mmeSolution->QORs[node] = std::make_shared<SolutionParams>();
            }
        }
        StrategyPtr s = std::make_shared<Strategy>(mmeSolution);
        for (BundleViewId bvd = 0; bvd < bundleData.getBundleViews()->getNumOfBundleViews(); bvd++)
        {
            s->setBVDMultiplier(bvd, BVDMultiplier());
        }
        bundleData.setFinalStrategy(s);
    }

    void initSlicedBVDs()
    {
        auto slicedBvds = getSlicedBvds();
        HB_ASSERT(!slicedBvds.empty(), "all tests run with at least 1 sliced bvd");
        unsigned        numBvds = getNumBVDs();
        NumSlicesPerBVD numSlicesPerBvd(numBvds, 1);
        for (auto bvd : slicedBvds)
        {
            numSlicesPerBvd[bvd] = m_numSlices;
        }
        BundleData& bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];
        bundleData.setNumOfSlicesPerBVD(numSlicesPerBvd);
    }

    HabanaGraph&                   m_graph;
    NodeVector                     m_bundleUnslicedNodes;
    NodeSet                        m_joinNodes;
    NodeSet                        m_reductionNodes;
    std::map<TensorPtr, DimVector> m_slicedDimsPerBigTensor;
    std::map<TensorPtr, BVDCoord>  m_bvdCoordPerSlice;
    unsigned                       m_numSlices;
};

class SchedulerTestCommonValidation
{
public:
    explicit SchedulerTestCommonValidation(const NodeVector& bundleNodes) : m_bundleNodes(bundleNodes) {}

    void validateAllBundleNodesScheduled()
    {
        // check that all bundle nodes have a valid operation index
        for (const NodePtr& n : m_bundleNodes)
        {
            ASSERT_TRUE(n->getNodeAnnotation().bundleInfo->operationIndex > 0)
                << "node isn't scheduled " << n->getNodeName();
        }
    }

protected:
    const NodeVector& m_bundleNodes;
};

struct RoutesCreationTestParams
{
    std::array<unsigned, 2> tpcProdChainLength;
    std::vector<unsigned> tpcConsChainLength;
};

class BundleSchedulerRoutesCreatorTest
: public SingleMmeWithChainsSlicedGraph
, public testing::WithParamInterface<RoutesCreationTestParams>
{
public:
    void test()
    {
        createSlicedGraph(GetParam().tpcProdChainLength, GetParam().tpcConsChainLength);
        const auto routeEnds(getRouteEnds());

        DfsRoutesCreator routesCreator(m_graph, m_bundleNodes);
        RoutesTable      routesTable = routesCreator.getRoutesPerSlice(routeEnds);

        validateEachSliceHasRoutes(routesTable);
        validateAllNodesInRoute(routesTable);
        validateNodesOrderInRoute(routesTable);
        validateSimpleRoutesAreSequential(routesTable);
    }

protected:
    // validate all output slices have a route for each slice
    void validateEachSliceHasRoutes(const RoutesTable& routesTable)
    {
        auto outputSlices = getOutputSlices();
        for (const auto& t : outputSlices)
        {
            ASSERT_TRUE(routesTable.find(t) != routesTable.end()) << "missing route for slice";
        }
    }

    // validate all bundle nodes are included in any route
    void validateAllNodesInRoute(const RoutesTable& routesTable)
    {
        NodeSet nodesInRoute;
        // insert all routes into a set
        for (const auto& sliceAndRoutes : routesTable)
        {
            for (const NodesRoute& route : sliceAndRoutes.second)
            {
                for (const NodePtr& n : route)
                {
                    nodesInRoute.insert(n);
                }
            }
        }
        // check that all bundle nodes exist in this set
        for (const NodePtr& n : m_bundleNodes)
        {
            if (Node::isJoinNode(n) || Node::isForkNode(n)) continue;
            ASSERT_TRUE(nodesInRoute.find(n) != nodesInRoute.end())
                << "missing node bundle routes " << n->getNodeName();
        }
    }

    void validateNodesOrderInRoute(const RoutesTable& routesTable)
    {
        for (const auto& sliceAndRoutes : routesTable)
        {
            for (const NodesRoute& route : sliceAndRoutes.second)
            {
                for (unsigned nodeIdx = 0; nodeIdx < route.size(); nodeIdx++)
                {
                    for (unsigned prevNodeIdx = 0; prevNodeIdx < nodeIdx; prevNodeIdx++)
                    {
                        // validate each route node isn't an ancestor of any node before it in the route
                        ASSERT_FALSE(m_graph.isAncestor(route[nodeIdx], route[prevNodeIdx]))
                            << "route node depends on a later node in the route";
                    }
                }
            }
        }
    }

    void validateSimpleRoutesAreSequential(const RoutesTable& routesTable)
    {
        std::vector<SlicedChains> allSlicedChains;
        allSlicedChains.insert(allSlicedChains.end(), m_slicedProducersChains.begin(), m_slicedProducersChains.end());
        allSlicedChains.insert(allSlicedChains.end(), m_slicedConsumerChains.begin(), m_slicedConsumerChains.end());
        for (const SlicedChains& simpleMultiChains : allSlicedChains)
        {
            for (const NodesRoute& simpleChain : simpleMultiChains)
            {
                bool chainFound = false;
                // find the route which contains this simple chain
                const NodePtr& firstNode = simpleChain.front();
                for (const auto& tensorAndRoutes : routesTable)
                {
                    for (const auto& route : tensorAndRoutes.second)
                    {
                        auto nodeIt = std::find(route.begin(), route.end(), firstNode);
                        if (nodeIt != route.end())
                        {
                            chainFound = true;
                            // validate the chain is sequential in the route
                            for (const NodePtr& node : simpleChain)
                            {
                                ASSERT_EQ(node, *nodeIt);
                                nodeIt++;
                            }
                        }
                    }
                }
                ASSERT_TRUE(chainFound);
            }
        }
    }
};

TEST_P(BundleSchedulerRoutesCreatorTest, test_routes_collection)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(single_mme_with_chains,
                         BundleSchedulerRoutesCreatorTest,
                         ::testing::Values(RoutesCreationTestParams {{2, 1}, {3, 2, 1}},
                                           RoutesCreationTestParams {{3, 3}, {3}},
                                           RoutesCreationTestParams {{0, 3}, {1, 3}},
                                           RoutesCreationTestParams {{3, 0}, {3}},
                                           RoutesCreationTestParams {{3, 3}, {}},
                                           RoutesCreationTestParams {{3, 3}, {1, 3}},
                                           RoutesCreationTestParams {{1, 4}, {2}}));

struct BvdsSorterTestParams
{
    unsigned  numGemms;
    DimVector walkPatternIn0Dims;  // input0 sliced dims in walk pattern order
};

class BundleSchedulerBvdsSorterTest
: public OutputBPTsSubGraph
, public testing::WithParamInterface<BvdsSorterTestParams>
{
public:
    void test()
    {
        createUnslicedGraph(GetParam().numGemms);
        std::map<TensorPtr, DimVector> slicedDimsPerTensor = getSlicedDimsPerTensor();
        LayeredBrainDataTestInfra      lbDataTest(m_graph, m_bundleUnslicedNodes, slicedDimsPerTensor, m_numSlices);
        setMmeWalkPatterns(lbDataTest);

        auto*       lbData     = m_graph.getLayeredBrainData();
        BundleData& bundleData = lbData->m_bundleData.at(c_bundleIndex);

        SlicedBVDsTraversalSorter bvdSorter;
        BvdTraversalPattern       sortedBVDs = bvdSorter.getBundleViewsByTraversalOrder(bundleData);
        validateMmeWalkingPattern(sortedBVDs);
    }

protected:
    std::vector<SlicedBVDsTraversalSorter::BvdsOrderPreference> m_walkPatterns;

    std::map<TensorPtr, DimVector> getSlicedDimsPerTensor()
    {
        std::map<TensorPtr, DimVector> slicedDimsPerTensor;
        for (unsigned gemmIdx = 0; gemmIdx < GetParam().numGemms; gemmIdx++)
        {
            NodePtr   gemm       = m_consGemms[gemmIdx];
            DimVector slicedDims = GetParam().walkPatternIn0Dims;
            slicedDimsPerTensor.emplace(gemm->getInput(0), slicedDims);
            slicedDimsPerTensor.emplace(gemm->getOutput(0), m_sliceDim);
        }
        return slicedDimsPerTensor;
    }

    void setMmeWalkPatterns(LayeredBrainDataTestInfra& lbDataTest)
    {
        // fake slicing on the walk pattern dim (isn't reflected in the sliced nodes), to set the mme walk pattern
        for (unsigned gemmIdx = 0; gemmIdx < GetParam().numGemms; gemmIdx++)
        {
            NodePtr                   gemm = m_consGemms[gemmIdx];
            std::vector<BundleViewId> walkPattern;
            for (unsigned i = 0; i < GetParam().walkPatternIn0Dims.size(); i++)
            {
                BundleViewId bvd = lbDataTest.getBvd(gemm->getInput(0), GetParam().walkPatternIn0Dims[i]);
                walkPattern.push_back(bvd);
            }
            lbDataTest.addWalkPattern(gemm, walkPattern);

            std::set<BundleViewId> outputBVDs;
            for (Dim dim = 0; dim < gemm->getOutput(0)->getDim(); dim++)
            {
                BundleViewId outBvd = lbDataTest.getBvd(gemm->getOutput(0), dim);
                if (std::find(walkPattern.begin(), walkPattern.end(), outBvd) == walkPattern.end())
                {
                    outputBVDs.insert(outBvd);
                }
            }
            SlicedBVDsTraversalSorter::BvdsOrderPreference preference = {walkPattern, outputBVDs};
            m_walkPatterns.emplace_back(preference);
        }
    }

    void validateMmeWalkingPattern(BvdTraversalPattern sortedBVDs)
    {
        for (auto preference : m_walkPatterns)
        {
            unsigned lastBvdIndex = 0;
            // validate the walking pattern BVDs have increasing index in ordered BVDs
            for (BundleViewId bvd : preference.highPrioBvdsByOrder)
            {
                if (isSlicedBvd(bvd))
                {
                    size_t bvdOrderedIndex = index_of(sortedBVDs, bvd);
                    ASSERT_NE(bvdOrderedIndex, -1) << "sliced bvd wasn't found in order";
                    ASSERT_GE(bvdOrderedIndex, lastBvdIndex);
                    lastBvdIndex = bvdOrderedIndex;
                }
            }
            // validate the output BVDs, which are not in the walking pattern, have higher index than the last walking
            // pattern index
            for (BundleViewId bvd : preference.lowPrioBvds)
            {
                if (isSlicedBvd(bvd))
                {
                    unsigned bvdOrderedIndex = index_of(sortedBVDs, bvd);
                    ASSERT_NE(bvdOrderedIndex, -1) << "sliced bvd wasn't found in order";
                    ASSERT_GT(bvdOrderedIndex, lastBvdIndex);
                }
            }
        }
    }

    bool isSlicedBvd(BundleViewId bvd)
    {
        BundleData& bundleData = m_graph.getLayeredBrainData()->m_bundleData[c_bundleIndex];
        return bundleData.getNumOfSlicesPerBVD(bvd) > 1;
    }
};

TEST_P(BundleSchedulerBvdsSorterTest, test_bvds_sort)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(mme_walk_pattern,
                         BundleSchedulerBvdsSorterTest,
                         ::testing::Values(BvdsSorterTestParams {4, {0}},
                                           BvdsSorterTestParams {4, {1}},
                                           BvdsSorterTestParams {4, {2}},
                                           BvdsSorterTestParams {4, {3}},
                                           BvdsSorterTestParams {4, {0, 1}},
                                           BvdsSorterTestParams {4, {2, 1}},
                                           BvdsSorterTestParams {4, {3, 1, 2}},
                                           BvdsSorterTestParams {4, {3, 1, 0}},
                                           BvdsSorterTestParams {4, {1, 0, 2}},
                                           BvdsSorterTestParams {4, {2, 0, 1}}));

struct SlicesSequencerTestParams
{
    unsigned  numGemms;
    DimVector walkPatternOutputDims;  // mme output sliced dims in walk pattern order
};

class BundleSchedulerSlicesSequencerTest
: public OutputBPTsSubGraph
, public testing::WithParamInterface<SlicesSequencerTestParams>
{
public:
    void test()
    {
        createUnslicedGraph(GetParam().numGemms);
        createSlicedGraphJoins(GetParam().walkPatternOutputDims);
        auto slicingDimsPerBigTensor    = getSlicedDimsPerOutputBPT(GetParam().walkPatternOutputDims);
        auto slicingDimsPerIntermediate = getIntermediatesSlicedDims(GetParam().walkPatternOutputDims);
        slicingDimsPerBigTensor.merge(slicingDimsPerIntermediate);
        auto reducedMmeOutputSlicedDims = getReducedMmeOutputSlicingDims(slicingDimsPerBigTensor);
        createSlicedGraphReductions(reducedMmeOutputSlicedDims);
        const auto&               joins(getJoins());
        const auto&               reductions(getReductions());
        LayeredBrainDataTestInfra lbDataTest(m_graph,
                                             m_bundleUnslicedNodes,
                                             joins,
                                             reductions,
                                             slicingDimsPerBigTensor,
                                             m_numSlices);

        m_bvdsOrder = getBvdsTraversalOrder(lbDataTest);

        auto*                     lbData = m_graph.getLayeredBrainData();
        std::vector<NodePtr>      routeEndNodes(joins.begin(), joins.end());
        routeEndNodes.insert(routeEndNodes.end(), reductions.begin(), reductions.end());
        SimpleSlicesSequencer     slicesSeq;
        std::vector<TensorVector> sliceSets =
            slicesSeq.getSliceSetsSequence(m_bvdsOrder, lbData->m_bundleData.at(c_bundleIndex), {}, routeEndNodes);

        validateEachSliceIsInSingleSet(sliceSets);
        validateSliceSetBvdCoord(sliceSets, lbDataTest);
        validateSetsOrderByWalkPattern(sliceSets, lbDataTest);
        validateSetsSizes(sliceSets);
    }

protected:
    BvdTraversalPattern m_bvdsOrder;

    const std::map<TensorPtr, DimVector> getIntermediatesSlicedDims(const DimVector& mmeOutSlicedDims)
    {
        std::map<TensorPtr, DimVector> dimsPerBigTensor;
        // project the output slicing dims on the inputs
        for (const auto& consumerGemm : m_consGemms)
        {
            for (unsigned inputIdx = 0; inputIdx < consumerGemm->getNumInputs(); inputIdx++)
            {
                const auto& queriedTensor = consumerGemm->getInput(inputIdx);
                HB_ASSERT_PTR(queriedTensor);
                const auto& givenTensor = consumerGemm->getOutput(0);
                HB_ASSERT_PTR(givenTensor);
                const auto& accessPattern = consumerGemm->getNodeAccessPattern();
                HB_ASSERT_PTR(accessPattern);
                DimVector queriedSlicingDims;
                for (Dim givenSlicingDim : mmeOutSlicedDims)
                {
                    MultiDims matchingSlicingDims =
                        accessPattern->getTensorMatchingSlicedDims(queriedTensor, givenTensor, givenSlicingDim);
                    if (!matchingSlicingDims.empty())
                    {
                        HB_ASSERT(matchingSlicingDims.size() == 1,
                                  "gemm out dim mapping is expected to be to a single input dim");
                        queriedSlicingDims.push_back(matchingSlicingDims.front());
                    }
                }
                if (!queriedSlicingDims.empty())
                {
                    dimsPerBigTensor.emplace(queriedTensor, queriedSlicingDims);
                }
            }
        }
        // add MME 0 inputs common dim slicing so it will be detected as a sliced BVD
        dimsPerBigTensor.emplace(m_prodGemms.at(0)->getInput(0), DimVector {0});
        return dimsPerBigTensor;
    }

    DimVector getReducedMmeOutputSlicingDims(const std::map<TensorPtr, DimVector>& slicedDimsPerBigTensor)
    {
        auto it = slicedDimsPerBigTensor.find(m_prodGemms.at(0)->getOutput(0));
        if (it != slicedDimsPerBigTensor.end())
        {
            return it->second;
        }
        return {};
    }

    BvdTraversalPattern getBvdsTraversalOrder(LayeredBrainDataTestInfra& lbDataTest)
    {
        BvdTraversalPattern bvdsOrder;
        // iterate the MME output dims to get the corresponding BVDs by order
        for (unsigned i = 0; i < GetParam().numGemms; i++)
        {
            for (Dim dim : GetParam().walkPatternOutputDims)
            {
                BundleViewId bvd = lbDataTest.getBvd(m_consGemms[i]->getOutput(0), dim);
                if (std::find(bvdsOrder.begin(), bvdsOrder.end(), bvd) == bvdsOrder.end())
                {
                    bvdsOrder.push_back(bvd);
                }
            }
        }
        return bvdsOrder;
    }

    void validateEachSliceIsInSingleSet(const std::vector<TensorVector>& sliceSets)
    {
        std::map<TensorPtr, unsigned> counterPerSlice;
        ASSERT_TRUE(!sliceSets.empty());
        for (const auto& set : sliceSets)
        {
            for (const auto& slice : set)
            {
                if (counterPerSlice.find(slice) == counterPerSlice.end())
                {
                    counterPerSlice[slice] = 0;
                }
                counterPerSlice[slice]++;
            }
        }
        auto outputSlices = getOutputSlices();
        for (const auto& t : outputSlices)
        {
            ASSERT_TRUE(counterPerSlice.find(t) != counterPerSlice.end()) << "missing slice in sets " << t->getName();
            ASSERT_EQ(counterPerSlice[t], 1) << "slice appears more than once";
        }
    }

    void validateSliceSetBvdCoord(const std::vector<TensorVector>& sliceSets,
                                  const LayeredBrainDataTestInfra& lbDataTest)
    {
        for (const auto& set : sliceSets)
        {
            BVDCoord setCoord;
            for (const auto& slice : set)
            {
                if (setCoord.empty())
                {
                    // this is the first slice - set its coord
                    setCoord = lbDataTest.getSliceBvdCoord(slice);
                }
                else
                {
                    // validate the same coord for this slice
                    BVDCoord coord = lbDataTest.getSliceBvdCoord(slice);
                    ASSERT_EQ(coord, setCoord) << "slice coord doesn't match the set coord";
                }
            }
        }
    }

    void validateSetsOrderByWalkPattern(const std::vector<TensorVector>& sliceSets,
                                        const LayeredBrainDataTestInfra& lbDataTest)
    {
        // create a vector of the expected sliced coords order - num_slices values per advancing coord
        std::vector<BVDCoord> slicedBvdsSubCoords = lbDataTest.getSlicedBvdsSubCoordsByOrder(m_bvdsOrder);
        ASSERT_LE(sliceSets.size(), slicedBvdsSubCoords.size()) << "different coordinate expected per slices set";

        // make sure the BVDs are ordered according to the required pattern
        // check that the set coordinate is in ascending order of the manual coords. not all coords have to have a set
        unsigned lastBvdSubCoordIndex = 0;
        for (unsigned setIdx = 0; setIdx < sliceSets.size(); setIdx++)
        {
            // get the set coordinate from any of the set tensors
            BVDCoord coord = lbDataTest.getSliceBvdCoord(sliceSets[setIdx].front());
            BVDCoord subCoord(m_bvdsOrder.size(), 0);
            // create the sub coord for this set
            for (unsigned i = 0; i < m_bvdsOrder.size(); i++)
            {
                subCoord[i] = coord[m_bvdsOrder[i]];
            }
            // find the sub coord index
            unsigned currentBvdSubCoordIndex = index_of(slicedBvdsSubCoords, subCoord);
            // validate the sub coord index is higher than prev set subcoord index
            ASSERT_GE(currentBvdSubCoordIndex, lastBvdSubCoordIndex) << "sets are not orderd by the sliced BVD order";
        }
    }

    void validateSetsSizes(const std::vector<TensorVector>& sliceSets)
    {
        ASSERT_TRUE(!sliceSets.empty());
        if (getReductions().empty())
        {
            // validate the set for 0,0,..0 coordinate includes all big bpts = all joins outputs
            ASSERT_EQ(sliceSets.at(0).size(), getJoins().size());
        }
    }
};

TEST_P(BundleSchedulerSlicesSequencerTest, test_slices_sequence)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(multiple_mmes_output_bpts,
                         BundleSchedulerSlicesSequencerTest,
                         ::testing::Values(SlicesSequencerTestParams {1, {3}},
                                           SlicesSequencerTestParams {1, {3, 0}},
                                           SlicesSequencerTestParams {1, {1, 3, 0}},
                                           SlicesSequencerTestParams {2, {3}},
                                           SlicesSequencerTestParams {2, {1, 3}},
                                           SlicesSequencerTestParams {2, {1, 0, 2}},
                                           SlicesSequencerTestParams {3, {2}},
                                           SlicesSequencerTestParams {3, {1, 0}},
                                           SlicesSequencerTestParams {3, {1, 0, 2}},
                                           SlicesSequencerTestParams {4, {1}},
                                           SlicesSequencerTestParams {4, {1, 3}},
                                           SlicesSequencerTestParams {4, {1, 3, 2}}));

struct ThreadsCreatorTestParams
{
    std::array<unsigned, 2> tpcProdChainLength;
    std::vector<unsigned>   tpcConsChainLength;
};

class BundleSchedulerThreadsCreatorTest
: public SingleMmeWithChainsSlicedGraph
, public testing::WithParamInterface<ThreadsCreatorTestParams>
{
public:
    void test()
    {
        createSlicedGraph(GetParam().tpcProdChainLength, GetParam().tpcConsChainLength);

        std::vector<TensorVector> sliceSets      = collectSliceSets();
        RoutesTable               routesPerSlice = collectRoutes();

        SimpleThreadsCreator     threadsCreator;
        ThreadsSequence          threads = threadsCreator.getThreadsSequeceBySliceSets(sliceSets, routesPerSlice);

        validateThreadsCount(sliceSets, threads);
        validateThreadsOrder(sliceSets, threads);
        validateThreadsIncludeAllNodes(routesPerSlice, sliceSets, threads);
    }

    void validateThreadsCount(const std::vector<TensorVector>& sliceSets, const ThreadsSequence& threads)
    {
        ASSERT_EQ(sliceSets.size(), threads.size());
    }

    void validateThreadsOrder(const std::vector<TensorVector>& sliceSets, const ThreadsSequence& threads)
    {
        for (unsigned i = 0; i < threads.size(); i++)
        {
            TensorVector slices = sliceSets[i];
            NodesThread  thread = threads[i];
            NodePtr          lastProducer   = getThreadLastProducer(thread);
            const TensorPtr& outputBptSlice = lastProducer->getOutput(0);
            ASSERT_TRUE(std::find(slices.begin(), slices.end(), outputBptSlice) != slices.end())
                << "can't find output slice " << outputBptSlice->getName();
        }
    }

    void validateThreadsIncludeAllNodes(const RoutesTable&               routesPerSlice,
                                        const std::vector<TensorVector>& sliceSets,
                                        const ThreadsSequence&           threads)
    {
        for (unsigned i = 0; i < threads.size(); i++)
        {
            const TensorVector& slices = sliceSets[i];
            const NodesThread&  thread = threads[i];
            // for each output slice - validate the entire route is included in the merged route
            for (const auto& slice : slices)
            {
                for (const auto& route : routesPerSlice.at(slice))
                {
                    for (const auto& n : route)
                    {
                        ASSERT_TRUE(std::find(thread.begin(), thread.end(), n) != thread.end());
                    }
                }
            }
        }
    }
};

TEST_P(BundleSchedulerThreadsCreatorTest, test_threads_creation)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(single_mme_with_chains,
                         BundleSchedulerThreadsCreatorTest,
                         ::testing::Values(ThreadsCreatorTestParams {{2, 2}, {2, 3}},
                                           ThreadsCreatorTestParams {{1, 2}, {}},
                                           ThreadsCreatorTestParams {{4, 2}, {3, 1, 2}},
                                           ThreadsCreatorTestParams {{3, 2}, {2}},
                                           ThreadsCreatorTestParams {{0, 2}, {2, 3}},
                                           ThreadsCreatorTestParams {{1, 0}, {}},
                                           ThreadsCreatorTestParams {{0, 2}, {3, 1, 2}},
                                           ThreadsCreatorTestParams {{0, 0}, {2}}));

struct ThreadsSchedulerTestParams
{
    unsigned pipelineDepth;

    std::array<unsigned, 2> tpcProdChainLength;
    std::vector<unsigned>   tpcConsChainLength;
};

class BundleSchedulerThreadsSchedulerTest
: public SingleMmeWithChainsSlicedGraph
, public testing::WithParamInterface<ThreadsSchedulerTestParams>
{
public:
    void test()
    {
        m_numSlices = 8;
        createSlicedGraph(GetParam().tpcProdChainLength, GetParam().tpcConsChainLength);
        ThreadsSequence               threads    = collectThreads();
        std::vector<TensorPtr>        outputBPTs = getOutputBPTs();

        MultiThreadScheduler multiScheduler(m_graph, m_bundleNodes);
        multiScheduler.scheduleNodes(threads, getRouteEnds(), GetParam().pipelineDepth);

        validateSchedOrderBetweenPipelinedThreads(threads);
        SchedulerTestCommonValidation val(m_bundleNodes);
        val.validateAllBundleNodesScheduled();
        validateScheduleOrderPerThread(threads);
        validateForksSchedule();
        validateJoinsSchedule();
        validateThreadsNodesHaveThreadId(threads);
    }

    bool isSingleNodeThreads(const ThreadsSequence& threads)
    {
        return std::all_of(threads.begin(), threads.end(), [](const NodesThread t) { return t.size() == 1; });
    }

    void validateScheduleOrderPerThread(const ThreadsSequence& threads)
    {
        NodeSet scheduledNodes;
        for (const auto& thread : threads)
        {
            unsigned lastOpIndex = 0;
            // validate the operation index within a route is ascending, unless the node was scheduled before
            for (const NodePtr& n : thread)
            {
                if (Node::isForkNode(n)) continue;
                if (scheduledNodes.find(n) == scheduledNodes.end())
                {
                    unsigned currOpIndex = n->getNodeAnnotation().bundleInfo->operationIndex;
                    ASSERT_GT(currOpIndex, lastOpIndex) << "bad op index for node:" << n->getNodeName();
                    lastOpIndex = currOpIndex;
                    scheduledNodes.insert(n);
                }
            }
        }
    }

    void validateSchedOrderBetweenPipelinedThreads(const ThreadsSequence& threads)
    {
        // The test is invalid if all threads sizes are 1. The order of slices changes, which is acceptable
        if (isSingleNodeThreads(threads)) return;

        // all threads are the same. calc the basic thread length (without fork and join)
        // since they are the same length - every pipeline depth tuple of threads is perfectly interleaved
        unsigned threadLength = GetParam().tpcProdChainLength.at(0) + GetParam().tpcProdChainLength.at(1) + 1;
        for (auto consLength : GetParam().tpcConsChainLength)
        {
            threadLength += consLength;
        }
        // validate each thread starts with op index larger than the previous tuple of pipeline depth threads
        unsigned pipelineDepth = GetParam().pipelineDepth;
        unsigned lastOpIdx     = 0;
        for (unsigned threadIdx = 0; threadIdx < threads.size() - pipelineDepth; threadIdx += pipelineDepth)
        {
            unsigned maxForkConsumerOpIdx = 0;
            for (unsigned ppLevel = 0; ppLevel < pipelineDepth; ppLevel++)
            {
                NodePtr  forkConsumer      = getThreadFirstConsumer(threads.at(threadIdx + ppLevel));
                unsigned forkConsumerOpIdx = forkConsumer->getNodeAnnotation().bundleInfo->operationIndex;
                ASSERT_GE(forkConsumerOpIdx, lastOpIdx);
                maxForkConsumerOpIdx = std::max(maxForkConsumerOpIdx, forkConsumerOpIdx);
            }
            lastOpIdx = maxForkConsumerOpIdx + threadLength;
        }
    }

    void validateForksSchedule()
    {
        for (unsigned prodIdx = 0; prodIdx < m_slicedProducersChains.size(); prodIdx++)
        {
            const auto& fork      = getForkNode(prodIdx);
            unsigned    forkOpIdx = fork->getNodeAnnotation().bundleInfo->operationIndex;
            for (const auto& consumer : m_graph.getNodeRealConsumers(fork, Node::TENSOR_TYPE_DATA))
            {
                unsigned consumerOpIdx = consumer->getNodeAnnotation().bundleInfo->operationIndex;
                ASSERT_LT(forkOpIdx, consumerOpIdx);
            }
        }
    }

    void validateJoinsSchedule()
    {
        for (const auto& node : m_bundleNodes)
        {
            if (!Node::isJoinNode(node)) continue;

            unsigned joinOpIdx = node->getNodeAnnotation().bundleInfo->operationIndex;
            for (const auto& producer : m_graph.getNodeRealProducers(node, Node::TENSOR_TYPE_DATA))
            {
                unsigned producerOpIdx = producer->getNodeAnnotation().bundleInfo->operationIndex;
                ASSERT_GT(joinOpIdx, producerOpIdx);
            }
        }
    }

    void validateThreadsNodesHaveThreadId(const ThreadsSequence& threads)
    {
        for (unsigned threadIdx = 0; threadIdx < threads.size() - 1; threadIdx++)
        {
            // validate the operation index within a route is ascending, unless the node was scheduled before
            for (const NodePtr& n : threads[threadIdx])
            {
                std::optional<unsigned> threadIndex = n->getNodeAnnotation().bundleInfo->threadIndex;
                ASSERT_TRUE(threadIndex.has_value());
                ASSERT_LE(*threadIndex, threadIdx) << "bad thread index for node:" << n->getNodeName();
            }
        }
    }

    NodeSet getExpectedBlockedNodes(unsigned threadId)
    {
        NodeSet blockedNodes;
        for (unsigned i = 0; i < 2; i++)
        {
            if (GetParam().tpcProdChainLength[i] > 0)
            {
                EXPECT_LT(threadId, m_slicedProducersChains[i].size());
                blockedNodes.insert(m_slicedProducersChains[i].at(threadId).front());
            }
        }
        if (blockedNodes.empty())
        {
            // the mme is the blocked node
            EXPECT_LT(threadId, m_slicedGemms.size());
            blockedNodes.insert(m_slicedGemms[threadId]);
        }
        return blockedNodes;
    }

    NodeSet getExpectedBlockingNodes(unsigned threadId)
    {
        NodeSet blockingNodes;
        for (const SlicedChains& multiConsChains : m_slicedConsumerChains)
        {
            EXPECT_LT(threadId, multiConsChains.size());
            // get the last node of the consumer chain for thread
            blockingNodes.insert(multiConsChains.at(threadId).back());
        }
        if (blockingNodes.empty())
        {
            // the mme is the blocking node
            EXPECT_LT(threadId, m_slicedGemms.size());
            blockingNodes.insert(m_slicedGemms[threadId]);
        }
        return blockingNodes;
    }
};

TEST_P(BundleSchedulerThreadsSchedulerTest, test_threads_sched)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(single_mme_with_chains,
                         BundleSchedulerThreadsSchedulerTest,
                         ::testing::Values(ThreadsSchedulerTestParams {2, {2, 2}, {2, 3}},
                                           ThreadsSchedulerTestParams {2, {4, 2}, {3, 1, 2}},
                                           ThreadsSchedulerTestParams {2, {1, 2}, {}},
                                           ThreadsSchedulerTestParams {2, {3, 2}, {2}},
                                           ThreadsSchedulerTestParams {2, {0, 0}, {}},
                                           ThreadsSchedulerTestParams {2, {0, 3}, {}},
                                           ThreadsSchedulerTestParams {2, {1, 0}, {}},
                                           ThreadsSchedulerTestParams {1, {2, 2}, {2, 3}},
                                           ThreadsSchedulerTestParams {1, {0, 0}, {4, 1, 2}},
                                           ThreadsSchedulerTestParams {1, {1, 2}, {}},
                                           ThreadsSchedulerTestParams {1, {0, 0}, {}},
                                           ThreadsSchedulerTestParams {1, {0, 2}, {}},
                                           ThreadsSchedulerTestParams {3, {3, 0}, {2}},
                                           ThreadsSchedulerTestParams {3, {2, 2}, {2, 1}},
                                           ThreadsSchedulerTestParams {3, {0, 0}, {1, 4, 2}},
                                           ThreadsSchedulerTestParams {3, {3, 1}, {}},
                                           ThreadsSchedulerTestParams {3, {0, 0}, {}},
                                           ThreadsSchedulerTestParams {3, {0, 3}, {}},
                                           ThreadsSchedulerTestParams {3, {1, 0}, {2}}));

struct BundleSchedulerTestParams
{
    unsigned numSlices;

    std::array<unsigned, 2> tpcProdChainLength;
    std::vector<unsigned>   tpcConsChainLength;
};

class BundleSchedulerTest
: public SingleMmeWithChainsSlicedGraph
, public testing::WithParamInterface<BundleSchedulerTestParams>
{
public:
    void test()
    {
        m_numSlices = GetParam().numSlices;
        HB_ASSERT(m_numSlices > 0, "num slices must be > 0");
        m_slicedDimSize *= m_numSlices;
        createSlicedGraph(GetParam().tpcProdChainLength, GetParam().tpcConsChainLength);
        LayeredBrainDataTestInfra lbDataTest(m_graph,
                                             m_bundleUnslicedNodes,
                                             getJoins(),
                                             getReductions(),
                                             getSlicedDimsPerOutputBPT({m_sliceDim}),
                                             m_numSlices);

        ASSERT_TRUE(bundleNodesSchedule(m_graph));

        SchedulerTestCommonValidation val(m_bundleNodes);
        val.validateAllBundleNodesScheduled();
    }

protected:
    void SetUp() override
    {
        SingleMmeWithChainsSlicedGraph::SetUp();
        setGlobalConfForTest(GCFG_ENABLE_LAYERED_PIPELINE_BRAIN, "true");
    }
};

TEST_P(BundleSchedulerTest, test_scheduler_pass)
{
    test();
}

INSTANTIATE_TEST_SUITE_P(single_mme_with_chains,
                         BundleSchedulerTest,
                         ::testing::Values(BundleSchedulerTestParams {3, {2, 2}, {2, 3}},
                                           BundleSchedulerTestParams {3, {4, 2}, {3, 1, 2}},
                                           BundleSchedulerTestParams {3, {1, 2}, {}},
                                           BundleSchedulerTestParams {3, {3, 2}, {2}},
                                           BundleSchedulerTestParams {1, {2, 2}, {2, 3}},
                                           BundleSchedulerTestParams {1, {4, 2}, {3, 1, 2}},
                                           BundleSchedulerTestParams {1, {1, 2}, {}},
                                           BundleSchedulerTestParams {1, {3, 2}, {2}}));
