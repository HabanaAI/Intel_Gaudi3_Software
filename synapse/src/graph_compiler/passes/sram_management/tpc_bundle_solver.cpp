#include "tpc_bundle_solver.h"
#include "habana_nodes.h"

TpcBundleSolver::TpcBundleSolver(const HalReader& halReader, const pBundle& bundle) : Solver(halReader, bundle) {}

bool TpcBundleSolver::effectiveForBundle()
{
    TPCScalarPipeSolver scalarPipeSolver(m_halReader, getBundle());
    if (getBundle()->getNodes().size() == 1 && scalarPipeSolver.effectiveForBundle())
    {
        // Not handling scalar pipe nodes
        return false;
    }
    for (const NodePtr& node : getBundle()->getNodes())
    {
        if (!HabanaGraph::runsOnTPC(node) && std::dynamic_pointer_cast<ReshapeNode>(node) == nullptr)
        {
            return false;
        }
    }

    return true;
}

bool TpcBundleSolver::checkIfDuplicatesExist(const std::vector<pSlicedOperand>& bundleTensors) const
{
    using Hasher                  = std::function<size_t(const pSlicedOperand& operand)>;
    using Comparator              = std::function<bool(const pSlicedOperand& lhs, const pSlicedOperand& rhs)>;
    Hasher hasherByOriginalTensor = [](const pSlicedOperand& operand) {
        return std::hash<pTensor> {}(operand->originalTensor);
    };

    Comparator compByOriginalTensor = [](const pSlicedOperand& lhs, const pSlicedOperand& rhs) {
        return lhs->originalTensor == rhs->originalTensor;
    };
    using SlicedOperandSet = std::unordered_set<pSlicedOperand, Hasher, Comparator>;
    SlicedOperandSet bundleTensorsSet(bundleTensors.begin(),
                                      bundleTensors.end(),
                                      0,
                                      hasherByOriginalTensor,
                                      compByOriginalTensor);
    return bundleTensorsSet.size() < bundleTensors.size();
}

void TpcBundleSolver::removeDuplicatesIfExist(std::vector<pSlicedOperand>& bundleTensors) const
{
    // First, check for duplicates, complexity: O(n).
    if (checkIfDuplicatesExist(bundleTensors))
    {
        // Remove duplicates from bundlesTensors, complexity: O(n^2).
        std::vector<pSlicedOperand> uniqueBundleTensors;
        for (const auto& bundleOperand : bundleTensors)
        {
            if (std::none_of(uniqueBundleTensors.begin(),
                             uniqueBundleTensors.end(),
                             [&bundleOperand](const pSlicedOperand& op) {
                                 return bundleOperand->originalTensor == op->originalTensor;
                             }))
            {
                uniqueBundleTensors.push_back(bundleOperand);
            }
        }
        bundleTensors = uniqueBundleTensors;
    }
}

void TpcBundleSolver::createAllStrategies()
{
    SLC_DEBUG("TPC bundle solver: applying on bundle {}", getBundle()->getName());
    const NodeList& finalNodes = getFinalNode();
    auto finalNodeIter = finalNodes.begin();
    const NodePtr& masterNode = *finalNodeIter;
    SlicingStrategyPtr strategy      = SlicingStrategy::createStrategy(m_halReader, masterNode);
    auto& slicingData = strategy->getSlicingData();

    removeDuplicatesIfExist(slicingData.bundleTensors);

    // Create all operands
    m_tensorToOperand.insert(std::make_pair(slicingData.masterOperand->originalTensor, slicingData.masterOperand));
    for (auto operand : slicingData.bundleTensors)
    {
        m_tensorToOperand.insert(std::make_pair(operand->originalTensor, operand));
    }

    const auto& bundleNodes = getBundle()->getNodes();
    for (const auto& node : bundleNodes)
    {
        std::list<pSlicedOperand> ipnutOperands = addTensorsOperands(node->getInputs(),
                                                                     slicingData);

        std::list<pSlicedOperand> outputOperands = addTensorsOperands(node->getOutputs(),
                                                                      slicingData);

        // Map all outputs - The connected tensor to the next node can be not the first one.
        for (auto outOperand : outputOperands)
        {
            if (HabanaGraph::runsOnTPC(node))
            {
                    slicingData.setOperandSliceBackwardMapping(outOperand,
                                TrivialSliceMapper::mapOutputToInputs(node, ipnutOperands, outOperand));

            }
            else
            {
                HB_ASSERT(std::dynamic_pointer_cast<ReshapeNode>(node) != nullptr, "Expected reshape node beside of TPC in TPC bundle");
                slicingData.setOperandSliceBackwardMapping(outOperand,
                                        ReshapeSliceMapper::mapOutputToInput(ipnutOperands.front(), outOperand));
            }
        }
    }
    // Add all parallel TPC nodes as slave
    for (++finalNodeIter; finalNodeIter != finalNodes.end(); ++finalNodeIter)
    {
        slicingData.addSlaveTraversalPattern(m_tensorToOperand[(*finalNodeIter)->getOutput(0)]);
    }

    addStrategy(strategy);
}

std::list<pSlicedOperand> TpcBundleSolver::addTensorsOperands(const TensorVector& tensors, StrategySlicingData& slicingData)
{
    std::list<pSlicedOperand> addedOperands;
    for (const TensorPtr& t : tensors)
    {
        auto operandIter = m_tensorToOperand.find(t);
        if (operandIter == m_tensorToOperand.end())
        {
            slicingData.bundleTensors.push_back(std::make_shared<SlicedOperand>(t));
            operandIter = m_tensorToOperand.insert(std::make_pair(t, slicingData.bundleTensors.back())).first;
        }
        addedOperands.push_back(operandIter->second);
    }

    return addedOperands;
}

NodeList TpcBundleSolver::getFinalNode()
{
    Graph graph;
    for (const auto& node : getBundle()->getNodes())
    {
        graph.addNode(node);
    }
    NodeList finalNodes;
    // Iterate over the nodes in the bundle according to the insertion order,
    // assuming the master TPC will be first and the parallel nodes will be last.
    for (const NodePtr& node : getBundle()->getNodes())
    {
        if (!graph.hasConsumer(*node))
        {
            finalNodes.push_back(node);
        }
    }
    return finalNodes;
}

// Return pair of dimension and divider
Settable<std::pair<uint32_t, uint32_t>> TpcBundleSolver::getDimAndChunk(const TensorPtr&           tensor,
                                                                        const StrategySlicingData& mmeSlicingData)
{
    Settable<std::pair<uint32_t, uint32_t>> ret;
    auto                                    slicedOperandIter = std::find_if(mmeSlicingData.bundleTensors.begin(),
                                          mmeSlicingData.bundleTensors.end(),
                                          [&tensor](const pSlicedOperand& op) { return op->originalTensor == tensor; });
    if (slicedOperandIter != mmeSlicingData.bundleTensors.end())
    {
        const pSlicedOperand& mmeOperand = *slicedOperandIter;
        const DimVector&      slicedDims = SlicedOperandUtils::getSlicedDims(mmeOperand);
        if (! slicedDims.empty())
        {
            // Slice the last dim sliced by the MME brain
            uint32_t dim = slicedDims.back();
            ret.set(std::make_pair(dim,
                                   mmeOperand->chunkDimensions[dim]));
        }
    }

    return ret;
}
