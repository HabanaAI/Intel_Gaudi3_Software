#include "compilation_hal_reader.h"
#include "graph_traits.h"
#include "transpose_nodes_creator.h"
#include "transpose_utils.h"
#include "fcd_ops_utils.h"
#include "node_factory.h"
#include "dma_transpose_cost_model.h"
#include "transpose_splitter.h"
#include <memory>
#include <queue>

static inline bool isForbiddenNode(const std::vector<TransposeSplitter::Node>& forbiddenNodes,
                                   const TransposeSplitter::Node&              node)
{
    return std::find(forbiddenNodes.begin(), forbiddenNodes.end(), node) != forbiddenNodes.end();
}

TransposeSplitter::TransposeSplitter(const TransposeNode& transpose, const bool skipPerfReport)
: m_originalTranspose(transpose),
  m_dummyInput(transpose.getInput(0)->clone()),
  m_dummyOutput(transpose.getOutput(0)->clone()),
  m_skipPerfReport(skipPerfReport)
{
    HB_ASSERT_PTR(m_dummyInput);
    HB_ASSERT_PTR(m_dummyOutput);
}

// return true if node is valid
bool TransposeSplitter::Path::addTransposeToPath(const TensorPtr&                 input,
                                                 const TensorPtr&                 output,
                                                 const TransposePermutationArray& permutation,
                                                 bool                             intermidiatePath)
{
    HB_ASSERT_PTR(m_splitter);
    // logical transpose after logical transpose is redundant
    bool prevPermutationIsPhysical = m_chain.empty() || TransposeNode::isPhysicalTranspose(m_chain.back());
    if (!prevPermutationIsPhysical && !TransposeNode::isPhysicalTranspose(input, permutation)) return false;

    if (intermidiatePath && isForbiddenNode(m_forbiddenNodes, addPermutations(m_node, permutation))) return false;

    m_chain.push_back(TransposeNodeParams {input, output, permutation});
    auto& transpose = m_chain.back();  // TODO: llvm_smallvector .emplace_back doesn't return a reference to the item

    auto [it, added] = m_splitter->m_edgesMap.try_emplace({m_node, transpose.permutation}, 0);
    if (added)
    {
        const auto edgeCost = m_splitter->m_creator.getTransposeCostByParams(transpose);
        it->second          = edgeCost;
    }
    m_cost += it->second;

    m_node = addPermutations(m_node, transpose.permutation);
    m_forbiddenNodes.push_back(m_node);
    return true;
}

// create the one-to-one transpose that complete the transpose chain
std::optional<TransposeSplitter::Path> TransposeSplitter::createLastTranspose(const TensorPtr& input, Path path) const
{
    // calculate the last permutation by subtract the current permutation from the original permutation
    const auto& lastPermutation = subtractPermutations(m_originalTranspose.permutation(), path.node());
    bool        res             = path.addTransposeToPath(input, m_dummyOutput, lastPermutation, false);
    if (res) return path;
    return std::nullopt;
}

const TransposeSplitter::Path& TransposeSplitter::getBestSplittingHandler() const
{
    auto it = m_nodesMap.find(m_originalTranspose.permutation());
    HB_ASSERT(it != m_nodesMap.end(), "Best splitting never calculated");
    return it->second;
}

TransposeSplitter::Path& TransposeSplitter::getBestSplittingHandler()
{
    return const_cast<TransposeSplitter::Path&>(const_cast<const TransposeSplitter*>(this)->getBestSplittingHandler());
}

bool TransposeSplitter::shouldSkipNewPath(const std::optional<Path>& newPath) const
{
    // If the new transpose is logical after logical or lead to forbidden node, newPath will be empty.
    if (!newPath.has_value()) return true;
    auto it = m_nodesMap.find(newPath->node());
    // Check if there is cheaper path to the newPath node.
    if (it != m_nodesMap.end() && it->second <= newPath.value()) return true;
    // Check if the cost of the intermidiate path is bigger than the current best cost to the end or from the threshold.
    return (getBestSplittingHandler() <= newPath.value() || newPath->cost() >= m_costThreshold);
}

// create the one-to-one transpose that complete the transpose chain
std::optional<TransposeSplitter::Path>
TransposeSplitter::createIntermidiateTranspose(const TensorPtr&                 input,
                                               TransposeSplitter::Path          path,
                                               const TransposePermutationArray& permutation)
{
    TensorPtr output = getTensorAfterTranspose(*input, permutation);
    bool      res    = path.addTransposeToPath(input, output, permutation, true);
    if (res) return path;
    return std::nullopt;
}

TransposeSplitter::Path TransposeSplitter::Path::createEmptyPath(TransposeSplitter* splitter)
{
    Path emptyPath(splitter);
    emptyPath.m_node           = getIdentityPermutation(splitter->m_dummyInput->getDim());
    emptyPath.m_forbiddenNodes = {splitter->m_originalTranspose.permutation(), emptyPath.m_node};
    return emptyPath;
}

void TransposeSplitter::initializeState()
{
    // initiate the best splitting without split
    auto trivialSplitting               = Path::createEmptyPath(this);
    m_nodesMap[trivialSplitting.node()] = trivialSplitting;  // Save the empty path in the map.
    // Add the original transpose to empty path to get trivial splitting (no splitting).
    trivialSplitting.addTransposeToPath(m_dummyInput, m_dummyOutput, m_originalTranspose.permutation(), false);

    // Update the best splitting to the trivial splitting.
    m_nodesMap[m_originalTranspose.permutation()] = trivialSplitting;
    float factor                                  = 1.0 - GCFG_TRANSPOSE_SPLITTING_THRESHOLD.value();
    m_costThreshold                               = factor * (float)trivialSplitting.cost();
}

// Find the best spliting, based on the cost model.
// The technique is to define a graph where nodes are the permutations
// (in fact the state of the input after apply the permutation), and edges are the transpose node that needed to apply
// to move from one node to another (where the weights defined by the cost model).
// Therefore, by applying Dijkstra algorithm from the "identity" node at the end we will find the shortest path to
// each node, includes the path to the output, however, since we calculates weights in "lazy" way, we have some
// exit point to avoid unnecessary calculation. In addition we limits the length of the path to save compilation time.
void TransposeSplitter::calculateBestSplitting()
{
    // since we create a lot of transpose nodes we don't want to add them to the logs
    auto gcLogLevel         = synapse::LogManager::instance().get_log_level(synapse::LogManager::LogType::GC);
    auto habanaNodeLogLevel = synapse::LogManager::instance().get_log_level(synapse::LogManager::LogType::HABANA_NODE);
    if (gcLogLevel < 3 || habanaNodeLogLevel < 3 /* warn */)
    {
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::GC, 4 /* error */);
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::HABANA_NODE, 4 /* error */);
    }

    initializeState();
    Path& bestSplitting = getBestSplittingHandler();

    auto emptyPathIt = m_nodesMap.find(getIdentityPermutation(m_dummyInput->getDim()));
    HB_ASSERT(emptyPathIt != m_nodesMap.end(), "Empty path not initialized");
    // Since Path class has operator ">" which return the more expensive path,
    // "toVisit" is sorted from cheap to expensive, therefore it is equivalent to minimum heap.
    // Also, "toVisit" contains only intermidiate paths, the first K-1 transpose transitions in a
    // potential K-length chain (without the final transpose), so it initialized with an "empty" intermidiate path.
    std::priority_queue<Path, std::deque<Path>, std::greater<Path>> toVisit;
    toVisit.push(emptyPathIt->second);
    while (!toVisit.empty())
    {
        const Path current = toVisit.top();
        toVisit.pop();

        // We use minimum heap, so if the cheapest path is not cheap enough we can break.
        if (bestSplitting < current) break;

        auto input = current.chain().empty() ? m_dummyInput : current.chain().back().output;

        // At first create the permutation that complements the chain to the original permutation.
        auto fullPath = createLastTranspose(input, current);

        // Check if this spitting is better than the current best.
        if (fullPath.has_value() && fullPath.value() < bestSplitting)
        {
            bestSplitting = fullPath.value();
            // Since we use minimum heap, the cost of every path in "toVisit" is at least like the cost of
            // the current intermidiate path, so if the last transpose cost is zero (must be logical node),
            // then it is guaranteed that there is no other path which is cheaper.
            if (bestSplitting.cost() == current.cost()) break;
        }

        // We limit the length of the transpose chain.
        if (current.chain().size() == TransposeSplitter::MAX_LENGTH - 1) continue;

        // Move deeper to find better chain.
        auto nextPermutation = getIdentityPermutation(input->getDim());
        // Loop over all possible permutation except the identity permutation.
        while (std::next_permutation(nextPermutation.begin(), nextPermutation.end()))
        {
            auto newPath = createIntermidiateTranspose(input, current, nextPermutation);
            if (shouldSkipNewPath(newPath)) continue;
            m_nodesMap[newPath->node()] = newPath.value();
            // Update the map with the cheaper path, and insert it into "toVisit".
            toVisit.push(newPath.value());
        }
    }

    // restore the original log level
    if (gcLogLevel < 3 || habanaNodeLogLevel < 3)
    {
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::GC, gcLogLevel);
        synapse::LogManager::instance().set_log_level(synapse::LogManager::LogType::HABANA_NODE, habanaNodeLogLevel);
    }
}

std::pair<NodeVector, uint64_t> TransposeSplitter::splitTransposeViaCostModel()
{
    TensorPtr origInput  = m_originalTranspose.getInput(0);
    TensorPtr origOutput = m_originalTranspose.getOutput(0);

    if (origInput->getDim() > MAX_DIM_FOR_SPLITTING)
    {
        if (!m_skipPerfReport)
        {
            LOG_TRACE_AND_PERF(
                TRANSPOSE_SPLIT,
                "skip transpose split for {}, due to high rank transpose (rank: {}, max rank for split: {})",
                m_originalTranspose.getNodeName(),
                origInput->getDim(),
                MAX_DIM_FOR_SPLITTING);
        }
        return m_creator.getTransposeNodesAndCost(m_originalTranspose);
    }

    calculateBestSplitting();
    Path bestSplitting = getBestSplittingHandler();

    // If there is no spliting, or the best spliting cost is greater than threshold,
    // then extract nodes with the traditional extractor (without split).
    if (bestSplitting.cost() >= m_costThreshold)
    {
        if (!m_skipPerfReport)
        {
            const auto optimalCost = TransposeNode::isPhysicalTranspose(origInput, m_originalTranspose.permutation())
                                         ? FcdOpsUtils::getOptimalCost(*origInput)
                                         : 0;
            if (bestSplitting.cost() > optimalCost)
            {
                LOG_DEBUG_AND_PERF(
                    TRANSPOSE_SPLIT,
                    "there is no valid split for {} which is {} transpose, optimal cost: {}, actual cost {})",
                    m_originalTranspose.getNodeName(),
                    TransposeNode::isPhysicalTranspose(origInput, m_originalTranspose.permutation()) ? "physical"
                                                                                                     : "logical",
                    optimalCost,
                    bestSplitting.cost());
            }
        }
        return m_creator.getTransposeNodesAndCost(m_originalTranspose);
    }

    // Replace dummy tensors from the edges of best split chain with the original input and output
    bestSplitting.replaceInput(origInput);
    bestSplitting.replaceOutput(origOutput);

    LOG_DEBUG(TRANSPOSE_SPLIT,
              "{} with original permutation: [{}] split into {} transposes",
              m_originalTranspose.getNodeName(),
              toString(m_originalTranspose.permutation(), ','),
              bestSplitting.chain().size());

    if (LOG_LEVEL_AT_LEAST_TRACE(TRANSPOSE_SPLIT))
    {
        uint64_t originalCost = m_creator.getTransposeCostByParams(TransposeNodeParams::fromNode(m_originalTranspose));
        HB_ASSERT(originalCost != 0, "if original cost is zero, split is unexpected");
        const auto& bestCost    = bestSplitting.cost();
        const auto& optimalCost = FcdOpsUtils::getOptimalCost(*origInput);
        LOG_TRACE(TRANSPOSE_SPLIT,
                  "original cost: {} new cost: {} improvement: {}%",
                  (float)originalCost / optimalCost,
                  (float)bestCost / optimalCost,
                  std::round(100 * (1 - (float)bestCost / originalCost)));
    }
    // extract the final nodes of the split permutation with the traditional extractor
    // it is needed since we used dummy names for splitting
    NodeVector ret;
    for (unsigned i = 0; i < bestSplitting.chain().size(); ++i)
    {
        auto newTranspose     = bestSplitting.chain().at(i);
        newTranspose.nodeName = fmt::format("{}/{}", m_originalTranspose.getNodeName(), i);
        NodeVector newNodes   = m_creator.getTransposeNodesByParams(newTranspose);
        ret.insert(ret.end(), newNodes.begin(), newNodes.end());
        LOG_TRACE(TRANSPOSE_SPLIT,
                  "input sizes {}, ({}) permutation [{}], output sizes {}",
                  newTranspose.input->getDimSizesStr(),
                  TransposeNode::isPhysicalTranspose(newTranspose) ? "physical" : "logical",
                  toString(newTranspose.permutation, ','),
                  newTranspose.output->getDimSizesStr());
    }
    return {ret, bestSplitting.cost()};
}
