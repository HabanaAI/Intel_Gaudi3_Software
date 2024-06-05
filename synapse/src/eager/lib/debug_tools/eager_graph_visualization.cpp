#include "eager_graph_visualization.h"

// eager includes (relative to src/eager/lib/)
#include "eager_graph.h"

// synapse-internal includes (relative to src/)
#include "graph_compiler/passes/graph_visualization.h"

// std includes
#include <algorithm>
#include <string>
#include <utility>

using namespace std;

using eager_mode::EagerGraph;
using vizAttrMap = google::protobuf::Map<string, viz::Attribute>;

namespace
{
enum EagerTensorType
{
    GRAPH_ENTRY_TENSOR     = 0,
    ENTRY_TENSOR_WAS_ADDED = 1,
    INTERMEDIATE           = 2,
    GRAPH_OUTPUT           = 3
};

using EagerTensorInfo = std::pair<EagerTensorType, std::string>;
using TensorsInfoMap  = std::map<unsigned int, EagerTensorInfo>;

class EagerGraphVisualization final : public GraphVisualization
{
public:
    EagerGraphVisualization(std::string_view fileName,
                            std::string_view startTensor = "",
                            std::string_view endTensor   = "",
                            bool             fullInfo    = true)
    : GraphVisualization(fileName, startTensor, endTensor, fullInfo)
    {
    }

    static void graphVisualizationOnDemand(const EagerGraph& graph, std::string_view name)
    {
        LOG_INFO(GC, "create eager graph visualization for {}", name);
        (void)EagerGraphVisualization(name).createGraphVizOutput(graph);
    }

private:
    bool createGraphVizOutput(const EagerGraph& g) const;
    void
    HandleNodeVizProtobuf(const EagerGraph& g, NodePtr node, viz::Graph& vizGraph, TensorsInfoMap& tensorsInfo) const;
    TensorsInfoMap calcTensorInfoMap(const EagerGraph& g) const;
    void           insertProtobufInputTensors(NodePtr node, viz::Graph& vizGraph, TensorsInfoMap& tensorsInfo) const;
    void           insertProtobufNode(const EagerGraph& g,
                                      const NodePtr     node,
                                      viz::Graph&       vizGraph,
                                      TensorsInfoMap&   tensorsInfo) const;
    void           insertProtobufOutputTensors(const EagerGraph& g,
                                               NodePtr           node,
                                               viz::Graph&       vizGraph,
                                               TensorsInfoMap&   tensorsInfo) const;
    void           writeProtobufNodeInputs(const NodePtr node, TensorsInfoMap& tensorsInfo, VizNodePtr nodeToAdd) const;
    void           writeProtobufNodeOutputs(const NodePtr node, VizNodePtr nodeToAdd) const;
    void           writeProtobufNodeParamsAndLayout(const NodePtr node, VizNodePtr nodeToAdd) const;
    void           insertProtobufExitNode(const EagerGraph& g,
                                          viz::Graph&       vizGraph,
                                          const TensorPtr   tensor,
                                          TensorsInfoMap&   tensorsInfo) const;
};
}  // anonymous namespace

static bool nodeUsesTensor(const Node& n, const std::string& tensorName)
{
    for (const TensorVector* ts : {&n.getInputs(), &n.getOutputs()})
    {
        if (std::any_of(ts->begin(), ts->end(), [&](const TensorPtr& t) { return t && t->getName() == tensorName; }))
            return true;
    }
    return false;
}

bool EagerGraphVisualization::createGraphVizOutput(const EagerGraph& g) const
{
    if (m_mode != VISUALIZATION_MODE_TF_PB) return true;

    // map tesnor to the name of it's producer, if the tensor is graph input store null
    TensorsInfoMap tensorsInfo = calcTensorInfoMap(g);

    viz::Graph vizGraph;

    // Walk through all nodes
    bool startReached = m_startTensor.empty();
    for (const NodePtr& node : g.getExeSortedNodes())
    {
        HB_ASSERT_PTR(node);
        if (!m_endTensor.empty() && nodeUsesTensor(*node, m_endTensor)) break;

        startReached = startReached || nodeUsesTensor(*node, m_startTensor);
        if (startReached) HandleNodeVizProtobuf(g, node, vizGraph, tensorsInfo);
    }

    // Finished going over nodes, write results.
    g.invalidateExecutionSchedule();
    return formatAndWriteProtobufFile(vizGraph, g.getRecipeName());
}

void EagerGraphVisualization::HandleNodeVizProtobuf(const EagerGraph& g,
                                                    NodePtr           node,
                                                    viz::Graph&       vizGraph,
                                                    TensorsInfoMap&   tensorsInfo) const
{
    // handle the node's inputs which are also inputs for the entire graph
    insertProtobufInputTensors(node, vizGraph, tensorsInfo);

    // Add current node to protobuf
    insertProtobufNode(g, node, vizGraph, tensorsInfo);

    // handle the node's outputs which are also outputs of the entire graph
    insertProtobufOutputTensors(g, node, vizGraph, tensorsInfo);
}

TensorsInfoMap EagerGraphVisualization::calcTensorInfoMap(const EagerGraph& g) const
{
    TensorsInfoMap tensorsInfo;
    for (const NodePtr& node : g.getExeSortedNodes())
    {
        // Walk through node's inputs
        for (const TensorPtr& input : node->getInputs())
        {
            // skip over optional input
            if (input == nullptr) continue;

            auto [it, added] = tensorsInfo.try_emplace(input->getId(), GRAPH_ENTRY_TENSOR, "");
            if (!added && it->second.first == GRAPH_OUTPUT)
            {
                it->second = make_pair(INTERMEDIATE, tensorsInfo[input->getId()].second);
            }
        }

        // Walk through node's outputs
        for (const TensorPtr& output : node->getOutputs())
        {
            // skip over optional output
            if (output == nullptr) continue;

            // add tensor to map and start by assuming it's a graph output tensor
            // if later on this tensor is consumed by another node then that node will update tensor info
            tensorsInfo.try_emplace(output->getId(), GRAPH_OUTPUT, node->getNodeName());
        }
    }
    return tensorsInfo;
}

static std::string toStringZeroPadToLen(unsigned number, unsigned length)
{
    return fmt::format("{:0{}}", number, length);
}

void EagerGraphVisualization::writeProtobufNodeInputs(const NodePtr   node,
                                                      TensorsInfoMap& tensorsInfo,
                                                      VizNodePtr      nodeToAdd) const
{
    const TensorVector& inputs           = node->getInputs();
    const auto&         inputDataLayouts = node->getInputLayouts();
    bool                isExistsInputThatIsNotNull =
        std::any_of(inputs.begin(), inputs.end(), [](const TensorPtr& tensor) { return tensor != nullptr; });

    if (isExistsInputThatIsNotNull)
    {
        const unsigned numOfDigits = std::to_string(inputs.size() - 1).size();
        for (unsigned i = 0; i < inputs.size(); ++i)
        {
            if (inputs.at(i) != nullptr)
            {
                string producerName = tensorsInfo[inputs.at(i)->getId()].second;
                if (!producerName.empty())
                {
                    nodeToAdd->add_input(producerName);
                }
                else
                {
                    nodeToAdd->add_input(inputs.at(i)->getName());
                }
                std::string_view dataLayout;
                if (i < inputDataLayouts.size() && !inputDataLayouts[i].isDontCare())
                {
                    dataLayout = inputDataLayouts[i].toString();
                }
                addProtobufTensorInfo(nodeToAdd,
                                      inputs.at(i),
                                      toStringZeroPadToLen(i, numOfDigits),
                                      "input",
                                      dataLayout);
            }
            else
            {
                nodeToAdd->add_input("unused_port");
            }
        }
    }
}

void EagerGraphVisualization::writeProtobufNodeOutputs(const NodePtr node, VizNodePtr nodeToAdd) const
{
    const TensorVector& outputs           = node->getOutputs();
    const auto&         outputDataLayouts = node->getOutputLayouts();
    bool                isExistsOutputThatIsNotNull =
        std::any_of(outputs.begin(), outputs.end(), [](const TensorPtr& tensor) { return tensor != nullptr; });

    if (isExistsOutputThatIsNotNull)
    {
        const unsigned numOfDigits = std::to_string(outputs.size() - 1).size();
        for (unsigned i = 0; i < outputs.size(); ++i)
        {
            if (outputs.at(i) != nullptr)
            {
                std::string_view dataLayout;
                if (i < outputDataLayouts.size() && !outputDataLayouts[i].isDontCare())
                {
                    dataLayout = outputDataLayouts[i].toString();
                }
                addProtobufTensorInfo(nodeToAdd,
                                      outputs.at(i),
                                      toStringZeroPadToLen(i, numOfDigits),
                                      "output",
                                      dataLayout);
            }
        }
    }
}

void EagerGraphVisualization::writeProtobufNodeParamsAndLayout(const NodePtr node, VizNodePtr nodeToAdd) const
{
    // Add node parameters string
    std::string nodeParameters = node->getNodeParametersStr();
    vizAttrMap& attrs          = *nodeToAdd->mutable_attr();
    if (!nodeParameters.empty())
    {
        *attrs["Parameters"].mutable_s() = nodeParameters;
    }

    // Add supported layouts information
    auto addSupportedDataLayouts = [&attrs](const LayoutVector& supportedDataLayouts, const string& parameterName) {
        std::stringstream supportedInputDataLayoutsStream;
        bool              anyAdded = false;
        for (int i = 0; i < supportedDataLayouts.size(); i++)
        {
            if (supportedDataLayouts[i].isDontCare()) continue;
            supportedInputDataLayoutsStream << supportedDataLayouts[i].toString();
            if (i != supportedDataLayouts.size() - 1) supportedInputDataLayoutsStream << " | ";
            anyAdded = true;
        }
        if (anyAdded)
        {
            *attrs[parameterName].mutable_s() = supportedInputDataLayoutsStream.str();
        }
    };

    const auto& supportedInputDataLayouts = node->getInputSupportedLayouts();
    addSupportedDataLayouts(supportedInputDataLayouts, "supportedInputDataLayouts");
    const auto& supportedOutputDataLayouts = node->getOutputSupportedLayouts();
    addSupportedDataLayouts(supportedOutputDataLayouts, "supportedOutputDataLayouts");

    *attrs["Exec_idx"].mutable_s() = std::to_string(node->getExecutionOrderedIndex());
}

void EagerGraphVisualization::insertProtobufNode(const EagerGraph& g,
                                                 const NodePtr     node,
                                                 viz::Graph&       vizGraph,
                                                 TensorsInfoMap&   tensorsInfo) const
{
    HB_ASSERT_PTR(node);

    VizNodePtr nodeToAdd = vizGraph.add_node();
    nodeToAdd->set_op(renameNode(node->getNodeTypeStr()));
    nodeToAdd->set_name(node->getNodeName());

    writeProtobufNodeInputs(node, tensorsInfo, nodeToAdd);
    writeProtobufNodeOutputs(node, nodeToAdd);

    addProtoBuffWaitNodeInfo(nodeToAdd, g, node);

    writeProtobufNodeParamsAndLayout(node, nodeToAdd);
}

void EagerGraphVisualization::insertProtobufExitNode(const EagerGraph& g,
                                                     viz::Graph&       vizGraph,
                                                     const TensorPtr   tensor,
                                                     TensorsInfoMap&   tensorsInfo) const
{
    HB_ASSERT_PTR(tensor);
    VizNodePtr nodeToAdd = insertTensorAsPlaceholder(vizGraph, tensor);

    // Set producer node as input
    if (!tensorsInfo[tensor->getId()].second.empty())
    {
        nodeToAdd->add_input(tensorsInfo[tensor->getId()].second);
    }

    // For output Nodes: list name of tensor in node
    nodeToAdd->set_op("OutputTensor");
}

void EagerGraphVisualization::insertProtobufInputTensors(NodePtr         node,
                                                         viz::Graph&     vizGraph,
                                                         TensorsInfoMap& tensorsInfo) const
{
    // Walk through node's inputs
    for (TensorPtr input : node->getInputs())
    {
        // skip over optional input
        if (input == nullptr)
        {
            continue;
        }
        if (tensorsInfo[input->getId()].first == GRAPH_ENTRY_TENSOR)
        {
            // Add tensor to pb graph
            insertProtobufEntryNode(vizGraph, input);
            tensorsInfo[input->getId()].first = ENTRY_TENSOR_WAS_ADDED;
        }
    }
}

void EagerGraphVisualization::insertProtobufOutputTensors(const EagerGraph& g,
                                                          NodePtr           node,
                                                          viz::Graph&       vizGraph,
                                                          TensorsInfoMap&   tensorsInfo) const
{
    // Walk through node's outputs
    for (TensorPtr output : node->getOutputs())
    {
        // skip over optional output
        if (output == nullptr)
        {
            continue;
        }

        // Construct the graph exit node list
        if (tensorsInfo[output->getId()].first == GRAPH_OUTPUT)
        {
            insertProtobufExitNode(g, vizGraph, output, tensorsInfo);
        }
    }
}

namespace eager_mode
{
void visualizeGraph(const EagerGraph& graph, std::string_view name)
{
    EagerGraphVisualization::graphVisualizationOnDemand(graph, name);
}
}  // namespace eager_mode