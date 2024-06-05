#include <memory>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <passes.h>
#include <node.h>
#include <tensor.h>
#include <sys/stat.h>
#include <unistd.h>
#include "layout.h"

#include "graph_visualization.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "graph_editor.h"
#include "protobuf/graph_visualization.pb.h"
#include <google/protobuf/text_format.h>
#include <unistd.h>
#include "data_type_utils.h"

using namespace std;

typedef google::protobuf::Map<string, viz::Attribute> vizAttrMap;

template <typename T>
static std::string toString(const T number, const unsigned length)
{
    std::stringstream ss;
    ss.width(length);
    ss.fill('0');
    ss << number;
    return ss.str();
}

namespace
{
// internal node to replace bundles
class BundleNode : public DebugNodeBase
{
public:

    unsigned m_bundleIndex;
    NodeList m_nodes;

    virtual std::string getNodeTypeStr() const override { return fmt::format("Bundle {}", m_bundleIndex); }

    BundleNode(unsigned bundleIndex)
    : DebugNodeBase({}, {}, fmt::format("bundle_{}", bundleIndex)), m_bundleIndex(bundleIndex), m_nodes({})
    {
    }
};

} // anonymous namespace

static bool graphVisualizationBundle(HabanaGraph& graph)
{
    HabanaGraphPtr bundleGraph = graph.clone(true); // clone the graph for edit
    // map from bundle index to new bundle node
    std::map<unsigned, NodePtr> bundlesMap;
    for (const NodePtr& node : bundleGraph->getExeSortedNodes())
    {
        Settable<BundleInfo> bundleInfo = node->getNodeAnnotation().bundleInfo;
        if (!bundleInfo.is_set()) continue; // not belong to bundle
        unsigned bundleIndex = bundleInfo->bundleIndex;
        if (bundlesMap.count(bundleIndex) == 0) // first occurance of node in this bundle index
        {
            bundlesMap[bundleIndex] = NodePtr(new BundleNode(bundleIndex));
        }
        const NodePtr& bundleNode = bundlesMap[bundleIndex];
        reinterpret_cast<BundleNode*>(bundleNode.get())->m_nodes.push_back(node); // add this node to bundle node
        for (const TensorPtr& input : node->getInputs()) // add inputs to bundle node
        {
            const NodePtr& producer = bundleGraph->getTensorProducer(input);
             // igonre internal tensors
            if (!producer || // graph input
                !producer->getNodeAnnotation().bundleInfo.is_set() || // not in bundle
                producer->getNodeAnnotation().bundleInfo->bundleIndex != bundleIndex) // in another bundle
            {
                bundleNode->addInput(bundleNode->getNumInputs(), input);
            }
        }
        for (const TensorPtr& output : node->getOutputs()) // add outputs to bundle node
        {
            const NodeList& consumers = bundleGraph->getTensorConsumers(output);
            // igonre internal tensors
            if (consumers.empty() || std::any_of(consumers.begin(), consumers.end(), [&](const NodePtr& consumer)
                {
                    return !consumer || // graph output
                           !consumer->getNodeAnnotation().bundleInfo.is_set() || // not in bundle
                           consumer->getNodeAnnotation().bundleInfo->bundleIndex != bundleIndex; // in another bundle
                }))
            {
                bundleNode->addOutput(output);
            }
        }
    }
    // replace bundles with the new node
    for (auto p : bundlesMap)
    {
        const NodePtr& bundleNode = p.second;
        const NodeList& originalNodes = reinterpret_cast<BundleNode*>(bundleNode.get())->m_nodes;
        GraphEditor::replaceNodes(*bundleGraph, originalNodes, {bundleNode});
    }
    GraphVisualization gv("BundleGraph");
    return gv.Apply(*bundleGraph);
}

bool graphVisualizationPre(HabanaGraph& graph)
{
    GraphVisualization gv("PreGraph");
    return gv.Apply(graph);
}

bool graphVisualizationPost(HabanaGraph& graph)
{
    if (GCFG_GRAPH_VISUALIZATION_COLLAPSE_BUNDLES.value())
    {
        graphVisualizationBundle(graph);
    }
    GraphVisualization gv("PostGraph");
    return gv.Apply(graph);
}

bool GraphVisualization::graphVisualizationPostOnFailure(HabanaGraph& graph)
{
    GraphVisualization gv("PostGraphFailed");
    return gv.Apply(graph);
}

bool GraphVisualization::graphVisualizationOnDemand(HabanaGraph& graph, const std::string& name)
{
    LOG_INFO(GC, "create graph visualization for {}", name);
    GraphVisualization gv(name);
    return gv.Apply(graph);
}

GraphVisualization::GraphVisualization(std::string_view fileName,
                                       std::string_view startTensor,
                                       std::string_view endTensor,
                                       bool             fullInfo)
: m_startTensor(startTensor.empty() ? GCFG_GRAPH_VISUALIZATION_START_TENSOR.value() : startTensor),
  m_endTensor(endTensor.empty() ? GCFG_GRAPH_VISUALIZATION_END_TENSOR.value() : endTensor),
  m_fileName(fileName),
  m_fullInfo(fullInfo),
  m_mode(getVisualizationMode())
{
    HB_ASSERT(m_fileName.empty() == false, "Unspecified file name");
}

GraphVisualization::GraphVisualization(const std::string& dirName, bool fullInfo)
: m_startTensor(GCFG_GRAPH_VISUALIZATION_START_TENSOR.value()), m_endTensor(GCFG_GRAPH_VISUALIZATION_END_TENSOR.value()),
m_mode(getVisualizationMode())
{
    HB_ASSERT(dirName.empty() == false, "Unspecified directory name");
    if (!createOutputDir(dirName))
    {
        m_fullInfo = false;
        return;
    }
    m_fullInfo = fullInfo;
}

bool GraphVisualization::Apply(HabanaGraph& g) const
{
    // Bail out if visualization compilation parameter was not turned on or the graph is empty
    if (!g.getVisualizationStatus() || g.isEmpty())
    {
        return true;
    }

    return createGraphVizOutput(g);
}

void GraphVisualization::setFileName(const std::string& fileName)
{
    m_fileName = fileName;
}

bool GraphVisualization::createOutputDir(const std::string& dirName)
{
    time_t rawtime;
    struct tm* timeinfo;
    char buffer[80];
    unsigned maxTries = 20;
    std::string fullPath;

    //create base dir may already exist
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr)
    {
        LOG_WARN(GC, "GraphVisualization: getcwd error {}", errno);
        return false;
    }

    std::string baseDirPath = cwd;
    baseDirPath += "/" + dirName + "/";
    if (mkdir(baseDirPath.c_str(), S_IRWXU) == -1)
    {
        if( errno != EEXIST ) // if folder already exists, everything is ok
        {
            LOG_WARN(GC, "GraphVisualization: mkdir error {}", errno);
            return false;
        }
    }

    //create subdir
    int iter = 1;
    std::string seqNumberStr = "";
    while (true)
    {
        time(&rawtime);
        timeinfo = localtime(&rawtime);
        strftime(buffer, sizeof(buffer),"%d-%m-%Y_%H:%M:%S",timeinfo);
        std::string timeStr(buffer);

        fullPath = baseDirPath + timeStr + seqNumberStr + "/";
        int res = mkdir(fullPath.c_str(), S_IRWXU);
        if (res)
        {
            if (iter == maxTries)
            {
                LOG_INFO(GC, "GraphVisualization: mkdir error {}", res);
                return false;
            }
            else
            {
                seqNumberStr = "_";
                seqNumberStr += std::to_string(iter);
            }
            iter++;
        }
        else
        {
            break;
        }
    }

    m_dirName = fullPath;
    std::string latestPath = baseDirPath + "latest";
    // for linux, create a symbolic link to latest log folder.
    // ignore return values / errors for non-existing link on the first time.
    int ignoredRetVal = unlink(latestPath.c_str());
    ignoredRetVal = symlink(fullPath.c_str(), latestPath.c_str());
    UNUSED(ignoredRetVal);
    return true;
}

VisualizationMode GraphVisualization::getVisualizationMode()
{
    return ((VisualizationMode)GCFG_VISUALIZATION_MODE.value());
}

bool GraphVisualization::createGraphVizOutput(HabanaGraph& g) const
{
    vector<string>              nodeRecords; // string record in json format for each node
    stringstream                argNodes;    // index list (as strings) into nodeRecords where graph inputs reside
    stringstream                exitNodes;   // index list (as strings) into nodeRecords where graph exit nodes reside
    map<unsigned, TensorInfo>   tensorInfoMap;
    bool                        startReached = false;
    bool                        endReached = false;

    viz::Graph vizGraph;
    unsigned pbRecordCount = 0;

    if (m_startTensor.empty())
    {
        startReached = true;
    }

    // Walk through all nodes
    for (const NodePtr& node : g.getExeSortedNodes())
    {

        HB_ASSERT_PTR(node);

        stringstream nodeInputs;

        if (endReached)
        {
            break;
        }
        if (!startReached)
        {
            for (const auto& tensor : node->getOperands())
            {
                if (tensor != nullptr && tensor->getName() == m_startTensor)
                {
                    startReached = true;
                    break;
                }
            }
            if (!startReached)
            {
                continue;
            }
        }
        if (!m_endTensor.empty())
        {
            for (const auto& tensor : node->getOperands())
            {
                if (tensor != nullptr && tensor->getName() == m_endTensor)
                {
                    endReached = true;
                    break;
                }
            }
        }

        if (!g.shouldVisualizeDmaNodes())
        {
            // For Goya, might want to Ignore DMA nodes at the graph entry and exit points
            if (node->isDma() && (node->getNumInputs() == 0 || node->getNumOutputs() == 0))
            {
                if (!m_fullInfo)
                {
                    continue;
                }
            }
        }

        if (m_mode == VISUALIZATION_MODE_TF_PB)
        {
            HandleNodeVizProtobuf(g, node, vizGraph, tensorInfoMap, pbRecordCount);
        }
        else
        {
            HandleNodeVizJson(g, node, argNodes, exitNodes, nodeRecords, tensorInfoMap);
        }
    }

    // Finished going over nodes, write results.
    if (m_mode == VISUALIZATION_MODE_TF_PB)
    {
        return formatAndWriteProtobufFile(vizGraph, g.getRecipeName());
    }
    else
    {
        return formatAndWriteJsonFile(nodeRecords, argNodes.str(), exitNodes.str(), g.getRecipeName());
    }

}

void GraphVisualization::HandleNodeVizJson(HabanaGraph& g, NodePtr node, stringstream& argNodes,
        stringstream& exitNodes, vector<string>& nodeRecords, map<unsigned, TensorInfo>& tensorInfoMap) const
{
    stringstream nodeInputs;

    // Walk through node's inputs
    for (TensorPtr input : node->getInputs())
    {
        // skip over optional input
        if (input == nullptr)
        {
            string prefix = nodeInputs.str().empty() ? "[" : ", [";
            nodeInputs << prefix << "-1,-1, 0]";
            continue;
        }

        if (tensorInfoMap.find(input->getId()) == tensorInfoMap.end())
        {
            // Handle graph entry nodes
            unsigned currentRecordIndex = insertRecord(nodeRecords, formatArgNodeRecord(input));
            tensorInfoMap[input->getId()] = {currentRecordIndex, 0};
            string prefix = argNodes.str().empty() ? "    " : ",\n    ";
            argNodes << prefix << currentRecordIndex;
        }

        // Collect node's inputs
        string prefix = nodeInputs.str().empty() ? "[" : ", [";
        nodeInputs << prefix
                    << tensorInfoMap[input->getId()].producerRecordIndex
                    << ", "
                    << tensorInfoMap[input->getId()].producerOutputPort
                    << ", 0]";
    }

    // Format a json record for the current node
    unsigned currentRecordIndex = insertRecord(nodeRecords, formatNodeRecord(g, node, nodeInputs.str()));

    // Walk through node's outputs
    unsigned outputIndex = 0;
    for (TensorPtr output : node->getOutputs())
    {
        // skip over optional output
        if (output == nullptr)
        {
            continue;
        }

        // Create an entry that holds the current (producer) node index in the json file
        // and the index of the specific output tensor among all node's outputs
        tensorInfoMap[output->getId()] = {currentRecordIndex, outputIndex};

        // Construct the graph exit node list
        if (g.isOutputTensor(output))
        {
            string prefix = exitNodes.str().empty() ? "[" : ", [";
            exitNodes << prefix << currentRecordIndex << ", " << outputIndex << ", 0]";
        }
        outputIndex++;
    }
}

// Insert a record to nodeRecords and return the record index
unsigned GraphVisualization::insertRecord(std::vector<std::string>& nodeRecords, std::string&& record) const
{
    nodeRecords.push_back(std::move(record));
    return nodeRecords.size() - 1;
}

string GraphVisualization::formatArgNodeRecord(const TensorPtr& tensor) const
{
    stringstream record;
    HB_ASSERT_PTR(tensor);
    record << "    {\n"
           << "      \"op\": \"null\",\n"
           << "      \"name\": \"" << tensor->getName() << "\",\n"
           << "      \"inputs\": []\n"
           << "    }";
    return record.str();
}

string GraphVisualization::formatNodeRecord(HabanaGraph& g, const NodePtr& node, const string& inputs) const
{
    stringstream    record;

    HB_ASSERT_PTR(node);

    const TensorVector& inputTensors = node->getInputs();
    const TensorVector& outputTensors = node->getOutputs();

    unsigned numInputsNotNull = std::count_if(inputTensors.begin(), inputTensors.end(),
                                              [](const TensorPtr& tensor) { return tensor != nullptr; });
    unsigned numOutputsNotNull = std::count_if(outputTensors.begin(), outputTensors.end(),
                                               [](const TensorPtr& tensor) { return tensor != nullptr; });

    record << "    {\n"
           << "      \"op\": \"" << renameNode(node->getNodeTypeStr()) << "\",\n"
           << "      \"name\": \"" << node->getNodeName() << "\",\n"
           << "      \"attrs\": {\n";
    if (numInputsNotNull)
    {
        for (unsigned i = 0; i < inputTensors.size(); ++i)
        {
            if (inputTensors.at(i) != nullptr)
            {
                record << "        \"input:" << i << "\": \""
                       << formatTensorInfo(inputTensors.at(i))
                       << "\"";
                string next = --numInputsNotNull ? ",\n" : "";
                record << next;
            }
        }
        record << ",\n";
    }
    if (numOutputsNotNull)
    {
        for (unsigned i = 0; i < outputTensors.size(); ++i)
        {
            if (outputTensors.at(i) != nullptr)
            {
                record << "        \"output_" << i << "\": \""
                       << formatTensorInfo(outputTensors.at(i))
                       << "\"";
                string next = --numOutputsNotNull ? ",\n" : "";
                record << next;
            }
        }
        record << ",\n";
    }

    addWaitNodeInfo(g, node, record);

    //Add node parameters string
    std::string nodeParameters = node->getNodeParametersStr();
    if (!nodeParameters.empty())
    {
        record << "        \"Parameters\": \"" << nodeParameters << "\",\n";
    }

    record << "        \"Bundle_idx\": \"";
    if (node->getNodeAnnotation().bundleInfo.is_set())
    {
        record << node->getNodeAnnotation().bundleInfo->bundleIndex;
        record << "\",\n        \"Op_idx\":\"" << node->getNodeAnnotation().bundleInfo->operationIndex;
    }
    else
    {
        record << "N/A";
    }
    record << "\",\n        \"Exec_idx\":\"" << node->getExecutionOrderedIndex() << "\"\n";
    record << "      },\n"
           << "      \"inputs\": [" << inputs << "]\n"
           << "    }";

    return record.str();
}

void GraphVisualization::addWaitNodeInfo(HabanaGraph& g, const NodePtr& node, stringstream& record) const
{
    bool     addWaitCycles = true;
    unsigned waitCycles;

    if (node->isWait())
    {
        std::shared_ptr<WaitNode> waitNode = std::dynamic_pointer_cast<WaitNode>(node);
        waitCycles = waitNode->getWaitCycles();

        NodeSet suspendedNodes = g.getBlockedNodes(waitNode);

        unsigned i = 0;
        unsigned suspendedNodesCount = suspendedNodes.size();
        for (NodePtr suspended : suspendedNodes)
        {
            record << "        \"Delay_Node_" << i << "\": \""
                   << suspended->getNodeName()
                   << "\"";
            string next = --suspendedNodesCount ? ",\n" : "";
            record << next;
            i++;
        }
        record << ",\n";
    }
    else if (node->getNodeAnnotation().waitCycles)
    {
        waitCycles = node->getNodeAnnotation().waitCycles;
    }
    else
    {
        addWaitCycles = false;
    }

    if (addWaitCycles)
    {
        record << "        \"Wait_Cycles"  << "\": \""
               << waitCycles
               << "\""
               << ",\n";
    }
}

void GraphVisualization::addProtoBuffWaitNodeInfo(VizNodePtr nodeToAdd, const HabanaGraph& g, const NodePtr& node) const
{
    bool        addWaitCycles = true;
    unsigned    waitCycles;
    vizAttrMap& attrs = *nodeToAdd->mutable_attr();

    if (node->isWait())
    {
        std::shared_ptr<WaitNode> waitNode = std::dynamic_pointer_cast<WaitNode>(node);
        HB_ASSERT_PTR(waitNode);
        waitCycles = waitNode->getWaitCycles();

        NodeSet suspendedNodes = g.getBlockedNodes(waitNode);

        unsigned i = 0;
        for (NodePtr suspended : suspendedNodes)
        {
            *attrs["Delay_Node_" + std::to_string(i)].mutable_s() = suspended->getNodeName();
            i++;
        }
    }
    else if (node->getNodeAnnotation().waitCycles)
    {
        waitCycles = node->getNodeAnnotation().waitCycles;
    }
    else
    {
        addWaitCycles = false;
    }

    if (addWaitCycles)
    {
        *attrs["Wait_Cycles"].mutable_s() = std::to_string(waitCycles);
    }
}

void GraphVisualization::addProtoBuffMmeNodeInfo(VizNodePtr nodeToAdd, const HabanaGraph& g, const NodePtr& node) const
{
    vizAttrMap& attrs   = *nodeToAdd->mutable_attr();
    MMENodePtr  mmeNode = std::dynamic_pointer_cast<MmeNode>(node);
    HB_ASSERT(mmeNode, "could not downcast Node to MME Node");
    const MmeExpBias& mmeExpBias     = mmeNode->getMmeExpBias();
    *attrs["fp8BiasIn"].mutable_s()  = std::to_string(mmeExpBias.fp8BiasIn[TENSOR_IFM]);
    *attrs["fp8BiasIn2"].mutable_s() = mmeExpBias.fp8BiasIn.size() > TENSOR_WEIGHT ? std::to_string(mmeExpBias.fp8BiasIn[TENSOR_WEIGHT]) : "NA",
    *attrs["fp8BiasOut"].mutable_s() = std::to_string(mmeExpBias.fp8BiasOut);
}

string GraphVisualization::renameNode(string name) const
{
    // We change the name of Convolution and FullyConnected nodes so the graph display software
    // (i.e. Netron) won't recognize these nodes and apply them special treatment that hides
    // Habana-specific behavior like the CIN connection into Convolution node.
    string convolution("Convolution");
    string fullyconnected("fully_connected");
    size_t pos = 0;

    // Replace "Convolution" with "Conv"
    while ((pos = name.find(convolution)) != string::npos)
    {
        name.replace(pos, convolution.size(), "Conv");
    }
    // Replace "fully_connected" with "FC"
    while ((pos = name.find(fullyconnected)) != string::npos)
    {
        name.replace(pos, fullyconnected.size(), "FC");
    }
    return name;
}

string GraphVisualization::formatTensorInfo(const TensorPtr& t) const
{
    const string    separator = "  |  ";

    stringstream    info;
    stringstream    sizes;
    std::string zero = "0";
    stringstream drange_stream;
    if (t->getDynamicRange().isSet)
    {
        drange_stream << std::setprecision(20) <<
                            "drange = (" << t->getDynamicRange().min <<
                            ", " << t->getDynamicRange().max << ")";
    }
    std::string drange = drange_stream.str();

    if (t->isDynamicShape())
    {
        sizes << "MaxSizes = " << t->getDimSizesStr(true, false) << separator
              << "MinSizes = " << t->getDimSizesStr(true, true) << separator;
    }
    else
    {
        sizes << "Sizes = " << t->getDimSizesStr(true, false) << separator;
    }

    info << t->getName() << separator << getStringFromSynDataType(t->getElementType()) << separator << sizes.str()
         << "expBias = " << t->getExpBias() << ", "
         << "scale = " << t->getScale() << separator << drange << separator << "ModelParam = " << t->isModelParameter()
         << separator;

    if (m_fullInfo)
    {
        info << "strides = " << t->getStridesStr(true) << separator;
        info << "data = " << (t->isBound() ? t->getAddress() : 0x0)<< ", sizeInBytes = "  << t->getTotalSizeInBytes() << separator;

        if (t->isPersistent())
        {
            info << "isPersistent" << separator;
        }

        if (t->isAliasedTensor())
        {
            info << "isAliased = " << t->getAliasTensor()->getName() << ", type = " << t->getAliasedTensorTypeStr() <<
                    ", offset: " << t->getAliasedByteOffset()  << separator;
        }

        if (t->isHostAliasedTensor())
        {
            info << "isHostAliased = " << t->getHostAliasTensor()->getName() << ", offset: " << t->getHostAliasOffset() << separator;
        }

        if (t->hasDramAllocatedTensor())
        {
            info << "isDramAliased = " << t->getDramAllocatedTensor()->getName() << ", offset: " << t->getDramAllocatedTensorOffset() << separator;
        }

        if (t->getTensorLocationString().compare("not allocated") !=0)
        {
            info << "location = " << t->getTensorLocationString() << separator;
        }

        if (t->tensorAllocatedInSram())
        {
            info << "sramOffset = 0x" << std::hex << t->getSramOffset() << std::dec << separator;
        }

        if (t->isDramOffsetSet())
        {
            info << "dramOffset = 0x" << std::hex << t->getDramOffset() << std::dec << separator;
        }

        // User memory sections
        if (t->isPersistent() || t->isPartOfRMWSection())
        {
            const std::string sectionType = t->isPersistent() ? "Persistent" : "RMW";
            const auto sectionId = t->isPersistent() ? t->getMemorySectionID() : t->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
            info << "userMemorySection(type=" << sectionType << ", id=" << sectionId << ")" << separator;
        }

        if (t->isWeights())
        {
            info << "isWeights" << separator;
        }

        if (t->isStaticParam())
        {
            info << "isStaticParam" << separator;
            info << "bufferDtype=" << getStringFromSynDataType(t->getBufferDataType()) << separator;
        }

        if (t->isLowered())
        {
            info << "isLowered" << separator;
        }

        if (!t->isDenseLayout())
        {
            info << "isSparseLayout" << separator;
        }

        if (t->isEnforcedOutput())
        {
            info << "isEnforcedOutput" << separator;
        }

        if (t->getTensorAnnotation().memory.pinned)
        {
            info << "isPinned" << separator;
        }

        if (t->getTensorAnnotation().memorySpaceInfo.prefetchInfo.prefetch)
        {
            info << "isPrefetch" << separator;
        }
        if (t->isReductionEnabled())
        {
            info << "isReductionEnabled" << separator;
        }
        if (t->isPerChannelQuant())
        {
            info << "isPerChannelQuant" << separator;
        }
        if (const auto& permutation = t->getPermutation(); permutation)
        {
            info << "permutation = " << permutation->toString();
        }
    }
    return info.str();
}

void GraphVisualization::addProtobufTensorInfo(VizNodePtr         nodeToAdd,
                                               const TensorPtr&   t,
                                               const std::string& inputNumStr,
                                               const std::string& ioString,
                                               std::string_view   dataLayout) const
{
    //TODO: separate to different fields and improve visibility of info. [SW-23534]
    vizAttrMap& attrs = *nodeToAdd->mutable_attr();

    const string    separator = "  |  ";
    stringstream    info;
    stringstream    sizes;

    stringstream drange_stream;
    if (t->getDynamicRange().isSet)
    {
        drange_stream << std::setprecision(20) <<
                            "drange = (" << t->getDynamicRange().min <<
                            ", " << t->getDynamicRange().max << ")";
    }
    std::string drange = drange_stream.str();
    if (t->isDynamicShape())
    {
        sizes << "MaxSizes = " << t->getDimSizesStr(true, false) << separator
              << "MinSizes = " << t->getDimSizesStr(true, true) << separator;
    }
    else
    {
        sizes << "Sizes = " << t->getDimSizesStr(true, false) << separator;
    }

    info << t->getName() << separator << sizes.str() << getStringFromSynDataType(t->getElementType()) << separator
         << "expBias = " << t->getExpBias() << ", "
         << "scale = " << t->getScale() << separator << drange << separator << "ModelParam = " << t->isModelParameter()
         << separator;

    if (m_fullInfo)
    {
        info << "strides = " << t->getStridesStr(true) << separator;
        info << "data = " << (t->isBound() ? t->getAddress() : 0x0)<< ", sizeInBytes = "  << t->getTotalSizeInBytes() << separator;

        if (t->getTotalElements() == 1 && t->isStaticParam() && t->getAddress() &&
            t->getElementType() == syn_type_float)
        {
            float data = *reinterpret_cast<float*>(t->getAddress());
            info << "buffer = " << data << separator;
        }

        if (t->isPersistent())
        {
            info << "isPersistent" << separator;
        }

        if (t->isAliasedTensor())
        {
            info << "isAliased = " << t->getAliasTensor()->getName() << ", type = " << t->getAliasedTensorTypeStr() <<
                    ", offset: " << t->getAliasedByteOffset()  << separator;
        }

        if (t->isHostAliasedTensor())
        {
            info << "isHostAliased = " << t->getHostAliasTensor()->getName() << ", offset: " << t->getHostAliasOffset() << separator;
        }

        if (t->hasDramAllocatedTensor())
        {
            info << "isDramAliased = " << t->getDramAllocatedTensor()->getName() << ", offset: " << t->getDramAllocatedTensorOffset() << separator;
        }

        if (t->getTensorLocationString().compare("not allocated") !=0)
        {
            info << "location = " << t->getTensorLocationString() << separator;
        }

        if (t->tensorAllocatedInSram())
        {
            info << "sramOffset = 0x" << std::hex << t->getSramOffset() << std::dec << separator;
        }

        if (t->isDramOffsetSet())
        {
            info << "dramOffset = 0x" << std::hex << t->getDramOffset() << std::dec << separator;
        }

        // User memory sections
        if (t->isPersistent() || t->isPartOfRMWSection())
        {
            const std::string sectionType = t->isPersistent() ? "Persistent" : "RMW";
            const auto sectionId = t->isPersistent() ? t->getMemorySectionID() : t->getTensorAnnotation().nonPersistentSectionInfo.sectionId.value();
            info << "userMemorySection(type=" << sectionType << ", id=" << sectionId << ")" << separator;
        }

        if (!dataLayout.empty())
        {
            info << "dataLayout = " << dataLayout << separator;
        }

        if (t->isWeights())
        {
            info << "isWeights" << separator;
        }

        if (t->isStaticParam())
        {
            info << "isStaticParam" << separator;
            info << "bufferDtype=" << getStringFromSynDataType(t->getBufferDataType()) << separator;
        }

        if (t->isLowered())
        {
            info << "isLowered" << separator;
        }

        if (!t->isDenseLayout())
        {
            info << "isSparseLayout" << separator;
        }

        if (t->isEnforcedOutput())
        {
            info << "isEnforcedOutput" << separator;
        }

        if (t->getTensorAnnotation().memory.pinned)
        {
            info << "isPinned" << separator;
        }

        if (t->getTensorAnnotation().memorySpaceInfo.prefetchInfo.prefetch)
        {
            info << "isPrefetch" << separator;
        }
        if (t->isReductionEnabled())
        {
            info << "isReductionEnabled" << separator;
        }
        if (t->isPerChannelQuant())
        {
            info << "isPerChannelQuant" << separator;
        }
        if (const auto& permutation = t->getPermutation(); permutation)
        {
            info << "permutation = " << permutation->toString();
        }
    }

    // ATM, one giant str with all input attributes
    *attrs[ioString + "Tensor:" + inputNumStr].mutable_s() = info.str();

    return;
}

bool GraphVisualization::formatAndWriteJsonFile(
        const vector<string>&  nodeRecords,
        const string&          argNodes,
        const string&          exitNodes,
        const string&          recipeName) const
{
    if (nodeRecords.size() == 0)
    {
        return true; // nothing to do
    }

    // In case the graph was given a recipe name before compilation, use it as filename prefix
    string prefix   = recipeName.empty() ? "" : recipeName + "-";
    string jsonPath = prefix + m_fileName + "-symbol.json";
    if (!m_dirName.empty())
    {
        jsonPath = m_fileName + "-" + prefix.substr(prefix.find("/") + 1) + "-symbol.json";
        jsonPath = m_dirName + "/" + jsonPath;
    }
    ofstream jsonfile;
    jsonfile.open(jsonPath);
    if (jsonfile.fail())
    {
        LOG_ERR(GC, "Visualization cannot create json file at {}", jsonPath);
        return false;
    }

    vector<string>::const_iterator itr = nodeRecords.begin();

    jsonfile << "{\n  \"nodes\": [\n";
    for (unsigned i = 0; i < nodeRecords.size()-1; ++i, ++itr)
    {
        jsonfile << *itr << ",\n";
    }
    jsonfile << *itr // avoid comma on last record
             << "\n  ],\n"
             << "  \"arg_nodes\": [\n" << argNodes << "\n  ],\n"
             << "  \"node_row_ptr\": [";
    for (unsigned i = 0; i < nodeRecords.size()-1; ++i)
    {
        jsonfile << i << ", ";
    }
    jsonfile << nodeRecords.size()-1 // avoid comma
             << "],\n"
             << "  \"heads\": [" << exitNodes << "],\n"
             << "  \"attrs\": {\"mxnet_version\": [\"int\", 10200]}\n"
             << "}";

    // file will be closed by d'tor
    return true;
}

void GraphVisualization::insertProtobufEntryNode(viz::Graph& vizGraph, const TensorPtr tensor) const
{
    HB_ASSERT_PTR(tensor);
    // do not need use return value, no need for it.
    insertTensorAsPlaceholder(vizGraph, tensor);
}

void GraphVisualization::insertProtobufExitNode(HabanaGraph& g, viz::Graph& vizGraph, const TensorPtr tensor) const
{
    HB_ASSERT_PTR(tensor);
    VizNodePtr nodeToAdd = insertTensorAsPlaceholder(vizGraph, tensor);

    // Set producer node as input.
    NodePtr producerNode = g.getTensorProducer(tensor);
    if (producerNode != nullptr)
    {
        nodeToAdd->add_input(producerNode->getNodeName());
    }

    // For output Nodes: list name of tensor in node
    nodeToAdd->set_op("OutputTensor");

}

VizNodePtr GraphVisualization::insertTensorAsPlaceholder(viz::Graph& vizGraph, const TensorPtr& tensor) const
{
    VizNodePtr nodeToAdd = vizGraph.add_node();
    nodeToAdd->set_op("Placeholder");
    nodeToAdd->set_name(tensor->getName());
    // obtain mutable attributes
    vizAttrMap& attrs = *nodeToAdd->mutable_attr();

    viz::Shape tensorShape;
    const SizeArray& inputSizes = tensor->getAllSizesInElements();
    for (auto sz : inputSizes)
    {
        tensorShape.add_dim()->set_size(sz);
    }
    viz::DataType dtype = ConvertSynToVizDataType(tensor->getElementType());
    attrs["dtype"].set_type(dtype);
    *attrs["synDataType"].mutable_s() = std::string(getStringFromSynDataType(tensor->getElementType()));
    *attrs["shape"].mutable_shape()   = tensorShape;
    const auto& permutation           = tensor->getPermutation();
    if (permutation)
    {
        *attrs["permutation"].mutable_s() = permutation->toString();
    }
    return nodeToAdd;
}

void GraphVisualization::insertProtobufNode(HabanaGraph& g, const NodePtr node, viz::Graph& vizGraph) const
{
    HB_ASSERT_PTR(node);

    const TensorVector& inputs = node->getInputs();
    const TensorVector& outputs = node->getOutputs();
    const auto&         inputDataLayouts  = node->getInputLayouts();
    const auto&         outputDataLayouts = node->getOutputLayouts();

    bool existsInputThatIsNotNull = std::any_of(inputs.begin(), inputs.end(),
                                                [](const TensorPtr& tensor) { return tensor != nullptr; });
    bool existsOutputThatIsNotNull = std::any_of(outputs.begin(), outputs.end(),
                                                 [](const TensorPtr& tensor) { return tensor != nullptr; });

    VizNodePtr nodeToAdd = vizGraph.add_node();
    nodeToAdd->set_op(renameNode(node->getNodeTypeStr()));
    nodeToAdd->set_name(node->getNodeName());

    if (existsInputThatIsNotNull)
    {
        const unsigned numOfDigits = std::to_string(inputs.size() - 1).size();
        for (unsigned i = 0; i < inputs.size(); ++i)
        {
            if (inputs.at(i) != nullptr)
            {
                NodePtr producerNode = g.getTensorProducer(inputs.at(i));
                if (producerNode != nullptr)
                {
                    nodeToAdd->add_input(producerNode->getNodeName());
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
                addProtobufTensorInfo(nodeToAdd, inputs.at(i), toString(i, numOfDigits), "input", dataLayout);
            }
            else
            {
                nodeToAdd->add_input("unused_port");
            }
        }

    }

    if (existsOutputThatIsNotNull)
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
                addProtobufTensorInfo(nodeToAdd, outputs.at(i), toString(i, numOfDigits), "output", dataLayout);
            }
        }
    }

    addProtoBuffWaitNodeInfo(nodeToAdd, g, node);
    if (HabanaGraph::runsOnMME(node))
    {
        addProtoBuffMmeNodeInfo(nodeToAdd, g, node);
    }

    //Add node parameters string
    std::string nodeParameters = node->getNodeParametersStr();
    vizAttrMap& attrs = *nodeToAdd->mutable_attr();
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

    // Add control dependencies
    for (NodePtr blockingNode : g.getBlockingNodes(node))
    {
        if (blockingNode != nullptr)
        {
            nodeToAdd->add_input(fmt::format("^{}", blockingNode->getNodeName()));
        }
    }

    std::string bundleIdxStr;
    std::string operationIdxStr;

    if (node->getNodeAnnotation().bundleInfo.is_set())
    {
        bundleIdxStr = std::to_string(node->getNodeAnnotation().bundleInfo->bundleIndex);
        operationIdxStr = std::to_string(node->getNodeAnnotation().bundleInfo->operationIndex);
    }
    else
    {
        bundleIdxStr = "N/A";
        operationIdxStr = "N/A";
    }

    *attrs["Bundle_idx"].mutable_s() = bundleIdxStr;
    *attrs["Op_idx"].mutable_s() = operationIdxStr;
    *attrs["Exec_idx"].mutable_s() = std::to_string(node->getExecutionOrderedIndex());

    // print nodes in BundleNode
    const std::shared_ptr<BundleNode> bundleNode = std::dynamic_pointer_cast<BundleNode>(node);
    if (bundleNode)
    {
        const NodeList& originalNodes = bundleNode->m_nodes;
        unsigned nodeCounter = 0;
        const unsigned numOfDigits = std::to_string(originalNodes.size() - 1).size();
        for (const NodePtr& n : originalNodes)
        {
            *attrs[fmt::format("_Node_{}", toString(nodeCounter, numOfDigits))].mutable_s() = n->getNodeName();
            ++nodeCounter;
        }
    }
}

bool GraphVisualization::formatAndWriteProtobufFile(viz::Graph& vizGraph, const string& recipeName) const
{
    // In case the graph was given a recipe name before compilation, use it as filename prefix
    string prefix   = recipeName.empty() ? "" : recipeName + "-";
    string pbPath = prefix + m_fileName + "-symbol.pbtxt";
    if (!m_dirName.empty())
    {
        pbPath = m_fileName + "-" + prefix.substr(prefix.find_last_of("/") + 1) + "-symbol.pbtxt";
        pbPath = m_dirName + "/" + pbPath;
    }

    string outStr = vizGraph.DebugString();
    ofstream pbFile(pbPath, ios::out | ios::trunc);
    pbFile << outStr;

    // file will be closed by d'tor
    return true;
}

void GraphVisualization::HandleNodeVizProtobuf(HabanaGraph& g, NodePtr node, viz::Graph& vizGraph,
                                               std::map<unsigned int, TensorInfo>& tensorInfoMap, unsigned& pbRecordCount) const
{

    // Walk through node's inputs
    for (TensorPtr input : node->getInputs())
    {
        // skip over optional input
        if (input == nullptr)
        {
            continue;
        }

        if (tensorInfoMap.find(input->getId()) == tensorInfoMap.end())
        {
            // Handle graph entry nodes
            insertProtobufEntryNode(vizGraph, input);
            unsigned currentRecordIndex = pbRecordCount++; // first mark current, then promote count
            tensorInfoMap[input->getId()] = {currentRecordIndex, 0};

        }
    }

    // Add current node to protobuf
    insertProtobufNode(g,node,vizGraph);
    unsigned currentRecordIndex = pbRecordCount++; // first mark current, then promote count

    // Walk through node's outputs
    unsigned outputIndex = 0;
    for (TensorPtr output : node->getOutputs())
    {
        // skip over optional output
        if (output == nullptr)
        {
            continue;
        }

        // Create an entry that holds the current (producer) node index in the json file
        // and the index of the specific output tensor among all node's outputs
        tensorInfoMap[output->getId()] = {currentRecordIndex, outputIndex};

        // Construct the graph exit node list
        if (g.isOutputTensor(output))
        {
            insertProtobufExitNode(g, vizGraph, output);
        }
        outputIndex++;
    }

}

viz::DataType GraphVisualization::ConvertSynToVizDataType(synDataType synType) const
{
    switch (synType)
    {
        default:
        case syn_type_na     :
            return viz::DataType::DT_INVALID;
        case syn_type_int8   : // alias to syn_type_fixed
            return viz::DataType::DT_INT8;
        case syn_type_bf16   :
            return viz::DataType::DT_BFLOAT16;
        case syn_type_float  : // alias to syn_type_single
            return viz::DataType::DT_FLOAT;
        case syn_type_int16  :
            return viz::DataType::DT_INT16;
        case syn_type_int32  :
            return viz::DataType::DT_INT32;
        case syn_type_int64:
            return viz::DataType::DT_INT64;
        case syn_type_uint8  :
            return viz::DataType::DT_UINT8;
        case syn_type_int4   :
            return viz::DataType::DT_VARIANT; // Using Variant as an alias int4 (unsupported in TF) [SW-23533]
        case syn_type_uint4  :
            return viz::DataType::DT_VARIANT_REF; // Using Variant Ref as an alias int4 (unsupported in TF) [SW-23533]
        case syn_type_fp16   :
            return viz::DataType::DT_HALF;
        case syn_type_uint16 :
            return viz::DataType::DT_UINT16;
        case syn_type_uint32 :
            return viz::DataType::DT_UINT32;
        case syn_type_uint64:
            return viz::DataType::DT_UINT64;
    }

}
