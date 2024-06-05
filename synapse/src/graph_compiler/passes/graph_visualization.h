#pragma once

#include "habana_pass.h"
#include "protobuf/graph_visualization.pb.h"
#include "types.h"

enum VisualizationMode
{
    VISUALIZATION_MODE_MXNET_JSON = 0,
    VISUALIZATION_MODE_TF_PB      = 1
};

// prevent long prototypes.
namespace viz = graph_visualization;

// We use regular pointers here since c++ protobuf API doesn't support shared_ptr
// OOTB. making it work will result in making changes to open source code that is problematic with apache license.
// in any case, there is NO use of new/delete, and these are all mutable references, with no ownership.
typedef viz::Node* VizNodePtr;

struct TensorInfo
{
    unsigned producerRecordIndex;  // index of this tensor's producer node within nodeRecords
    unsigned producerOutputPort;   // index of this output tensor within the producer vector of outputs
};

class GraphVisualization
{
public:
    GraphVisualization(std::string_view fileName,
                       std::string_view startTensor = "",
                       std::string_view endTensor   = "",
                       bool             fullInfo    = true);
    GraphVisualization(const std::string& dirName, bool fullInfo);

    bool Apply(HabanaGraph& g) const;

    void setFileName(const std::string& fileName);

    static bool graphVisualizationPostOnFailure(HabanaGraph& graph);

    static bool graphVisualizationOnDemand(HabanaGraph& graph, const std::string& name);

    static VisualizationMode getVisualizationMode();

protected:
    bool        createOutputDir(const std::string& dirName);
    bool        createGraphVizOutput(HabanaGraph& g) const;
    void        HandleNodeVizJson(HabanaGraph&                    g,
                                  NodePtr                         node,
                                  std::stringstream&              argNodes,
                                  std::stringstream&              exitNodes,
                                  std::vector<std::string>&       nodeRecords,
                                  std::map<unsigned, TensorInfo>& tensorInfoMap) const;
    void        HandleNodeVizProtobuf(HabanaGraph&                        g,
                                      NodePtr                             node,
                                      viz::Graph&                         vizGraph,
                                      std::map<unsigned int, TensorInfo>& tensorInfoMap,
                                      unsigned&                           pbRecordCount) const;
    unsigned    insertRecord(std::vector<std::string>& nodeRecords, std::string&& record) const;
    void        addWaitNodeInfo(HabanaGraph& g, const NodePtr& node, std::stringstream& record) const;
    void        addProtoBuffWaitNodeInfo(VizNodePtr nodeToAdd, const HabanaGraph& g, const NodePtr& node) const;
    void        addProtoBuffMmeNodeInfo(VizNodePtr nodeToAdd, const HabanaGraph& g, const NodePtr& node) const;
    std::string renameNode(std::string name) const;
    std::string formatArgNodeRecord(const TensorPtr& tensor) const;
    std::string formatNodeRecord(HabanaGraph& g, const NodePtr& node, const std::string& inputs) const;
    std::string formatTensorInfo(const TensorPtr& t) const;
    bool        formatAndWriteJsonFile(const std::vector<std::string>& nodeRecords,
                                       const std::string&              argNodes,
                                       const std::string&              exitNodes,
                                       const std::string&              recipeName) const;

    void          insertProtobufEntryNode(viz::Graph& vizGraph, const TensorPtr tensor) const;
    void          insertProtobufExitNode(HabanaGraph& g, viz::Graph& vizGraph, const TensorPtr tensor) const;
    void          insertProtobufNode(HabanaGraph& g, const NodePtr node, viz::Graph& vizGraph) const;
    bool          formatAndWriteProtobufFile(viz::Graph& vizGraph, const std::string& recipeName) const;
    void          addProtobufTensorInfo(VizNodePtr         nodeToAdd,
                                        const TensorPtr&   t,
                                        const std::string& inputNumStr,
                                        const std::string& ioString,
                                        std::string_view   dataLayout) const;
    viz::DataType ConvertSynToVizDataType(synDataType synType) const;
    VizNodePtr    insertTensorAsPlaceholder(viz::Graph& vizGraph, const TensorPtr& tensor) const;

    std::string       m_startTensor;
    std::string       m_endTensor;
    std::string       m_fileName;
    std::string       m_dirName;
    bool              m_fullInfo;
    VisualizationMode m_mode;
};
