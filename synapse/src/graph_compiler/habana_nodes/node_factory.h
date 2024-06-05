#ifndef _NODE_FACTORY_H_
#define _NODE_FACTORY_H_

#include "habana_nodes.h"
#include "node.h"
#include "tensor.h"
#include "types.h"

#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

//A "class" (more like namespace) for generating nodes for insertion into a graph

class NodeFactory
{
public:

    static NodeFactory& getInstance()
    {
        static NodeFactory instance;
        return instance;
    }

    virtual ~NodeFactory();

    // Helper overloads for createNode, for cases where userParamsSize is NOT being passed
    static NodePtr createNode(const TensorVector&         inputs,
                              const TensorVector&         outputs,
                              std::nullptr_t              userParams,
                              std::string_view            guid,
                              std::string_view            name,
                              const Node::NodeProperties& properties = Node::NodeProperties())
    {
        return createNode(inputs, outputs, nullptr, 0, guid, name, properties);
    }

    static NodePtr createInternalNode(const TensorVector&         inputs,
                                      const TensorVector&         outputs,
                                      std::nullptr_t              userParams,
                                      std::string_view            guid,
                                      std::string_view            name,
                                      std::string_view            guidWithoutDtype = "",
                                      const Node::NodeProperties& properties       = Node::NodeProperties())
    {
        return createNode(inputs,
                          outputs,
                          nullptr,
                          0,
                          guid,
                          guidWithoutDtype.empty() ? guid : guidWithoutDtype,
                          name,
                          false,
                          properties);
    }

    template<class T>
    static NodePtr createNode(const TensorVector&         inputs,
                              const TensorVector&         outputs,
                              const T*                    userParams,
                              std::string_view            guid,
                              std::string_view            name,
                              const Node::NodeProperties& properties = Node::NodeProperties())
    {
        return createNode(inputs, outputs, const_cast<T*>(userParams), sizeof(T), guid, name, properties);
    }

    template<class T>
    static NodePtr createInternalNode(const TensorVector&         inputs,
                                      const TensorVector&         outputs,
                                      const T*                    userParams,
                                      std::string_view            guid,
                                      std::string_view            name,
                                      std::string_view            guidWithoutDtype = "",
                                      const Node::NodeProperties& properties       = Node::NodeProperties())
    {
        return createNode(inputs,
                          outputs,
                          const_cast<T*>(userParams),
                          sizeof(T),
                          guid,
                          guidWithoutDtype.empty() ? guid : guidWithoutDtype,
                          name,
                          false,
                          properties);
    }

    static NodePtr createNode(const TensorVector&         inputs,
                              const TensorVector&         outputs,
                              UserParams                  userParams,
                              unsigned                    userParamsSize,
                              std::string_view            guid,
                              std::string_view            name,
                              const Node::NodeProperties& properties = Node::NodeProperties(),
                              const synDeviceType*        deviceType = nullptr);
    // Same as above for a generic TPC node

    //Create a "generic" node - for most part custom kernels running on TPC
    static NodePtr createGenericTPCNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        std::nullptr_t      userParams,
                                        std::string_view    guid,
                                        std::string_view    name = "")
    {
        return createGenericTPCNode(inputs, outputs, nullptr, 0, guid, name);
    }

    template<class T>
    static NodePtr createGenericTPCNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        const T*            userParams,
                                        std::string_view    guid,
                                        std::string_view    name = "")
    {
        return createGenericTPCNode(inputs, outputs, const_cast<T*>(userParams), sizeof(T), guid, name);
    }

    static NodePtr createGenericTPCNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        unsigned            paramsSize,
                                        std::string_view    guid,
                                        std::string_view    name = "");


    //Debug nodes, no functionality, only structure
    //b output, a input
    static NodePtr createDebugNode(const TensorPtr& opA, const TensorPtr& opB, const std::string& name = "");
    //b, c outputs, a input
    static NodePtr
    createDebugForkNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name = "");
    //c output, a, b inputs
    static NodePtr
    createDebugJoinNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name = "");
    //Same as debug node but different opcode
    static NodePtr createDebug2Node(const TensorPtr& opA, const TensorPtr& opB, const std::string& name = "");

    static bool        isInternalNode(const StringViewWithHash& guid);
    static bool isApiNode(const std::string& guid);
    static std::size_t getNumApiNodes();

    // clears resources
    void clear();

    static const char* convolutionNodeTypeName;
    static const char* convolution3DNodeTypeName;
    static const char* gemmNodeTypeName;
    static const char* gemmDeDxNodeTypeName;
    static const char* gemmDeDwNodeTypeName;
    static const char* batchGemmNodeTypeName;
    static const char* batchGemmDeDxNodeTypeName;
    static const char* batchGemmDeDwNodeTypeName;
    static const char* maskedBatchGemmNodeTypeName;
    static const char* transposeNodeTypeName;
    static const char* transposeDmaNodeTypeName;
    static const char* transposeMmeNodeTypeName;
    static const char* transposeLogicNodeTypeName;
    static const char* broadcastNodeTypeName;
    static const char* concatenateNodeTypeName;
    static const char* concatenateNodeInternalTypeName;
    static const char* concatenateNodeLogicalInternalTypeName;
    static const char* reductionNodeTypeName;
    static const char* stridedViewNodeTypeName;
    static const char* stridedInsertNodeTypeName;
    static const char* multiInsertNodeTypeName;
    static const char* logicalStridedViewTypeName;
    static const char* logicalStridedInsertTypeName;
    static const char* flattenNodeTypeName;
    static const char* expandDimsNodeTypeName;
    static const char* splitNodeTypeName;
    static const char* splitNodeInternalTypeName;
    static const char* sliceAxisNodeTypeName;
    static const char* sliceNodeTypeName;
    static const char* logicalSliceFwdNodeTypeName;
    static const char* reshapeNodeTypeName;
    static const char* staticReshapeNodeTypeName;
    static const char* dynamicReshapeNodeTypeName;
    static const char* tpcMemcpyNodeTypeName;
    static const char* dmaMemcpyNodeTypeName;
    static const char* beamSearchNodeTypeName;
    static const char* embeddingNodeTypeName;
    static const char* addNodeTypeName;
    static const char* reluNodeTypeName;
    static const char* reverseNodeTypeName;
    static const char* deDxNodeTypeName;
    static const char* deDx3DNodeTypeName;
    static const char* deDwNodeTypeName;
    static const char* deDw3DNodeTypeName;
    static const char* tensorViewNodeTypeName;
    static const char* memcpyNodeTypeName;
    static const char* memsetNodeTypeName;
    static const char* dmaMemsetNodeTypeName;
    static const char* tpcMemsetNodeTypeName;
    static const char* clAwareMemsetNodeTypeName;
    static const char* clAwareMemgetNodeTypeName;
    static const char* clAwareHybridNodeTypeName;
    static const char* nmsNodeTypeName;
    static const char* waitNodeTypeName;
    static const char* DebugNodeTypeName;
    static const char* identityNodeTypeName;
    static const char* momentsFwdNodeTypeName;
    static const char* tfBatchNormNodeTypeName;
    static const char* tfFusedBatchNormGradName;
    static const char* stridedSliceGradNodeTypeName;
    static const char* sliceInsertNodeTypeName;
    static const char* sliceBwdNodeTypeName;
    static const char* logicalSliceBwdNodeTypeName;
    static const char* logicalSliceInsertNodeTypeName;
    static const char* logicalRequantNodeTypeName;
    static const char* rotateNodeTypeName;
    static const char* serializeDMANodeTypeName;
    static const char* deserializeDMANodeTypeName;
    static const char* serializeTPCNodeTypeName;
    static const char* deserializeTPCNodeTypeName;
    static const char* physicalReshapeNodeTypeName;
    static const char* dynamicStridedDMANodeTypeName;
    static const char* dynamicSliceDMANodeTypeName;
    static const char* dynamicStridedTPCNodeTypeName;
    static const char* dynamicSliceTPCNodeTypeName;
    static const char* memcpyNdNodeTypeName;
    static const char* memcpyNdInt64NodeTypeName;
    static const char* memcpyNdUint64NodeTypeName;
    static const char* physicalConcatNodeTypeName;
    static const char* physicalConcatSplitSubNodeTypeNameDMA;
    static const char* physicalConcatSplitSubNodeTypeNameTPC;
    static const char* extractShapeNodeTypeName;
    static const char* mergeShapesNodeTypeName;
    static const char* splitShapeNodeTypeName;
    static const char* flattenShapeNodeTypeName;
    static const char* expandDimsShapeNodeTypeName;
    static const char* squeezeShapeNodeTypeName;
    static const char* transposedShapeNodeTypeName;
    static const char* transposeSliceH2DNodeTypeName;
    static const char* squeezeNodeTypeName;
    static const char* FrobeniusNormTypeName;
    static const char* physicalSplitNodeTypeName;
    static const char* einsumTypeName;
    static const char* einsumExpandShapeNodeTypeName;
    static const char* dynamicSplitNodeTypeName;
    static const char* physicalFlattenNodeTypeName;
    static const char* dynamicRangeNodeTypeName;
    static const char* inferShapeNodeTypeName;
    static const char* reinterpretCastNodeTypeName;
    static const char* inferMaxShapeNodeTypeName;
    static const char* tileShapeNodeTypeName;
    static const char* transposedDeDxNodeTypeName;
    static const char* transposedDeDx3DNodeTypeName;

    // H2D manipulation nodes
    static const char* dynamicStridedDmaExpandH2DNodeTypeName;
    static const char* dynamicStridedDmaReinterpretH2DNodeTypeName;
    static const char* dynamicSliceDmaExpandH2DNodeTypeName;
    static const char* stridedOpsConversionNodeTypeName;
    static const char* sliceConversionNodeTypeName;

    // Inference TPC kernel guids that requires GC attention
    static const char* bitshiftNodeTypeName;
    static const char* tanhNodeTypeName;
    static const char* sigmoidNodeTypeName;
    static const char* sequenceLengthNodeTypeName;
    static const char* sequenceMaskNodeTypeName;
    static const char* rnnNodeTypeName;
    static const char* softmaxNodeTypeName;
    static const char* raggedSoftmaxNodeTypeName;
    static const char* maxPoolRoiNodeTypeName;
    static const char* andNodeTypeName;
    static const char* maxPool2dNodeTypeName;
    static const char* avgPool2dNodeTypeName;
    static const char* orNodeTypeName;
    static const char* xorNodeTypeName;
    static const char* dropOutNodeTypeName;
    static const char* notNodeTypeName;
    static const char* leakyReluNodeTypeName;
    static const char* batchNormNodeTypeName;
    static const char* maxPool3dNodeTypeName;
    static const char* constantNodeTypeName;
    static const char* sequenceReverseNodeTypeName;
    static const char* upsampleNodeTypeName;
    static const char* negNodeTypeName;
    static const char* clipNodeTypeName;
    static const char* filter2dNodeTypeName;
    static const char* staticReshapeShapeNodeTypeName;
    static const char* cropMirorNormNodeTypeName;

    // End of inference TPC kernel guids list

    // Functions for selecting TPC/DMA node type depending on device
    static std::string_view getSerializeNodeGUID();
    static std::string_view getDeserializeNodeGUID();
    static std::string_view getDynamicStridedMemcpyNodeGUID();
    static std::string_view getDynamicSliceMemcpyNodeGUID();
    static std::string_view getPhysicalSplitConcatSubNodeGUID();

private:
    NodeFactory();
    NodeFactory (NodeFactory const&)   = delete;
    void operator=(NodeFactory const&) = delete;

    static NodePtr createNode(const TensorVector&         inputs,
                              const TensorVector&         outputs,
                              UserParams                  userParams,
                              unsigned                    paramsSize,
                              std::string_view            guid,
                              std::string_view            guidWithoutDType,
                              std::string_view            name,
                              bool                        isMarkedAsTPCNode,
                              const Node::NodeProperties& properties = Node::NodeProperties(),
                              const synDeviceType*        deviceType = nullptr);

    typedef NodePtr (*CreateNode)(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name);

    std::unordered_map<StringViewWithHash, CreateNode> m_factoryMap;
    std::unordered_set<StringViewWithHash>             m_internalNodes;
    std::unordered_set<std::string_view>               m_apiNodes;

    typedef NodePtr (*CreateNodeWithParamSize)(const TensorVector& inputs,
                                               const TensorVector& outputs,
                                               UserParams          userParams,
                                               unsigned            userParamsSize,
                                               std::string_view    guid,
                                               std::string_view    name);
    std::unordered_map<StringViewWithHash, CreateNodeWithParamSize> m_factoryMapWithSize;
};

#endif
