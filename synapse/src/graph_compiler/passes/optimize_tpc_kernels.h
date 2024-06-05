#pragma once

#include "habana_graph.h"
#include "habana_nodes.h"
#include "types.h"

class SuggestedManipulationHandlerBase
{
public:
    SuggestedManipulationHandlerBase(HabanaGraph& graph, TPCNode& node) : m_graph(graph), m_node(node) {}
    virtual ~SuggestedManipulationHandlerBase() = default;

    static bool shouldSkipSuggestedTensorManipulation(TPCNode& node, const HabanaGraph& graph);

    bool isSuggestedTensorManipulationAvailable();

    virtual bool applySuggestedTensorManipulation() = 0;

    const NodeVector& extract() const { return m_nodesToAdd; }

    static NSizeArray getReshapeMinTensorShape(const TensorPtr& tensor, const TSize* newMaxShape, uint32_t opDims);

protected:
    virtual std::string getSuggestedTensorManipulationNodeName(tpc_lib_api::TensorOperationType opType,
                                                               unsigned                         tensorIdx,
                                                               bool                             isInput)                 = 0;
    virtual void        setManipulatedTensorNameName(Tensor& t, std::string_view nameSuffix) = 0;

    static inline NodePtr getReshapeNode(const TensorPtr& in, const TensorPtr& out, std::string_view name);
    static NodePtr
    getTransposeNode(const TensorPtr& in, const TensorPtr& out, std::string_view name, const uint32_t* permutation);

    static TPCNodePtr
    getTileNode(const TensorPtr& in, const TensorPtr& out, const uint64_t* newShape, std::string_view name);
    static inline uint32_t getNewShapeDim(const TensorPtr& tensor, const TSize* newShape);
    static bool            isOuterDynamicDimReshaped(const TensorPtr& t,
                                                     unsigned         firstDynamicDim,
                                                     NSizeArray&      suggestion /* INOUT */,
                                                     uint32_t         opDims);

    static NSizeArray getReshapeMinTensorShapeWithFDD(const TensorPtr& tensor,
                                                      const TSize*     newMaxShape,
                                                      unsigned         newFirstDynamicDim,
                                                      unsigned         origFirstDynamicDim);

    static bool getTiledTensorShape(const TensorPtr& tensor, const uint64_t* newShape, NSizeArray& newTiledShape);

    TensorPtr createClonedTensor(const TensorPtr& tensor,
                                 const TSize*     newMaxShape,
                                 const TSize*     newMinShape,
                                 std::string_view nameSuffix);

    static bool isTensorSparseAfterRunLogicalOps(const TensorPtr& tensor, const NodeList& nodes, bool isInput);

    virtual bool isTensorSparse(const TensorPtr& tensor) const;

    bool shouldRejectAliasedMemcopy() const;
    static bool shouldRejectMemcopyOptimization(const TPCNode& n, const HabanaGraph& g);

    bool isHugeTensor(const TensorPtr& tensor) const;

    bool shouldRejectSparseTensorShapeManipulation(const TensorPtr tensor,
                                                   bool            stridedTensorManipulationFeatureEnabled,
                                                   float           stridedTensorManipulationUtilizationThreshold,
                                                   unsigned        tpcVectorSize) const;

    bool applyManipulationCheckAndProcess(const tpc_lib_api::TensorOperation* suggestion,
                                          const TensorVector&                 tensors,
                                          TensorVector&                       newTensors,
                                          bool&                               abortManipulation,
                                          bool                                isInput);

    static void
         calcInversedTransposePermutation(const uint32_t* permutation, unsigned* inversedPermutation, unsigned dims);
    bool addNodesNeededForSelectedManipulation(const TensorVector* tensors, TensorVector* newTensors, bool isInput);
    static inline bool isDifferentShape(const TensorPtr& tensor, const tpc_lib_api::TensorOperation& newOp);
    static inline bool isIdentityPermutation(const TensorPtr& tensor, const uint32_t* newPermutation);
    bool               isEmptyTensorManipulationSuggestion() const;
    static TensorVector            filterAuxAndNullTensors(const TensorVector& src);
    static inline std::string_view getTileGuid(synDataType type);

protected:
    HabanaGraph&                              m_graph;
    TPCNode&                                  m_node;
    bool                                      m_skipDynamicNodeHandling = false;
    tpc_lib_api::TensorManipulationSuggestion m_suggestion;
    TPCLibTensorOperandsSuggestionsVector     m_operandsSuggestions;
    NodeVector                                m_nodesToAdd;
    PermutationVector                         m_newInputPermutations;
};

class GraphModeSuggestedManipulationHandler final : public SuggestedManipulationHandlerBase
{
public:
    GraphModeSuggestedManipulationHandler(HabanaGraph& graph, TPCNode& node)
    : SuggestedManipulationHandlerBase(graph, node)
    {
    }
    void addShapeTensorSuggestions();
    bool applySuggestedTensorManipulation() override;

private:
    std::string getSuggestedTensorManipulationNodeName(tpc_lib_api::TensorOperationType opType,
                                                       unsigned                         tensorIdx,
                                                       bool                             isInput) override;
    void        setManipulatedTensorNameName(Tensor& t, std::string_view nameSuffix) override;
    TPCNodePtr  createConcreteTpcNode(const TensorVector& newInputs, const TensorVector& newOutputs) const;
};