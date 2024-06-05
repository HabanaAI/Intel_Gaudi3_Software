#pragma once
#include "types.h"
#include "habana_graph.h"
#include "transpose_node.h"
// contain common definitions and methods for pattern matcher and fuser
class TransposeFuserBase
{
public:
    TransposeFuserBase()  = default;
    ~TransposeFuserBase() = default;

    static bool isValidPermutation(const TransposePermutationArray& permutation, const NodePtr& mmeNode)
    {
        return isValidPermutation(permutation, getSupportedPermutations(mmeNode));
    }

protected:
    static const TransposePermutationArrayVec& getSupportedPermutations(const NodePtr& mmeNode);
    static bool                                isValidPermutation(const TransposePermutationArray&    permutation,
                                                                  const TransposePermutationArrayVec& supportedPermutations);

    bool isTransposeNode(const NodePtr& node) const;
    bool isValidTransposePermutation(const NodePtr& transposeNode, const NodePtr& mmeNode) const;

    typedef enum
    {
        NOOP                    = 0b00000000000,
        TRANSPOSE_A_INPUT       = 0b00000000001,
        REPLACE_A_INPUT         = 0b00000000010,
        RESHAPE_A_INPUT         = 0b00000000100,
        REMOVE_TRANSPOSE_A_NODE = 0b00000001000,
        TRANSPOSE_B_INPUT       = 0b00000010000,
        REPLACE_B_INPUT         = 0b00000100000,
        RESHAPE_B_INPUT         = 0b00001000000,
        REMOVE_TRANSPOSE_B_NODE = 0b00010000000,
        SWAP_INPUTS             = 0b00100000000,
        TRANSPOSE_OUTPUT        = 0b01000000000,
        REPLACE_OUTPUT          = 0b10000000000,
    } TransposeActions;
};

class TransposeFuseCandidateFinder : public TransposeFuserBase
{
public:
    TransposeFuseCandidateFinder()  = default;
    ~TransposeFuseCandidateFinder() = default;
    NodeVector findCandidates(HabanaGraph& graph);

private:
    bool checkInputCandidates(const NodePtr& node, const HabanaGraph& graph);
    bool checkOutputCandidates(const NodePtr& node, const HabanaGraph& graph);
};

class TransposeFuser : public TransposeFuserBase
{
public:
    TransposeFuser()  = default;
    ~TransposeFuser() = default;
    bool fuseTransposeIntoMmeNode(HabanaGraph& g, const NodePtr& mmeNode);

private:
    void        fuse(HabanaGraph& g, const NodePtr& mmeNode);
    void        checkTransposeA(HabanaGraph& g, const NodePtr& mmeNode);
    void        checkTransposeB(HabanaGraph& g, const NodePtr& mmeNode);
    void        checkTransposeOut(HabanaGraph& g, const NodePtr& mmeNode);
    void        doTransposeA(const std::shared_ptr<GEMMNode>& newGemmNode, synGEMMParams& newGemmParams);
    void        doTransposeB(const std::shared_ptr<GEMMNode>& newGemmNode, synGEMMParams& newGemmParams);
    void        doSwapInputs(const std::shared_ptr<GEMMNode>& newGemmNode, synGEMMParams& newGemmParams);
    void        doReplaceOutput(const std::shared_ptr<GEMMNode>& newGemmNode);
    bool        canConvertToGemm(const NodePtr& node);
    bool        checkInputsDim(const NodePtr& node);
    std::string actionToStr() const;
    bool        isFlatten(const NodePtr& node) const
    {
        std::shared_ptr<FlattenNode> flatten = std::dynamic_pointer_cast<FlattenNode>(node);
        if (flatten != nullptr) return true;
        std::shared_ptr<ReshapeNode> reshape = std::dynamic_pointer_cast<ReshapeNode>(node);
        if (reshape == nullptr) return false;
        const auto& input  = reshape->getInput(0);
        const auto& output = reshape->getOutput(0);
        if (input->getDim() < output->getDim())
        {
            return false;
        }
        const auto& inputSizes  = input->getAllSizesInElements();
        const auto& outputSizes = output->getAllSizesInElements();
        for (auto inputIt = std::next(inputSizes.begin()); inputIt != inputSizes.end(); inputIt++)
        {
            unsigned size0 = multiplyElements(inputSizes.begin(), inputIt);
            unsigned size1 = multiplyElements(inputIt, inputSizes.end());
            if (outputSizes[0] == size0 && outputSizes[1] == size1)
            {
                return true;
            }
        }
        return false;
    }
    unsigned    m_actions      = 0;
    NodeVector  m_oldNodes     = {};
    NodeVector  m_newNodes     = {};
    NodeVector  m_mmeProducerA = {};
    NodeVector  m_mmeProducerB = {};
    NodeVector  m_mmeConsumer  = {};
};
