#include "dma_transpose_node.h"

#include "access_pattern_generator.h"
#include "defs.h"
#include "dma_memcopy_node.h"
#include "dma_transpose_helper.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "sif/shape_inference_metadata.h"
#include "split_strategies.h"
#include "transpose_node.h"
#include "transpose_permutation.h"
#include "types_exception.h"

const TransposePermutationArray DMATransposeNode::s_permutation =
{
    TransposePermutationDim::TPD_Width,
    TransposePermutationDim::TPD_Channel,
    TransposePermutationDim::TPD_Height,
    TransposePermutationDim::TPD_Depth,
    TransposePermutationDim::TPD_Batch
};

/**
 * @brief Returns whether a permutation is supported by the hardware transpose engine specification. Currently relevant for Gaudi2, Greco and Gaudi.
 * @param perm The permutation to test
 * @return true When permutation can be executed on the transpose engine
 * @return false otherwise
 */
static bool isSupportedPermutation(TransposePermutationArray perm)
{
    perm.erase(std::remove(perm.begin(), perm.end(), TransposePermutationDim::TPD_Channel));
    for (size_t i = 0; i < perm.size(); i++)
    {
        if(perm[i] != static_cast<TransposePermutationDim>(i + 1))
        {
            return false;
        }
    }
    return true;
}

DMATransposeNode::DMATransposeNode(const TensorPtr& in, const TensorPtr& out, std::string_view name)
: DMANode(in, out, name, DMA_TYPE_INTERNAL, SIF_TRANSPOSE)
{
    HB_ASSERT(out->isDenseLayout(),
              "Tensor {} can't be sparse layout because it is DMATransposeNode output",
              out->getName());
    getNodeAnnotation().canSkipSplitToLogical = false;
    getNodeAnnotation().splitToLogicalROIs    = GCFG_ENABLE_TRANSPOSE_LOGICAL_ROIS_SPLIT.value();
    m_outputs[0]->getTensorAnnotation().dataInfo.mustBeDense = true;
}

bool DMATransposeNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "DMATransposeNode Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    return getInput(TENSOR_IFM)->compareGeometryWithTranspose(*getOutput(TENSOR_OFM), m_permutation) && DMANode::validateNode();
}

NodePtr DMATransposeNode::clone() const
{
    return NodePtr(new DMATransposeNode(*this));
}

bool DMATransposeNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getTraits().getHalReader()->isDmaTransposeSupported(getInput(0)->getElementType()))
    {
        LOG_ERR(HABANA_NODE, "DMATransposeNode Invalid element type: {}", getInput(0)->getElementType());
        return false;
    }
    return isSupported(g.getDeviceType()) && DMANode::validateNodeForGraph(g);
}

NodePtr DMATransposeNode::createNode(const TensorVector& inputs,
                                     const TensorVector& outputs,
                                     UserParams          userParams,
                                     std::string_view    guid,
                                     std::string_view    name)
{
    NodePtr dmaTransposeNode(new DMATransposeNode(inputs[0], outputs[0], name));
    dmaTransposeNode->setParams(userParams, sizeof(TransposePermutationArray));
    return dmaTransposeNode;
}

void DMATransposeNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    TransposePermutationArray permutation;
    if (userParams == nullptr)
    {
        TransposePermutationArray array;
        for (int i = 0; i < m_inputs[0]->getDim(); i++)
        {
            array.push_back(TransposePermutationDim(i));
        }
        std::swap(array[0], array[1]);
        permutation = array;
    }
    else
    {
        if (userParamsSize != sizeof(TransposePermutationArray))
        {
            LOG_ERR(HABANA_NODE, "DMATransposeNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(TransposePermutationArray));
        }
        permutation = *reinterpret_cast<TransposePermutationArray*>(userParams);
    }
    m_permutation = std::move(permutation);
    m_sifMetadata = std::make_shared<SifTransposeMetadata>();
    memcpy(m_sifMetadata->permutation, m_permutation.data(), m_permutation.size() * sizeof(m_permutation[0]));
    if (!isSupportedPermutation(m_permutation))
    {
        throw SynapseException(fmt::format("Unsupported permutation: ", toString(m_permutation, ',')));
    }
    LOG_TRACE(HABANA_NODE,
              "DMATransposeNode name - {}, params - permutation={}, in sizes={}",
              getNodeName(),
              toString(m_permutation, ','),
              toString(m_inputs[0]->getNSizesInElements(), ','));
}

DMA_OP_TYPE DMATransposeNode::getOpType() const
{
    return DMA_OP_TYPE::DMA_OP_TRANSPOSE;
}

SifNodeParams DMATransposeNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(m_sifMetadata.get());
}

size_t DMATransposeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifTransposeMetadata);
}

gc::access_pattern::NodeAccessPatternPtr DMATransposeNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternTransposeGenerator::generate(this, m_permutation);
}

void DMATransposeNode::replaceOutput(unsigned index, const TensorPtr& newTensor)
{
    HB_ASSERT(index == 0, "Can't replace not existing output");
    m_outputs[0]->getTensorAnnotation().dataInfo.mustBeDense = false;
    newTensor->getTensorAnnotation().dataInfo.mustBeDense = true;
    Node::replaceOutput(index, newTensor);
}

const std::shared_ptr<SplitStrategy>& DMATransposeNode::getSplitStrategy()
{
    if (!m_splitStrategy)
    {
        HB_ASSERT_PTR(this->getGraphTraits());
        auto params = this->getGraphTraits()->getHalReader()->getDmaTransposeEngineParams();

        if (isFullyUtilized())
        {
            m_splitStrategy = std::make_shared<SplitFullyUtilizedTranspose>(params,
                                                                            this->permutation(),
                                                                            getSplitDimensionsOrder());
        }
        else
        {
            m_splitStrategy = std::make_shared<SplitTransposeToLowDescriptorCount>(params,
                                                                                   this->permutation(),
                                                                                   getSplitDimensionsOrder());
        }
    }
    return m_splitStrategy;
}

bool DMATransposeNode::RunOnCpu()
{
    std::shared_ptr<Tensor> in  = getInput(TENSOR_IFM);
    std::shared_ptr<Tensor> out = getOutput(TENSOR_OFM);
    TransposeNode::transposeOnCpu(in, out, s_permutation);
    return true;
}

bool StridedDMANodeViaTransposeNode::validateNode() const
{
    if (m_inputs.at(0)->getDim() != 2)
    {
        LOG_ERR(HABANA_NODE, "StridedDMANodeViaTransposeNode: input dim must be 2)");
        return false;
    }

    if (m_inputs.at(0)->getSizeInElements(0) != 1)
    {
        LOG_ERR(HABANA_NODE, "StridedDMANodeViaTransposeNode: dim 0 size must be 1)");
        return false;
    }

    return Node::validateNode();
}
