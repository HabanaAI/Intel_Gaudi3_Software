#include "mme_fusion_utils.h"

#include "habana_graph.h"
#include "habana_nodes/habana_nodes.h"
#include "types_exception.h"

unsigned MMEFusionUtils::getOutputChannels(const pNode& mmeNode, bool isConvNode)
{
    unsigned outputChannels;
    if (isConvNode)
    {
        outputChannels = mmeNode->outputDimNameToSize(TENSOR_OFM, 'C');
    }
    else
    {
        synGEMMParams gemmParams = static_cast<GEMMNode *>(mmeNode.get())->getGEMMParams();
        bool isTransposed = gemmParams.transpose_b;
        if (isTransposed)
        {
            outputChannels = mmeNode->getInput(TENSOR_WEIGHT)->getSizeInElements(WEIGHT_DIM_C);
        }
        else
        {
            outputChannels = mmeNode->getInput(TENSOR_WEIGHT)->getSizeInElements(WEIGHT_DIM_K);
        }
    }
    return outputChannels;
}

bool MMEFusionUtils::canBeBias(uint8_t biasCandidate, const pNode& addNode, const pNode& mmeNode, HabanaGraph& g)
{
    bool isConvNode = (mmeNode->getNodeType() == Node::eNodeType::TYPE_CONVOLUTION);

    if (mmeNode->getInput(TENSOR_BIAS) != nullptr) return false;// mme node already has bias

    pTensor biasCandidateTensor = addNode->getInput(biasCandidate);
    if (!biasCandidateTensor->isStaticParam()) return false;  // bias must a static tensor for MME

    // since bias will be quantized differently (in int32 and depending on the weights scale), if there are other
    // consumers then we can't use it as bias.
    // TODO: create a new tensor copy in this case (if we see there's a need for this optimization)
    if (g.getNumberOfTensorConsumers(biasCandidateTensor) > 1) return false;

    unsigned dims = mmeNode->getOutput(TENSOR_OFM)->getDim();
    HB_ASSERT( (!isConvNode || dims == 4), "Convolution output should be 4D");
    unsigned outputChannels = getOutputChannels(mmeNode, isConvNode);


    unsigned originalTensorDims = biasCandidateTensor->getDim();
    TSize originalTensorSizes[Tensor::c_tensorMaxDim];
    biasCandidateTensor->getAllSizesInElements(originalTensorSizes, Tensor::c_tensorMaxDim);

    if (originalTensorDims != dims)
    {
        // resize the tensor to be with the same dimensions as the mme output
        if (!biasCandidateTensor->resizeDims(dims))
        {
            return false;
        }
    }
    if (biasCandidateTensor->getDim() == mmeNode->getOutput(TENSOR_OFM)->getDim())
    {
        const LayoutVector& convOutLayouts = mmeNode->getOutputLayouts();
        HB_ASSERT(convOutLayouts.size() == 1, "mme should have only 1 output layout");

        // set both add input layouts to be the same as the mme output layout (add inputs should have the same layout)
        addNode->setInputLayouts({convOutLayouts[0], convOutLayouts[0]});

        // the tensor's C dimension must match the mme output channels, and that should be the whole tensor (other dims are degenerate)
        if ( (!isConvNode || addNode->inputDimNameToSize(biasCandidate, 'C') == outputChannels) &&
             outputChannels == biasCandidateTensor->getTotalElements())
        {
            //resize to 1D
            TSize sizes[Tensor::c_tensorMaxDim] = {outputChannels, 1, 1, 1, 1};
            biasCandidateTensor->reshape(1, sizes, nullptr);
            LOG_TRACE(GC, "Reshape static tensor {} to be 1D bias of size {}", biasCandidateTensor->getName(), outputChannels);
            return true;
        }
    }

    if (originalTensorDims != dims)
    {
        // we resized the tensor but since it can't be a bias, we should resize back
        if (!biasCandidateTensor->resizeDims(originalTensorDims))
        {
            LOG_ERR(GC, "Can't resize the tensor back to {} dimensions", originalTensorDims);
            throw CannotResizeTensor();
        }
    }
    return false;
}

bool MMEFusionUtils::canBeCin(uint8_t cinCandidate, const pNode& addNode, const pNode& mmeNode)
{
    pTensor cinCandidateTensor = addNode->getInput(cinCandidate);

    return (mmeNode->getInput(TENSOR_CIN) == nullptr // mme node can't have previous CIN
            && cinCandidateTensor->compareGeometry(*mmeNode->getOutput(TENSOR_OFM))); // cin must have the same shape as the mme output
}
