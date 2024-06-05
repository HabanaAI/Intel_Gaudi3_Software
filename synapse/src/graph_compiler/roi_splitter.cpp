#include "roi_splitter.h"

#include <algorithm>
#include <list>
#include <memory>
#include <queue>

#include "code_generator.h"
#include "dma_transpose_node.h"
#include "habana_global_conf.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "split_strategies.h"
#include "types_exception.h"
#include "types.h"
#include "utils.h"

static std::list<NodeROI> splitDmaTranspose(DMANode& node, HabanaGraph& g, NodeROI& baseRoi)
{
    std::list<NodeROI> ret;

    dynamic_cast<DMATransposeNode&>(node).getSplitStrategy()->splitLogical(node,
                                                                           baseRoi,
                                                                           g.getPipelineDepth(node),
                                                                           node.parallelLevel(),
                                                                           ret);

    return ret;
}

void calculateAddressOffset(NodeROI& roi, TensorROI& tensorROI, unsigned elementSizeInBits)
{
    for (int i = 0; i < Tensor::c_tensorMaxNDim; i++)
    {
        uint64_t offset = roi.baseOffset[i];
        if (i == 0)
        {
            tensorROI.getLayout().baseAddress += safeBitsToByte(offset * elementSizeInBits);
        }
        else
        {
            tensorROI.getLayout().baseAddress += offset * tensorROI.getLayout().spatialStrides[i - 1];
        }
    }
}


std::list<NodeROI> ROISplitter::splitBatchGemm(pNode node, HabanaGraph& g) const
{
    return *(g.GetNodeROIs(node)); // currently, common implementation is no split, sub-class can overwrite
}

std::list<NodeROI> ROISplitter::splitTranspose(pNode node, HabanaGraph& g) const
{
    return *(g.GetNodeROIs(node)); // currently, common implementation is no split, sub-class can overwrite
}

std::list<NodeROI> ROISplitter::splitRoiSpatialSize(pNode node, HabanaGraph& g) const
{
    return *(g.GetNodeROIs(node)); // currently, common implementation is no split, sub-class can overwrite
}

std::list<NodeROI> ROISplitter::splitTPC(pNode node, HabanaGraph& g) const
{
    // currently, this default implementation is suitable for gaudi (1&2), sub-class can overwrite
    return splitTPCPipeline(g.GetNodeROIs(node)->front(),
                            node,
                            g.getNumTpcEng(),
                            calculateTPCPipelineDepth(node, g),
                            g);
}

std::list<NodeROI> ROISplitter::splitSpecialKernels(NodeROI& roi, unsigned nTPCEngines)   const
{
    return std::list<NodeROI> {roi};
}

std::list<NodeROI> ROISplitter::splitDMA(pNode node, HabanaGraph& g) const
{
    NodeROI& roi = g.GetNodeROIs(node)->front();
    return splitDMA(node, g, roi);
}

std::list<NodeROI> ROISplitter::splitDMA(pNode node, HabanaGraph& g, NodeROI& baseRoi) const
{
    std::shared_ptr<DMANode> dmaNode = std::dynamic_pointer_cast<DMANode>(node);

    if (dmaNode->isTranspose() && !std::dynamic_pointer_cast<StridedDMANodeViaTransposeNode>(dmaNode))
    {
        return splitDmaTranspose(*dmaNode, g, baseRoi);
    }
    else if (dmaNode->isBroadcast())
    {
        return splitDMABroadcast(node, baseRoi);
    }
    else if (dmaNode->isLinearDma())
    {
        return ROISplitter::splitDenseDMA(node, g, baseRoi);
    }
    else if (dmaNode->isNodeTwoDimensionalStrided())
    {
        return split2dStridedDMA(node, g, baseRoi);
    }
    else
    {
        //The generic case for a strided tensor- same as TPC splitting
        return ROISplitter::splitFullRoiToLogicalRoisAlongExternalAxis(baseRoi,
                                                                       dmaNode->getSplitDimensionsOrder(),
                                                                       g.getDefaultPipelineDepth(),
                                                                       dmaNode->getNodeName());
    }
}

std::list<NodeROI> ROISplitter::splitRotate(pNode node, HabanaGraph& g) const
{
    // Logical split is for K parts along the batch dim
    // The magic value for K is 3
    std::list<NodeROI> ret;
    pTensor tensor = node->getOutput(0);

    NodeROI& roi               = g.GetNodeROIs(node)->front();
    RotateNode* rotateNode     = reinterpret_cast<RotateNode*>(node.get());

    LOG_DEBUG(GC,
              "Splitting Rotate node {} to logical ROIs. Original size: [{}]",
              node->getNodeName(),
              toString(roi.size, roi.size + SYN_MAX_TENSOR_DIM, ','));

    pTensor src = rotateNode->getInput(0);
    pTensor dst = rotateNode->getOutput(0);

    const SizeArray outputDims = dst->getAllSizesInElements();

    uint32_t batchSize = outputDims[NCHW_N_DIM];
    uint32_t numChannels = outputDims[NCHW_C_DIM];
    uint32_t outputHeight = outputDims[NCHW_H_DIM];
    uint32_t outputWidth = outputDims[NCHW_W_DIM];

    // logicalRoi params
    int numLogicalRois = std::min(g.getDefaultPipelineDepth(), batchSize);
    uint32_t batchSizeChunk = batchSize / numLogicalRois;

    for (int logicalRoiIdx=0; logicalRoiIdx < numLogicalRois; logicalRoiIdx++)
    {
        uint32_t logicalRoiBatchStart = logicalRoiIdx * batchSizeChunk;
        uint32_t logicalRoiBatchSize = (logicalRoiIdx == (numLogicalRois - 1)) ?      // if last chunk
                                        batchSize - logicalRoiIdx * batchSizeChunk : batchSizeChunk;
        NodeROI newRoi(roi);
        newRoi.pipelineLevel = logicalRoiIdx;
        // Set the roi sizes
        newRoi.size[4] = 1;
        newRoi.size[NCHW_N_DIM] = logicalRoiBatchSize;
        newRoi.size[NCHW_C_DIM] = numChannels;
        newRoi.size[NCHW_H_DIM] = outputHeight;
        newRoi.size[NCHW_W_DIM] = outputWidth;
        // Set the Roi coordinates
        newRoi.baseOffset[3] = logicalRoiBatchStart;
        newRoi.baseOffset[2] = newRoi.baseOffset[1] = newRoi.baseOffset[0] = 0;
        // Set the Roi strides
        newRoi.spatialStrides[NCHW_N_DIM] = newRoi.spatialStrides[NCHW_C_DIM] = numChannels * outputHeight * outputWidth;
        newRoi.spatialStrides[NCHW_H_DIM] = outputHeight * outputWidth;
        newRoi.spatialStrides[NCHW_W_DIM] = outputWidth;

        ret.push_back(newRoi);
    }

    LOG_TRACE(GC, "Rotate node {} with output tensor {} of size {} was split to {} ROIs",
              node->getNodeName(), dst->getName(), dst->getTotalSizeInBytes(), numLogicalRois);

    return ret;
}

// This DMA split strategy is good for linear DMA where (dense tensors) where we can "flatten"
// the tensor to one dimension and split along that dimension only. Goya uses this DMA split.
// Gaudi and Greco supports tensor DMA for non-dense tensors - use different splitting in this case.
std::list<NodeROI> ROISplitter::splitDenseDMA(pNode node, HabanaGraph& g, NodeROI& roi)
{
    std::list<NodeROI> ret;
    pTensor            tensor = (node->getNumInputsDataTensors() > 0) ? node->getInput(0) : node->getOutput(0);
    std::unique_ptr<CodeGenerator>& codeGenerator = g.getCodeGenerator();
    HB_ASSERT(tensor->isDenseLayout(), "DMA node {} expecting dense tensor ({})", node->getNodeName(), tensor->getName());

    unsigned elementSizeInBits = tensor->getElementSizeInBits();
    uint64_t numElem           = BITS_PER_BYTE * tensor->getTotalSizeInBytes() / elementSizeInBits;
    DMANode* dmaNode           = reinterpret_cast<DMANode*>(node.get());
    uint64_t roiSize           = BITS_PER_BYTE * dmaNode->chunkSizeInBytes() * dmaNode->parallelLevel() /
                       elementSizeInBits;  // goya parallelLevel is always 1, roi size in elements
    unsigned maxNumChunks = codeGenerator->getMaxDMAChunks();

    LOG_DEBUG(ROI_SPLITTER,
              "Splitting linear DMA node {} to logical ROIs. Original size: [{}]",
              node->getNodeName(),
              toString(roi.size, roi.size + SYN_MAX_TENSOR_DIM, ','));

    if (numElem / roiSize > maxNumChunks)
    {
        LOG_TRACE(ROI_SPLITTER, "Increasing roiSize so we won't have more than {} rois", maxNumChunks);
        roiSize = div_round_up(numElem * dmaNode->parallelLevel(), maxNumChunks);
    }

    for (TSize elem = 0; elem < numElem; elem += roiSize)
    {
        NodeROI newRoi(roi);
        memset(newRoi.baseOffset, 0, sizeof(TOffset) * Tensor::c_tensorMaxNDim);
        std::fill_n(newRoi.size, Tensor::c_tensorMaxDim, 1);

        newRoi.baseOffset[0] = elem;
        newRoi.size[0] = std::min<TSize>(roiSize, numElem - elem);
        ret.push_back(newRoi);
    }
    LOG_TRACE(ROI_SPLITTER, "Node {} with tensor {} of size {} was split to {} ROIs",
              node->getNodeName(), tensor->getName(), tensor->getTotalSizeInBytes(), ret.size());

    return ret;
}

// DMA broadcast splitted into log2(tensor size) + 1 ROIs,
// each ROI multipile the output size by factor 2, so we end with the biggest power of 2 that is smaller or equal
// to tensor size, and we add last ROI for the remainder (if needed)
std::list<NodeROI> ROISplitter::splitDMABroadcast(const NodePtr& node, const NodeROI& baseRoi)
{
    std::list<NodeROI> ret;
    const TensorPtr&   input = node->getInput(0);

    HB_ASSERT(input->isDenseLayout(), "DMA node {} expecting dense tensor ({})", node->getNodeName(), input->getName());
    const NodeROI& roi = baseRoi;

    // create the first copy
    NodeROI newRoi(roi);
    memset(newRoi.baseOffset, 0, sizeof(TOffset) * Tensor::c_tensorMaxNDim);
    std::fill_n(newRoi.size, Tensor::c_tensorMaxDim, 1);

    newRoi.size[0] = input->getDenseSizeInElements();
    ret.push_back(newRoi);

    const TSize broadcastSize        = node->getOutput(0)->getDenseSizeInElements() / newRoi.size[0];
    TSize       currentBroadcastSize = broadcastSize;
    TSize       currentScdSize       = 1;
    while (currentBroadcastSize != 1)
    {
        newRoi.size[0]       = input->getDenseSizeInElements() * currentScdSize;
        newRoi.baseOffset[0] = newRoi.size[0];
        ret.push_back(newRoi);

        currentBroadcastSize /= 2;
        currentScdSize *= 2;
    }

    TSize remainder = broadcastSize - currentScdSize;
    if (remainder != 0)
    {
        newRoi.size[0]       = input->getDenseSizeInElements() * remainder;
        newRoi.baseOffset[0] = input->getDenseSizeInElements() * currentScdSize;

        ret.push_back(newRoi);
    }
    LOG_TRACE(ROI_SPLITTER,
              "Node {} with tensor {} of size {} was split to {} ROIs",
              node->getNodeName(),
              input->getName(),
              input->getTotalSizeInBytes(),
              ret.size());

    return ret;
}

std::list<NodeROI> ROISplitter::split2dStridedDMA(pNode node, HabanaGraph& g, NodeROI& baseRoi)
{
    //------------------------------------------------------------------------------------
    // for this splitting strategy, we assume strided tensor has no more than 2 dimensions
    //------------------------------------------------------------------------------------

    std::list<NodeROI>        ret;
    pTensor                   tensor = (node->getNumInputsDataTensors() > 0) ? node->getInput(0) : node->getOutput(0);
    std::shared_ptr<DMANode>  dmaNode      = std::dynamic_pointer_cast<DMANode>(node);
    TSize                     numElem      = tensor->getDenseSizeInElements();
    TSize                     dim0Elems    = tensor->getSizeInElements(0);
    TSize                     roiSize      = 0;
    TSize                     sizes[Tensor::c_tensorMaxNDim];
    unsigned                  maxNumChunks = g.getCodeGenerator()->getMaxDMAChunks();

    HB_ASSERT(baseRoi.size[2] == 1 && baseRoi.size[3] == 1 && baseRoi.size[4] == 1,
              "this function assumes strided tensor has only 2 dims");
    LOG_DEBUG(ROI_SPLITTER,
              "Splitting strided DMA node {} to logical ROIs. Original size: [{}]",
              node->getNodeName(),
              toString(baseRoi.size, baseRoi.size + SYN_MAX_TENSOR_DIM, ','));

    std::fill_n(sizes, Tensor::c_tensorMaxNDim, 1);

    sizes[0] = dim0Elems;
    sizes[1] = div_round_up(dmaNode->chunkSizeInBytes(), tensor->getSizeInBytes(0)) * dmaNode->parallelLevel();
    roiSize  = sizes[0] * sizes[1];

    if (numElem / roiSize > maxNumChunks)
    {
        LOG_TRACE(ROI_SPLITTER, "Increasing roiSize so we won't have more than {} rois", maxNumChunks);
        sizes[1] = div_round_up(numElem * dmaNode->parallelLevel(), maxNumChunks);
        roiSize  = sizes[0] * sizes[1];
    }

    for (TSize elem = 0; elem < numElem; elem += roiSize)
    {
        NodeROI newRoi(baseRoi);
        memset(newRoi.baseOffset, 0, sizeof(TOffset) * Tensor::c_tensorMaxNDim);

        if (roiSize > numElem - elem)
        {
            TSize reminder = numElem - elem;
            sizes[1] = reminder / dim0Elems;
            HB_ASSERT(reminder % dim0Elems == 0, "reminder didn't divide to dim0Elems");
        }

        // Set ROI base offset index
        findIndex(newRoi.size, 2, elem, newRoi.baseOffset);

        castNcopy(newRoi.size, sizes, Tensor::c_tensorMaxNDim);
        ret.push_back(newRoi);
    }

    LOG_TRACE(ROI_SPLITTER, "Node {} with tensor {} of size {} was split to {} ROIs",
              node->getNodeName(), tensor->getName(), tensor->getTotalSizeInBytes(), ret.size());

    return ret;
}

bool ROISplitter::splitAllNodes(HabanaGraph& g) const
{
    if (g.isDebugMode()) return true;
    for (const pNode& n : g.getExeSortedNodes())
    {
        validateDcoreRoi(n);
        if (n->isLogicalOperation() || !n->getNodeAnnotation().splitToLogicalROIs) continue;
        if (!splitNode(g, n)) return false;
    }
    return true;
}

void ROISplitter::validateDcoreRoi(pNode node) const
{
    if (node->getNodeAnnotation().m_dcoreROIs.size() > 0)
    {
        HB_ASSERT(!node->getNodeAnnotation().splitToLogicalROIs, "when using locality can't split to Rois");
    }
    if (node->getNodeAnnotation().splitToLogicalROIs)
    {
        HB_ASSERT(node->getNodeAnnotation().m_dcoreROIs.size() == 0, "when using locality can't split to Rois");
    }
}
bool ROISplitter::splitNode(HabanaGraph& graph, pNode node) const
{
    HB_ASSERT(!node->isLogicalOperation() && node->getNodeAnnotation().splitToLogicalROIs,
              "cannot split node to logical ROIs");
    std::list<NodeROI>* rois = graph.GetNodeROIs(node);

    bool isMME    = HabanaGraph::runsOnMME(node);
    bool isTPC    = HabanaGraph::runsOnTPC(node);
    bool isDMA    = node->isDma();
    bool isRotate = node->isRotate();

    if (rois->size() > 1)
    {
        LOG_WARN(ROI_SPLITTER,
                 "Expected only one ROI per node at this stage, got {} for node {}",
                 rois->size(),
                 node->getNodeName());
    }

    std::list<NodeROI> newROIs;

    if (isMME)
    {
        if (node->getNodeType() == Node::TYPE_BATCH_GEMM || node->getNodeType() == Node::TYPE_BATCH_GEMM_DEDX ||
            node->getNodeType() == Node::TYPE_BATCH_GEMM_DEDW || node->getNodeType() == Node::TYPE_MASKED_BATCH_GEMM)
        {
            newROIs = splitBatchGemm(node, graph);
        }
        else if (node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE)
        {
            newROIs = splitTranspose(node, graph);
        }
        else
        {
            newROIs = splitRoiSpatialSize(node, graph);
        }
    }
    else if (isTPC)
    {
        newROIs = splitTPC(node, graph);
    }
    else if (isDMA)
    {
        newROIs = splitDMA(node, graph);
    }
    else if (isRotate)
    {
        newROIs = splitRotate(node, graph);
    }
    else
    {
        LOG_WARN(ROI_SPLITTER,
                 "node {} is not MME, TPC or DMA, don't know how to split it to ROIs",
                 node->getNodeName());
        return true;
    }

    unsigned currPipelineLevel = 0;

    for (NodeROI& roi : newROIs)
    {
        roi.pipelineLevel = currPipelineLevel;
        ++currPipelineLevel;
    }

    rois->clear();
    *rois = newROIs;

    return true;
};

std::vector<TSize> ROISplitter::splitSamplesIntoNonzeroPipeline(TSize    numSamples,
                                                                unsigned pipelineDepth,
                                                                unsigned numOfPhysicalEngs)
{
    if (numSamples >= pipelineDepth)
    {
        return splitToChunks(numSamples, pipelineDepth, 0, numOfPhysicalEngs);
    }
    std::vector<TSize> ret;
    ret.assign(numSamples, 1);
    return ret;
}

std::list<NodeROI> ROISplitter::splitFullRoiToLogicalRoisAlongExternalAxis(const NodeROI&     roi,
                                                                           const DimVector&   splitDimsOrder,
                                                                           unsigned           nChunks,
                                                                           const std::string& nodeName,
                                                                           unsigned           numOfPhysicalEngs)
{
    std::list<NodeROI> ret;
    ret.push_back(roi);

    std::list<NodeROI> tmp;
    validateSplitInput(roi.size, splitDimsOrder, nChunks);

    LOG_DEBUG(ROI_SPLITTER, "Splitting the index space to {} chunks for pipelining", nChunks);

    for (unsigned dimIdx : splitDimsOrder)
    {
        tmp.clear();
        if (nChunks < 1) break;
        std::vector<TSize> chunks = splitSamplesIntoNonzeroPipeline(roi.size[dimIdx], nChunks, numOfPhysicalEngs);

        //Split all existing ROIs along <dimIdx> into <sz> chunks
        for (NodeROI& chunk : ret)
        {
            LOG_DEBUG(ROI_SPLITTER, "Splitting dim {} into {} chunks", dimIdx, chunks.size());
            TOffset offset = chunk.baseOffset[dimIdx];
            for (TSize c : chunks)
            {
                chunk.baseOffset[dimIdx] = offset;
                chunk.size[dimIdx]       = c;
                offset                  += c;
                tmp.push_back(chunk);
            }
        }
        ret     = tmp;
        nChunks = div_round_up(nChunks, chunks.size());
    }
    if (nChunks > 1)
    {
        LOG_WARN(ROI_SPLITTER, "Could not fully pipeline index space for {}", nodeName);
    }
    return ret;
}

void ROISplitter::validateSplitInput(const TSize* roiSize, const DimVector& dimPreference, unsigned nChunks)
{
    if (nChunks == 0)
    {
        LOG_ERR(ROI_SPLITTER, "Can not split NodeROI into 0 chunks");
        throw InvalidPipelineParamsException();
    }
    for (unsigned dimIdx: dimPreference)
    {
        if (roiSize[dimIdx] == 0)
        {
            LOG_ERR(ROI_SPLITTER, "Index space size of dimension {} is 0", dimIdx);
            throw InvalidPipelineParamsException();
        }
    }
}

std::list<NodeROI> ROISplitter::SplitROIBetweenEngines(NodeROI&         roi,
                                                       const DimVector& splitDimsOrder,
                                                       unsigned         nEngines,
                                                       unsigned&        nextSubSplitStartIndex,
                                                       unsigned&        nextMajorSplitStartIndex,
                                                       bool             preferSplitOnFcd)
{
    unsigned nChunks = nEngines;

    DimVector dimPreference = splitDimsOrder;
    if (std::find(dimPreference.begin(), dimPreference.end(), 0) != dimPreference.end())
    {
        if (preferSplitOnFcd)
        {
            // Using the old but gold dim-preference heuristic
            // This dim prefernce was chosen for goya's TPC after trail and error and comparing
            // on couple main topologies. Need to check relevance for today
            dimPreference.erase(std::remove(dimPreference.begin(), dimPreference.end(), 0));
            dimPreference.insert(dimPreference.begin(), 0);
        }
        else if (dimPreference.size() != 1)
        {
            dimPreference.erase(std::remove(dimPreference.begin(), dimPreference.end(), 0));
        }
    }

    LOG_DEBUG(ROI_SPLITTER,
              "Splitting an ROI sized [{}] between {} TPC engines",
              toString(roi.size, roi.size + SYN_MAX_TENSOR_DIM, ','),
              nEngines);

    ROISplitter::validateSplitInput(roi.size, dimPreference, nChunks);

    return splitAllSamples(roi, dimPreference, nChunks, nChunks, nextSubSplitStartIndex, nextMajorSplitStartIndex);
}

std::vector<TSize> ROISplitter::splitSamplesIntoPerfectSplit(TSize numSamples, unsigned pipelineDepth)
{
    std::vector<TSize> ret;
    ret.assign(pipelineDepth, numSamples/pipelineDepth);
    return ret;
}


std::vector<TSize> ROISplitter::perfectSplitIfPossible(TSize numSamples, unsigned pipelineDepth)
{
    if (pipelineDepth != 0)
    {
        TSize samplePipelineGCD = gcd(numSamples, static_cast<TSize>(pipelineDepth));

        if (samplePipelineGCD == pipelineDepth) /* Can simply split into pipelineDepth */
        {
            return splitSamplesIntoPerfectSplit(numSamples, pipelineDepth);
        }
        else if (samplePipelineGCD == numSamples)
        {
            return splitSamplesIntoPerfectSplit(numSamples, numSamples);
        }
    }
    std::vector<TSize> ret;
    ret.push_back(numSamples);
    return ret;
}

template<class CONTAINER>
void splitNodeRoiChunks(NodeROI&                chunk,
                        std::vector<TSize>      pipelineSplit,
                        CONTAINER&              outList,
                        unsigned                dimIdx)
{
    int64_t offset = chunk.baseOffset[dimIdx];
    for (TSize c : pipelineSplit)
    {
        chunk.baseOffset[dimIdx] = offset;
        chunk.size[dimIdx]       = c;
        offset                  += c;
        outList.push_back(chunk);
    }
}

std::list<NodeROI> ROISplitter::translateSplitToNodeROI(NodeROI&                               roi,
                                                        const DimVector&                       dimPreference,
                                                        const std::vector<std::vector<TSize>>& chunks)
{
    std::list<NodeROI> ret;
    ret.push_back(roi);

    std::list<NodeROI> tmp;
    for (unsigned i = 0 ; i < chunks.size() ; ++i)
    {
        unsigned dimIdx = dimPreference[i];
        tmp.clear();
        std::vector<TSize> dimChunks = chunks[i];

        for (NodeROI& node : ret)
        {
            splitNodeRoiChunks<std::list<NodeROI>>(node, dimChunks, tmp, dimIdx);
        }
        ret = tmp;
    }
    return ret;
}

void ROISplitter::deployPerfectSplitStrategy(const TSize*                        roiSize,
                                             const DimVector&                    dimPreference,
                                             unsigned&                           chunksLeft,
                                             std::vector<std::vector<TSize>>&    chunks)
{
    for (unsigned dimIdx: dimPreference)
    {
        if (chunksLeft == 1) break;
        std::vector<TSize> splitOut = perfectSplitIfPossible(roiSize[dimIdx], chunksLeft);
        auto splitOutSize = static_cast<unsigned>(splitOut.size());

        // check if succeeded to split
        if (splitOutSize == 1 && splitOut[0] != 1) break;

        chunks.push_back(splitOut);
        chunksLeft = chunksLeft / splitOutSize;
    }
}

void ROISplitter::deployZeroRemainderStrategy(const TSize*                        roiSize,
                                              const DimVector&                    dimPreference,
                                              unsigned&                           chunksLeft,
                                              unsigned                            nChunks,
                                              std::vector<std::vector<TSize>>&    chunks,
                                              unsigned                            maxChunks)
{
    TSize zrDivLeft = chunksLeft;
    std::vector<std::vector<TSize>> zrChunks = chunks;

    for(auto i = static_cast<unsigned>(chunks.size()); i < dimPreference.size(); ++i)
    {
        unsigned dimIdx = dimPreference[i];
        auto     gcdChunks = gcd(roiSize[dimIdx], zrDivLeft);
        zrDivLeft = zrDivLeft / gcdChunks;

        // check if done
        if (zrDivLeft == 1)
        {
            std::vector<TSize> zrPerfectSplit = splitSamplesIntoPerfectSplit(roiSize[dimIdx], gcdChunks);
            zrChunks.push_back(zrPerfectSplit);
            chunksLeft = zrDivLeft;
            chunks = zrChunks;
            return;
        }

        // prep for next dimension:
        zrChunks.push_back(splitSamplesIntoPerfectSplit(roiSize[dimIdx], gcdChunks));
    }
}


// as we reach this function per logical roi and split it to physical rois, if all logical rois looks similar (as is
// often the case), but are not split perfectly into the number of physical engines, we will have a problem of unbalanced
// splitting, as each time the same engines (in the typical case the first ones) will get the bigger chunks.
// in order to solve that, we try to do a "round robin" by maintaining nextSubSplitStartIndex and nextMajorSplitStartIndex
//
// when we split an roi in a given dimension, we have 2 main scenarios:
// 1. When roi size (in the chosen dimension) >= number of chunks we want to split to (pipeline depth)
// in that case we will split the roi to x chunks with some chunk size and (total chunks - x) chunks with (chunk size + 1)
// (for this case we use nextSubSplitStartIndex param)
// i.e. we will split 11 into 8 by having 3 chunks of size 2 and 5 chunks of size 1
// 2. When roi size (in the chosen dimension) < number of chunks we want to split to (pipeline depth)
// in that case we will split the roi completely in that dimension, but in addition, we will sub-split more dimensions
// i.e. if we have an roi of size [4,11] and we want to split to 6, we could split it to [1,11], [1,11], [1,6], [1,5], [1,6], [1,5]
// (for this case, for the "major split" part, we use nextMajorSplitStartIndex param)
void ROISplitter::deployPipelineStrategy(const DimVector&               dimPreference,
                                         std::queue<PipelineSplitInfo>& nodesToSplit,
                                         std::list<NodeROI>&            finalSplit,
                                         unsigned&                      nextSubSplitStartIndex,  // in/out parameter
                                         unsigned&                      nextMajorSplitStartIndex)                     // in/out parameter
{
    unsigned curSubSplitStartIndex = nextSubSplitStartIndex;
    unsigned totalNumChunksToSplit = 0, totalSubSplittedChunks = 0;
    if (!nodesToSplit.empty())
    {    // keep the "main" number of chunks we want to split to when entering this function for the given logical roi
        totalNumChunksToSplit = nodesToSplit.front().chunksToSplit;
    }
    while (!nodesToSplit.empty())
    {
        PipelineSplitInfo pInfo = nodesToSplit.front();
        nodesToSplit.pop();

        // check if we finished iterating over the dimension preference
        if (pInfo.workingDimensionIdx > (dimPreference.size() - 1))
        {
            finalSplit.push_back(pInfo.nodeToSplit);
            continue;
        }

        unsigned dimIdx = dimPreference[pInfo.workingDimensionIdx];
        TSize currRoiSize[Tensor::c_tensorMaxNDim];
        memcpy(currRoiSize, pInfo.nodeToSplit.size, sizeof(pInfo.nodeToSplit.size));

        if (pInfo.chunksToSplit <= currRoiSize[dimIdx])
        {
            // note that we since inside the scope of this function we deal with a single logical ROI, we use the same
            // curSubSplitStartIndex here, while nextSubSplitStartIndex is being updated after each call (and is used
            // in later calls to deployPipelineStrategy())
            std::vector<TSize> pipelineSplit = splitToChunksWithIndexes(currRoiSize[dimIdx],
                                                                        pInfo.chunksToSplit,
                                                                        curSubSplitStartIndex,
                                                                        nextSubSplitStartIndex);
            splitNodeRoiChunks<std::list<NodeROI>>(pInfo.nodeToSplit, pipelineSplit, finalSplit, dimIdx);
            if (pInfo.chunksToSplit < currRoiSize[dimIdx]) // if these are equal we do not count it as a real "sub split"
            {
                totalSubSplittedChunks += pInfo.chunksToSplit;
            }
        }
        else if (pInfo.chunksToSplit > currRoiSize[dimIdx])
        {
            std::vector<TSize> pipelineSplit = splitSamplesIntoPerfectSplit(currRoiSize[dimIdx], currRoiSize[dimIdx]);
            std::vector<NodeROI> onesNodes;
            splitNodeRoiChunks<std::vector<NodeROI>>(pInfo.nodeToSplit, pipelineSplit, onesNodes, dimIdx);
            std::vector<TSize> revPipelineSplit = splitToChunks(pInfo.chunksToSplit, currRoiSize[dimIdx]);
            for (TSize i = 0 ; i < currRoiSize[dimIdx] ; ++i)
            {
                if (revPipelineSplit[i] == 1)
                {
                    // Can not pipeline this piece anymore
                    finalSplit.push_back(onesNodes[i]);
                }
                else
                {
                    // We can try and split it again in the next dimension
                    PipelineSplitInfo newSplit(revPipelineSplit[i], pInfo.workingDimensionIdx + 1, onesNodes[i]);
                    nodesToSplit.push(newSplit);
                }
            }
        }
    }

    if  (totalNumChunksToSplit > 0 && totalSubSplittedChunks > 0 && totalSubSplittedChunks < totalNumChunksToSplit)
    {
        // since we first inserted to finalSplit the rois that were not sub splitted (hence only splitted in one dimension),
        // in order to balance them with the previous logical roi, we need to to move nextMajorSplitStartIndex from the end
        // to the beginning (or move totalNumChunksToSplit - nextMajorSplitStartIndex from the beginning to the end)
        unsigned numElemsMoveToEnd = totalNumChunksToSplit - nextMajorSplitStartIndex;
        finalSplit.splice(finalSplit.end(), finalSplit, finalSplit.begin(), std::next(finalSplit.begin(), numElemsMoveToEnd));
        // update nextMajorSplitStartIndex for future calls to this function:
        unsigned totalNotSubSplittedChunks = totalNumChunksToSplit - totalSubSplittedChunks;
        nextMajorSplitStartIndex = (nextMajorSplitStartIndex + totalNotSubSplittedChunks) % totalNumChunksToSplit;
    }
}

std::list<NodeROI> ROISplitter::splitAllSamples(NodeROI&         roi,
                                                const DimVector& dimPreference,
                                                unsigned         nChunks,
                                                unsigned         maxChunks,
                                                unsigned&        nextSubSplitStartIndex,
                                                unsigned&        nextMajorSplitStartIndex)
{
    std::vector<std::vector<TSize>> chunks;
    unsigned nChunksLeft = nChunks;

    // Startegy A
    // perfect-split
    deployPerfectSplitStrategy(roi.size, dimPreference, nChunksLeft, chunks);
    if (nChunksLeft == 1 || (chunks.size() == dimPreference.size()))
    {
        return translateSplitToNodeROI(roi, dimPreference, chunks);
    }

    // Strategy B
    // 0-remainder
    deployZeroRemainderStrategy(roi.size, dimPreference, nChunksLeft, nChunks, chunks, maxChunks);
    if (nChunksLeft == 1)
    {
        return translateSplitToNodeROI(roi, dimPreference, chunks);
    }

    // translate chunks mode to NodeROIs
    std::list<NodeROI> currSplit;
    if (nChunks > nChunksLeft)
    {
        currSplit = translateSplitToNodeROI(roi, dimPreference, chunks);
    }
    else
    {
        currSplit.push_back(roi);
    }

    // Strategy C
    // simple-pipeline
    std::list<NodeROI> finalSplit;
    std::queue<PipelineSplitInfo> nodesInfoToSplit;
    for (NodeROI& node: currSplit)
    {
        PipelineSplitInfo pInfo(nChunksLeft, 0, node);
        nodesInfoToSplit.push(pInfo);
    }

    deployPipelineStrategy(dimPreference, nodesInfoToSplit, finalSplit, nextSubSplitStartIndex, nextMajorSplitStartIndex);

    return finalSplit;
}

std::list<NodeROI> ROISplitter::splitRoisBetweenEngines(std::list<NodeROI>& roisToSplit,
                                                        const DimVector&    splitDimsOrder,
                                                        unsigned            nEngines,
                                                        const std::string&  nodeName,
                                                        bool                preferSplitOnFcd) const
{
    std::list<NodeROI> ret;
    // maintenance variables for spreading the work more evenly between different physical engines in different logical ROIs:
    unsigned nextSubSplitStartIndex = 0, nextMajorSplitStartIndex = 0;
    for (NodeROI &nr : roisToSplit)
    {
        std::list<NodeROI> r;
        r = ROISplitter::SplitROIBetweenEngines(nr,
                                                splitDimsOrder,
                                                nEngines,
                                                nextSubSplitStartIndex,
                                                nextMajorSplitStartIndex,
                                                preferSplitOnFcd);
        ret.splice(ret.end(), r);
    }

    for (const auto& roi: ret)
    {
        LOG_DEBUG(ROI_SPLITTER,
                  "[CWHN] ROI size: [{}], ROI base: [{}]",
                  toString(roi.size, roi.size + SYN_MAX_TENSOR_DIM, ','),
                  toString(roi.baseOffset, roi.baseOffset + SYN_MAX_TENSOR_DIM, ','));
    }
    return ret;
}

std::list<NodeROI> ROISplitter::splitSpecialKernelsBetweenEngines(std::list<NodeROI>& roisToSplit, unsigned nTPCEngines)
{
    std::list<NodeROI> ret;
    for (const NodeROI& roi : roisToSplit)
    {
        NodeROI tmp = roi;
        std::vector<TSize> widthPerEngine  = splitToChunks(roi.size[DIM_W], nTPCEngines);
        unsigned wIdx = 0;
        for (unsigned w : widthPerEngine)
        {
            tmp.baseOffset[DIM_W] = wIdx;
            tmp.size[DIM_W]       = w;
            ret.push_back(tmp);
            wIdx += w;
        }

    }
    return ret;
}

std::list<NodeROI> ROISplitter::splitTPCPipeline(NodeROI&       roi,
                                                 const NodePtr& n,
                                                 unsigned       nEngines,
                                                 unsigned       pipeDepth,
                                                 HabanaGraph&   g) const
{
    std::list<NodeROI> ret;
    ret.push_back(roi);

    if (!n || !HabanaGraph::runsOnTPC(n)) return ret;
    const auto& tpcNode = static_cast<TPCNode&>(*n);

    //Work-around for buggy kernels with illegal index spaces
    const auto& instance = tpcNode.getInstance();
    if (instance.indexSpaceRank == 0)
    {
        LOG_WARN(ROI_SPLITTER, "Not splitting node {} because its index-space is empty", n->getNodeName());
        return ret;
    }
    TSize roiTotalSize = multiplyElements(roi.size, roi.size + instance.indexSpaceRank);
    if (roiTotalSize <= 1)
    {
        LOG_WARN(ROI_SPLITTER, "Not splitting node {} because its index-space is atomic", n->getNodeName());
        return ret;
    }
    if (tpcNode.isLoweringKernel())
    {
        // TODO: Do more intelligent decision of TPC ROI split
        return splitSpecialKernels(roi, nEngines);
    }

    //Split ROIs for pipelining
    LOG_DEBUG(ROI_SPLITTER,
              "Splitting index space of {}. Original size: [{}]",
              tpcNode.getNodeName(),
              toString(roi.size, roi.size + SYN_MAX_TENSOR_DIM, ','));

    return ROISplitter::splitFullRoiToLogicalRoisAlongExternalAxis(
        roi,
        tpcNode.getNodeAnnotation().tpcSplitDims,
        pipeDepth,
        tpcNode.getNodeName(),
        getNumEnginesForDeviceType(g.getNodeUtility().getNodeDeviceType(n), *g.getHALReader()));
}

std::list<NodeROI> ROISplitter::projectDmaRoisToFullSizes(const Node* node, std::list<NodeROI>& roisToExpand)
{
    // Currently we don't project to full sizes, but just calculate the memory offsets since
    // the original ROI size is currently EQUAL to the tensor size.
    std::list<NodeROI> returnList;

    //We use the output tensor because there will not be an input in a memset node.
    unsigned int elementSizeInBits = node->getOutput(0)->getElementSizeInBits();

    //For now we just calculate the base addresses between the split ROIs.
    for (NodeROI &roi : roisToExpand)
    {
        if(roi.outputRois.empty())
        {
            LOG_ERR(ROI_SPLITTER, "projectDmaRoisToFullSizes No output roi for node {}", node->getNodeName());
            throw InvalidNodeParamsException(node->getNodeName());
        }

        if (roi.inputRois.size())
        {
            calculateAddressOffset(roi, roi.inputRois[0], elementSizeInBits);
            std::copy(roi.size, roi.size + Tensor::c_tensorMaxDim, roi.inputRois[0].getLayout().m_size.data());
            std::copy(roi.baseOffset, roi.baseOffset + Tensor::c_tensorMaxDim, roi.inputRois[0].getLayout().m_baseOffset);
        }
        calculateAddressOffset(roi, roi.outputRois[0], elementSizeInBits);
        std::copy(roi.size, roi.size + Tensor::c_tensorMaxDim, roi.outputRois[0].getLayout().m_size.data());
        std::copy(roi.baseOffset, roi.baseOffset + Tensor::c_tensorMaxDim, roi.outputRois[0].getLayout().m_baseOffset);
        returnList.push_back(roi);
    }
    return returnList;
}

std::list<NodeROI> ROISplitter::projectTPCRois(const pNode& node, std::list<NodeROI>&  rois)
{
    std::list<NodeROI> returnList;
    for (NodeROI& roi : rois)
    {
        int roiIndex = 0;
        for (unsigned inputIdx = 0; inputIdx < node->getNumInputs(); ++inputIdx)
        {
            auto input = node->getInput(inputIdx);
            if (input != nullptr && input->isShapeTensor()) continue;
            NodeROI inputRoi = node->getInputROI(roi, inputIdx).value();
            memcpy(roi.inputRois[roiIndex].getLayout().m_size.data(), inputRoi.size, sizeof(inputRoi.size));
            memcpy(roi.inputRois[roiIndex].getLayout().m_baseOffset, inputRoi.baseOffset, sizeof(inputRoi.baseOffset));
            ++roiIndex;
        }

        roiIndex = 0;
        for (unsigned outputIdx = 0; outputIdx < node->getNumOutputs(); ++outputIdx)
        {
            auto output = node->getOutput(outputIdx);
            if (output != nullptr && output->isShapeTensor()) continue;
            NodeROI outputRoi = node->getOutputROI(roi, outputIdx).value();
            memcpy(roi.outputRois[roiIndex].getLayout().m_size.data(), outputRoi.size, sizeof(outputRoi.size));
            memcpy(roi.outputRois[roiIndex].getLayout().m_baseOffset, outputRoi.baseOffset, sizeof(outputRoi.baseOffset));
            ++roiIndex;
        }

        returnList.push_back(roi);
    }

    return returnList;
}

unsigned ROISplitter::calculateTPCPipelineDepth(const NodePtr& node, const HabanaGraph& g)
{
    unsigned pipeLineDepth = g.getPipelineDepth(node);

    if (!node || !HabanaGraph::runsOnTPC(node))
    {
        LOG_ERR(ROI_SPLITTER, "Failed to cast {} to TPC node", node->getNodeName());
        HB_ASSERT(node, "Failed to cast TPC node");
        return pipeLineDepth;
    }
    const auto& tpcNode = static_cast<TPCNode&>(*node);

    const uint64_t* indexSpaceSizes    = tpcNode.getInstance().indexSpaceGeometry;
    unsigned        dim                = tpcNode.getInstance().indexSpaceRank;
    unsigned indexSpaceElements = multiplyElements(indexSpaceSizes, indexSpaceSizes + dim);
    unsigned        indexSpaceDepth =
        std::max((unsigned)(indexSpaceElements / (g.getNumTpcEng() * GCFG_MIN_TPC_PIPELINE_FACTOR.value())), 1U);
    pipeLineDepth = std::min(pipeLineDepth, indexSpaceDepth);
    return pipeLineDepth;
}
