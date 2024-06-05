#include "code_generation/tensor_size_validator.h"
#include "infra/threads/thread_pool.h"
#include "infra/threads/thread_work_item.h"
#include "habana_graph.h"
#include "habana_nodes.h"
#include "passes.h"

#include "tpc_node.h"
#include "roi_splitter.h"
#include "defs.h"
#include "dma_transpose_helper.h"
#include "infra/math_utils.h"
#include "dma_transpose_node.h"
#include "split_strategies.h"
#include "transpose_utils.h"
#include "graph_traits.h"
#include "split_to_physical_rois.h"

static bool isHugeTensorROI(const HabanaGraph& g, const NodePtr& n, const TensorROI& roi)
{
    const TensorPtr& t = roi.m_parentTensor;
    if (!t->isDataTensor()) return false;
    unsigned dim = roi.m_layout.tensorDim;

    // tensor ROI strides
    NStrideArray strides    = {1};
    const auto&  roiStrides = roi.getStridesWithFcdDim();
    std::copy(roiStrides.begin(), roiStrides.end(), strides.begin());

    return !TensorSizeValidator(/* print only on trace */ static_cast<unsigned>(SPDLOG_LEVEL_TRACE))
                .validateTensor(n, t, roi.m_layout.m_size, strides);
}

static bool isHugeTensorROI(const HabanaGraph& g, const NodePtr& n, const std::list<NodeROI>& rois)
{
    for (const auto& nodeRoi : rois)
    {
        auto isHugeRoi = [&g, &n](const TensorROI& roi) { return isHugeTensorROI(g, n, roi); };
        if (std::any_of(nodeRoi.inputRois.begin(), nodeRoi.inputRois.end(), isHugeRoi)) return true;
        if (std::any_of(nodeRoi.outputRois.begin(), nodeRoi.outputRois.end(), isHugeRoi)) return true;
    }
    return false;
}

std::vector<TSize>
calculateHugeDmaChunks(const TSize roiSize, const TStride srcStride, const TStride dstStride, uint64_t maxRegVal)
{
    LOG_TRACE(ROI_SPLITTER,
              "Huge ROI Size: {}. srcStride: {}, dstStride: {}, maxRegVal: {}",
              roiSize,
              srcStride,
              dstStride,
              maxRegVal);

    std::vector<TSize> ret;
    uint64_t           sizeRemainder = roiSize;
    unsigned           numChunks     = 0;
    if ((roiSize - 1) * std::max(srcStride, dstStride) < maxRegVal)
    {
        ret.assign(1, roiSize);
        return ret;
    }
    TSize chunkSize = std::min(roiSize, (maxRegVal / std::max(srcStride, dstStride) + 1));
    while (sizeRemainder)
    {
        ret.push_back(std::min(chunkSize, sizeRemainder));
        sizeRemainder = sizeRemainder > chunkSize ? sizeRemainder - chunkSize : 0;
        numChunks++;
    }
    LOG_DEBUG(ROI_SPLITTER,
              "Split Huge Tensor of Original Size: {} to {} of size: {} each",
              roiSize,
              numChunks,
              chunkSize);
    return ret;
}

// This function enables support for Huge Tensor memcopy (tensor values exceed 32 bit representation)
// Given the shape [x_0,x_1,…,x_N ] and the strides [s_0,s_1,…,s_N ]
// According to H6 DMA Architectural Specification.docx, We must keep the following limitations:
// s_i∙(x_i - 1) < 2^32 – due to src/dst_offset 32b width
// x_i < 2^32 – due to src/dst_size 32b width
// The condition s_i∙(x_i-1) < (2^32-1) satisfies both (1) and (2) (since s_i ≥ 1).
// From conditions (1) and (2) we can set:
// x ̃_i = min⁡(x_i, floor((2^32-1)  / max⁡(s_i_src, s_i_dst))+1)

std::list<NodeROI> splitHugeDma(ROISplitter&        roiSplitter,
                                std::list<NodeROI>& roisToSplit,
                                const DimVector&    splitDimsOrder,
                                uint64_t            maxRegVal,
                                unsigned            nEngines,
                                const std::string&  nodeName)
{
    std::list<NodeROI> ret;
    for (NodeROI& roi : roisToSplit)
    {
        std::list<NodeROI> newRoi;
        newRoi.push_back(roi);
        std::list<NodeROI> tmpRoiSplit;

        // get tensor strides. gaudi1 memset is set as DMA copy, so no input in that case.
        std::optional<TensorStridesVector> inputStrides;
        if (roi.inputRois.size() > 0)
        {
            inputStrides = roi.inputRois[0].getStridesWithFcdDim();
        }
        const auto& outputStrides = roi.outputRois[0].getStridesWithFcdDim();

        for (unsigned dimIdx : splitDimsOrder)  // no meaning for split order but minimizes bunmer of iterations
        {
            tmpRoiSplit.clear();
            TStride            srcStride = inputStrides.has_value() ? inputStrides.value()[dimIdx] : 0;
            TStride            dstStride = outputStrides[dimIdx];
            std::vector<TSize> chunks    = calculateHugeDmaChunks(roi.size[dimIdx], srcStride, dstStride, maxRegVal);

            // Split all existing ROIs along <dimIdx> into <sz> chunks
            for (NodeROI& chunk : newRoi)
            {
                LOG_DEBUG(ROI_SPLITTER, "Splitting dim {} into {} chunks", dimIdx, chunks.size());
                TOffset offset = chunk.baseOffset[dimIdx];
                for (TSize c : chunks)
                {
                    chunk.baseOffset[dimIdx] = offset;
                    chunk.size[dimIdx]       = c;
                    offset += c;
                    tmpRoiSplit.push_back(chunk);
                }
            }
            newRoi = tmpRoiSplit;
        }
        newRoi = roiSplitter.splitRoisBetweenEngines(newRoi,
                                                     splitDimsOrder,
                                                     nEngines,
                                                     nodeName,
                                                     false /* don't prefer split on FCD*/);
        SplitStrategy::updateNumSignals(newRoi.begin(), newRoi.end(), nEngines);
        ret.splice(ret.end(), newRoi);
    }
    return ret;
}

static std::list<NodeROI> split2dStridedDmaRoiToPhysicalRois(std::list<NodeROI>& roisToSplit,
                                                             unsigned            nEngines)
{
    //------------------------------------------------------------------------------------
    // For this splitting strategy, we assume strided tensor has no more than 2 dimensions
    //------------------------------------------------------------------------------------
    unsigned inOffsetAccum = 0;
    unsigned outOffsetAccum = 0;
    std::list<NodeROI>  ret;
    for (NodeROI &nr : roisToSplit)
    {
        HB_ASSERT(nr.size[2] == 1 && nr.size[3] == 1 && nr.size[4] == 1, "this function assumes strided tensor has only 2 dims");


        NodeROI             baseRoi(nr);
        unsigned            reminder         = 0;
        unsigned            workDistributed  = 0;
        unsigned            workingEngines   = 0;
        unsigned            baseOffset       = 0;
        uint64_t            totalSize        = static_cast<uint64_t>(nr.size[0]) * nr.size[1];

        baseRoi.size[1] = nr.size[1] / nEngines;
        reminder = nr.size[1] % nEngines;

        while (workDistributed < totalSize && workingEngines < nEngines)
        {
            NodeROI newRoi(baseRoi);
            newRoi.baseOffset[1] += baseOffset;

            if (reminder)
            {
                newRoi.size[1]++;
                reminder--;
            }

            // update the base offset of both input and output tensor ROIs
            if (newRoi.inputRois.size() > 0)
            {
                TensorROILayout& layout = newRoi.inputRois[0].getLayout();
                layout.baseAddress += inOffsetAccum;
                layout.m_size[0] = newRoi.size[0];
                layout.m_size[1] = newRoi.size[1];
                layout.m_baseOffset[0] = newRoi.baseOffset[0];
                layout.m_baseOffset[1] = newRoi.baseOffset[1];
                inOffsetAccum += layout.spatialStrides[0] * newRoi.size[1];
            }
            if (newRoi.outputRois.size() > 0)
            {
                TensorROILayout& layout = newRoi.outputRois[0].getLayout();
                layout.baseAddress += outOffsetAccum;
                layout.m_size[0] = newRoi.size[0];
                layout.m_size[1] = newRoi.size[1];
                layout.m_baseOffset[0] = newRoi.baseOffset[0];
                layout.m_baseOffset[1] = newRoi.baseOffset[1];
                outOffsetAccum += layout.spatialStrides[0] * newRoi.size[1];
            }

            workDistributed += newRoi.size[0] * newRoi.size[1];
            baseOffset += newRoi.size[1];
            ret.push_back(newRoi);
            workingEngines++;
        }

        HB_ASSERT(workDistributed == totalSize, "don't have enough engines for this logical roi");
        HB_ASSERT(reminder == 0, "didn't expect to have a reminder");
    }
    return ret;
}

static std::list<NodeROI> dmaTransposeToPhysicalRois(DMANode&                 node,
                                                     std::list<NodeROI>&      roisToSplit,
                                                     unsigned                 nEngines)
{
    auto& dmaTransposeNode = static_cast<DMATransposeNode&>(node);
    return dmaTransposeNode.getSplitStrategy()->splitToPhysical(node, roisToSplit, nEngines);
}

static std::list<NodeROI> splitLinearDmaRoiToPhysicalRois(std::list<NodeROI>& roisToSplit,
                                                          uint64_t            maxNumEngines,
                                                          uint64_t            elementSizeInBits,
                                                          unsigned            cacheLineSizeInBytes,
                                                          unsigned            chunkSize,
                                                          bool                isBroadcast)
{
    TOffset             offsetAccum    = 0;
    TOffset             offsetElements = 0;
    std::list<NodeROI>  ret;
    for (NodeROI &nr : roisToSplit)
    {
        NodeROI   baseRoi(nr);
        unsigned  reminder        = 0;
        unsigned  workDistributed = 0;
        unsigned  workingEngines  = 0;
        uint64_t  totalSize       = nr.size[0];

        // set minimum size for ROI - give each engine at least this amount of work.
        unsigned nEngines = std::min(maxNumEngines, safeBitsToByte(nr.size[0] * elementSizeInBits) / chunkSize);
        nEngines = (nEngines == 0)? 1 : nEngines;

        // calc transfer size per engine and the reminder in bytes
        baseRoi.size[0] = safeBitsToByte(nr.size[0] * elementSizeInBits) / nEngines;
        reminder        = safeBitsToByte(nr.size[0] * elementSizeInBits) % nEngines;

        // for efficiency, deal with cache line size transfers only, the last engine will handle the reminders
        reminder        += (baseRoi.size[0] % cacheLineSizeInBytes) * nEngines;
        baseRoi.size[0] -= baseRoi.size[0] % cacheLineSizeInBytes;

        // switch back to elements
        reminder        = BITS_PER_BYTE * reminder / elementSizeInBits;
        baseRoi.size[0] = BITS_PER_BYTE * baseRoi.size[0] / elementSizeInBits;

        workingEngines = baseRoi.size[0] == 0 ? 1 : nEngines; // the reminder covers it all and it goes to one engine

        for (unsigned i = 0; i < workingEngines; ++i)
        {
            NodeROI newRoi(baseRoi);
            newRoi.baseOffset[0] += offsetElements;

            if (i == workingEngines - 1) // last iteration handles the reminder
            {
                newRoi.size[0] += reminder;
            }

            // update the base offset of both input and output tensor ROIs
            if (newRoi.inputRois.size() > 0)
            {
                TensorROILayout& layout = newRoi.inputRois[0].getLayout();
                // in case of broadcast each ROI read from the begining of the tensor
                if (isBroadcast)
                {
                    layout.baseAddress += safeBitsToByte(workDistributed * elementSizeInBits);
                    layout.m_baseOffset[0] = workDistributed;
                }
                else
                {
                    layout.baseAddress += offsetAccum;
                    layout.m_baseOffset[0] += offsetElements;
                }
                layout.m_size[0] = newRoi.size[0];
            }
            if (newRoi.outputRois.size() > 0)
            {
                TensorROILayout& layout = newRoi.outputRois[0].getLayout();
                layout.baseAddress += offsetAccum;
                layout.m_baseOffset[0] += offsetElements;
                layout.m_size[0] = newRoi.size[0];
            }

            offsetAccum      += safeBitsToByte(newRoi.size[0] * elementSizeInBits);
            offsetElements   += newRoi.size[0];
            workDistributed  += newRoi.size[0];

            ret.push_back(newRoi);
        }

        HB_ASSERT(workDistributed == totalSize, "don't have enough engines for this logical roi");
    }
    return ret;
}

static std::list<NodeROI> splitRotateRoiToPhysicalRois(std::list<NodeROI>& roisToSplit,
                                                       pNode&              node,
                                                       unsigned            numRotators,
                                                       unsigned            rotateStripeWidth,
                                                       unsigned            rotateStripeHeightStraightAngle)
{
    std::list<NodeROI>      physicalRois;
    RotateNode*             rotateNode      = reinterpret_cast<RotateNode*>(node.get());
    float                   rotationAngle   = rotateNode->getRotationAngle();      // In case of +/-90 we need to split the physical rois
    pTensor                 outputTensor    = rotateNode->getOutput(0);
    const SizeArray         output_dims     = outputTensor->getAllSizesInElements();
    uint32_t                outputHeight    = output_dims[NCHW_H_DIM];
    uint32_t                outputWidth     = output_dims[NCHW_W_DIM];
    unsigned                maxStripeHeight = ((rotationAngle == 90) || (rotationAngle==270)) ? rotateStripeHeightStraightAngle : outputHeight;
    unsigned                pipelineLevel   = 0;

    // Run over the logical rois and split each of them to rois of size {1,1,H,128 (or less)}
    // In case angle is 90 or 270, we also split vertically according to maxStripeHeight value
    for (NodeROI &roi : roisToSplit)
    {
        unsigned roiBatchSize = roi.size[NCHW_N_DIM];
        unsigned roiChannelSize = roi.size[NCHW_C_DIM];
        unsigned roiIdx = 0;

        for (int b = 0; b < roiBatchSize; b++)
        {
            for (int c = 0; c < roiChannelSize; c++)
            {
                // rotateStripeWidth is calculated for "unsupported" angles in gaudi2. For greco it will stay 128
                for (uint32_t x = 0; x < outputWidth; x += rotateStripeWidth)
                {
                    for (uint32_t y = 0; y < outputHeight; y += maxStripeHeight)
                    {
                        NodeROI newRoi(roi);
                        newRoi.pipelineLevel = pipelineLevel;
                        // Set the engine index
                        newRoi.engineIndex = (roiIdx++) % numRotators;
                        // Set the roi dimensions
                        unsigned stripeWidth = ((x + rotateStripeWidth) >= outputWidth) ? outputWidth - x : rotateStripeWidth;
                        unsigned stripeHeight = ((y + maxStripeHeight) >= outputHeight) ? outputHeight - y : maxStripeHeight;
                        newRoi.size[NCHW_N_DIM] = newRoi.size[NCHW_C_DIM] = 1;    // batch and channel are of size 1
                        newRoi.size[NCHW_H_DIM] = stripeHeight;
                        newRoi.size[NCHW_W_DIM] = stripeWidth;
                        // Set the roi location within the tensor: {roiBatch + b, c, 0, x}
                        newRoi.baseOffset[NCHW_N_DIM] = roi.baseOffset[NCHW_N_DIM] + b;
                        newRoi.baseOffset[NCHW_C_DIM] = c;
                        newRoi.baseOffset[NCHW_H_DIM] = y;
                        newRoi.baseOffset[NCHW_W_DIM] = x;

                        // By default,an roi does not signal. Only the last ones will later set to signal
                        newRoi.numSignals = 0;

                        physicalRois.push_back(newRoi);
                    }
                }
            }
        }
        pipelineLevel++;

        // The last ROI that is sent to each of the rotate engines should signal Sync
        // So we set the last rois according to numRotators value
        size_t count = numRotators;
        for (auto itr = physicalRois.rbegin(); (count != 0U) && itr != physicalRois.rend(); itr++, count--)
        {
            itr->numSignals = 1;
        }
    }
    return physicalRois;
}

class PhysicalRoiSplitter : public synapse::ThreadWorkItem
{
public:
    PhysicalRoiSplitter(const HabanaGraph&  graph,
                        const pNode&        node,
                        std::list<NodeROI>* rois,
                        std::list<NodeROI>& phisicalRois)
    : m_graph(graph), m_node(node), m_rois(rois), m_physicalRois(phisicalRois)
    {
    }

    static void assignPhysicalEngineToRois(std::list<NodeROI>& physicalRois, unsigned nEngines, bool multipleDescsPerEngine=false)
    {
        // Set the engineIndex of the roi according to pipeline level and up to the number of engines
        unsigned int assignedEngine = 0;
        unsigned int prevPipelineLevel = physicalRois.begin()->pipelineLevel;

        for (NodeROI& roi : physicalRois)
        {
            // We need to reset the roiCounter whenever we move to the next pipeline level
            if (roi.pipelineLevel != prevPipelineLevel)
            {
                assignedEngine = 0;
            }
            // Verify that roiCounter does not exceed the number of physical engines
            HB_ASSERT(assignedEngine < nEngines || multipleDescsPerEngine, "physical roi exceeds num of engines");

            roi.engineIndex = assignedEngine % nEngines;

            assignedEngine++;
            prevPipelineLevel = roi.pipelineLevel;
        }
    }

    virtual void doWork() override
    {
        if (m_rois == nullptr)
        {

            HB_ASSERT(false, "{}: no rois defined for node", __func__);
            return;
        }
        std::copy(m_rois->begin(), m_rois->end(), std::back_inserter(m_physicalRois));

        // Already done on calculate linear ranges.
        if (HabanaGraph::runsOnMME(m_node))
        {
            return;
        }

        if (HabanaGraph::runsOnTPC(m_node))
        {
            std::shared_ptr<TPCNode> tpcNode = std::dynamic_pointer_cast<TPCNode>(m_node);
            HB_ASSERT(tpcNode != nullptr, "In SplitToPhysicalROIs, tpcNode is Null");

            LOG_DEBUG(GC, "Splitting node {} to physical ROIs", m_node->getNodeName());

            if (tpcNode->isLoweringKernel())
            {
                m_physicalRois = m_splitter.splitSpecialKernelsBetweenEngines(m_physicalRois, m_graph.getNumTpcEng());
            }
            else
            {
                m_physicalRois = m_splitter.splitRoisBetweenEngines(m_physicalRois,
                                                                    tpcNode->getNodeAnnotation().tpcSplitDims,
                                                                    m_graph.getNumTpcEng(),
                                                                    tpcNode->getNodeName());
            }
            m_physicalRois = m_splitter.projectTPCRois(m_node, m_physicalRois);
            unsigned nEngines = m_graph.getNumTpcEng();
            assignPhysicalEngineToRois(m_physicalRois, nEngines);
        }
        else if (m_node->isDma())
        {
            pTensor tensor = (m_node->getNumInputsDataTensors() > 0) ? m_node->getInput(0) : m_node->getOutput(0);
            DMANode* dmaNode = static_cast<DMANode*>(m_node.get());
            unsigned                  nEngines = dmaNode->parallelLevel();
            const uint64_t            maxRegVal = m_graph.getHALReader()->getMaxRegValForDma();
            bool                      hugeDma   = isHugeTensorROI(m_graph, m_node, m_physicalRois);
            // Linear semantic memset that are done using DMA copy should be treated as Linear DMA
            // The node itself is not linear because it is strided copy from address 0 (W/A for HW bug)
            if ((dmaNode->isLinearDma() || (dmaNode->isMemset() && dmaNode->getOpType() == DMA_OP_TYPE::DMA_OP_COPY)) &&
                !hugeDma)
            {
                m_physicalRois = splitLinearDmaRoiToPhysicalRois(m_physicalRois,
                                                                 nEngines,
                                                                 tensor->getElementSizeInBits(),
                                                                 m_graph.getHALReader()->getCacheLineSizeInBytes(),
                                                                 static_cast<unsigned>(dmaNode->chunkSizeInBytes()),
                                                                 dmaNode->isBroadcast());
            }
            else if (dmaNode->isTranspose())
            {
                m_physicalRois = dmaTransposeToPhysicalRois(*dmaNode, m_physicalRois, nEngines);
            }
            else if (!hugeDma && dmaNode->isNodeTwoDimensionalStrided())
            {
                m_physicalRois = split2dStridedDmaRoiToPhysicalRois(m_physicalRois, nEngines);
            }
            else
            {
                if (hugeDma)
                {
                    m_physicalRois = splitHugeDma(m_splitter,
                                                  m_physicalRois,
                                                  dmaNode->getSplitDimensionsOrder(),
                                                  m_graph.getHALReader()->getMaxRegValForDma(),
                                                  nEngines,
                                                  dmaNode->getNodeName());
                }
                else
                {
                    m_physicalRois = m_splitter.splitRoisBetweenEngines(m_physicalRois,
                                                                        dmaNode->getSplitDimensionsOrder(),
                                                                        nEngines,
                                                                        dmaNode->getNodeName(),
                                                                        false /* don't prefer split on FCD*/);
                }
                m_physicalRois = m_splitter.projectDmaRoisToFullSizes(dmaNode, m_physicalRois);
            }

            assignPhysicalEngineToRois(m_physicalRois, nEngines, true);
        }
        else if (m_node->isRotate())
        {
            std::shared_ptr<RotateNode> rotateNode = std::dynamic_pointer_cast<RotateNode>(m_node);

            m_physicalRois = splitRotateRoiToPhysicalRois(m_physicalRois,
                                                          m_node,
                                                          m_graph.getHALReader()->getNumRotatorEngines(),
                                                          m_graph.getRotateStripeWidth(rotateNode),
                                                          m_graph.getHALReader()->getRotateStripeHeightStraightAngle());
        }
        m_node->setPhysicalRois(m_physicalRois);
    }

private:
    ROISplitter          m_splitter;
    const HabanaGraph&   m_graph;
    pNode                m_node;
    std::list<NodeROI>*  m_rois;
    std::list<NodeROI>&  m_physicalRois;
};

bool splitToPhysicalROIs(HabanaGraph& g)
{
    synapse::ThreadPool threadPool;

    threadPool.start();

    const auto& sortedNodes = g.getExeSortedNodes();

    for (const pNode& n : sortedNodes)
    {
        if (GCFG_ARC_ARCHITECTURE.value() && HabanaGraph::runsOnTPC(n))
        {
            // In DSD we operate on physical ROIs throughout.
            // When a WD context is in play (TPC), logical ROIs take the role of physical ROIs.
            // Previously physical ROIs remained empty, which breaks DSD.
            // Here we set physical ROIs to be the same as logical ROIs, which restores DSD function
            auto* logicalROIs = g.GetNodeROIs(n);
            n->setPhysicalRois(*logicalROIs);
            continue;
        }
        threadPool.addJob(new PhysicalRoiSplitter(g, n, g.GetNodeROIs(n), g.getCodeGenerator()->getPhysicalRois(n)));
    }
    threadPool.finish();

    return true;
}

void splitToPhysicalROIsForNode(const HabanaGraph&  graph,
                                const pNode&        node,
                                std::list<NodeROI>* rois,
                                std::list<NodeROI>& phisicalRois)
{
    PhysicalRoiSplitter(graph, node, rois, phisicalRois).doWork();
}
