#include "project_node_rois_to_tensor_rois.h"
#include <algorithm>
#include <memory>
#include "broadcast_node.h"
#include "node_roi.h"
#include "habana_nodes.h"
#include "tpc_node.h"
#include "habana_pass.h"
#include "habana_graph.h"
#include "transpose_utils.h"
#include "dma_transpose_node.h"

// legacy strides have trivial FCD stride, and do not include it.
// until we include FCD strides in ROI we will use the old format.
static void
getNStridesInBytesNoFcd(const TensorPtr& t, TStride strides[Tensor::c_numOfNStrides], unsigned dimToStart = 0)
{
    HB_ASSERT(!t->isStridedOnFCD(), "unexpected stride on FCD in tensor {}", t->getName());
    const TStride* origStrides = t->getNStridesInBytes();
    std::copy(origStrides + dimToStart + 1, origStrides + Tensor::c_numOfNStrides, strides);
    // update the rest of dims with the max stride
    TStride* stridesEnd       = strides + Tensor::c_numOfNStrides;
    TStride* fillStridesStart = stridesEnd - dimToStart - 1;
    TStride  maxStride        = *std::max_element(strides, fillStridesStart);
    std::fill(fillStridesStart, stridesEnd, maxStride);
}

static void SetFullROI(TensorROI& roi, std::shared_ptr<Tensor> t)
{
    if (!t) return;
    roi.m_parentTensor = t;
    TensorROILayout& layout = roi.getLayout();
    layout.tensorDim = t->getDim();

    layout.m_size = t->getNSizesInElements();
    memset(layout.m_baseOffset, 0, sizeof(layout.m_baseOffset));

    getNStridesInBytesNoFcd(t, layout.spatialStrides);
}

static void ProjectMMERoi(NodeROI& roi, std::shared_ptr<Node> n)
{
    TensorROI ifmRoi, wghRoi, biasRoi, cinRoi, ofmRoi;
    pTensor IFM  = n->getInput(TENSOR_IFM);
    pTensor WGH  = n->getInput(TENSOR_WEIGHT);
    pTensor CIN  = nullptr;
    pTensor bias = nullptr;
    pTensor OFM  = n->getOutput(TENSOR_OFM);

    std::shared_ptr<MmeNode> mmeNode = std::dynamic_pointer_cast<MmeNode>(n);

    if (mmeNode == nullptr)
    {
        LOG_ERR(GC, "Invalid node type encountered in project ROIs");
        return;
    }

    roi.inputRois.resize(TENSOR_INPUT_MAX);

    if (mmeNode->hasBias())
    {
        bias = n->getInput(TENSOR_BIAS);
        SetFullROI(biasRoi, bias);
        roi.inputRois[TENSOR_BIAS] = biasRoi;
    }

    if (mmeNode->hasCin())
    {
        CIN  = n->getInput(TENSOR_CIN);
        SetFullROI(cinRoi, CIN);
        roi.inputRois[TENSOR_CIN] = cinRoi;
    }

    SetFullROI(ifmRoi, IFM);
    SetFullROI(wghRoi, WGH);
    SetFullROI(ofmRoi, OFM);

    roi.inputRois[TENSOR_IFM] = ifmRoi;
    roi.inputRois[TENSOR_WEIGHT] = wghRoi;
    roi.outputRois.push_back(ofmRoi);
}

static void ProjectTPCRoi(NodeROI& roi, std::shared_ptr<Node> n)
{
    for (unsigned inputIdx = 0; inputIdx < n->getNumInputs(); ++inputIdx)
    {
        auto input = n->getInput(inputIdx);
        if (input->isShapeTensor()) continue;
        NodeROI inputRoi = n->getInputROI(roi, inputIdx).value();
        TensorROI newRoi;
        SetFullROI(newRoi, input);
        memcpy(newRoi.getLayout().m_size.data(), inputRoi.size, sizeof(inputRoi.size));
        memcpy(newRoi.getLayout().m_baseOffset, inputRoi.baseOffset, sizeof(inputRoi.baseOffset));
        newRoi.m_parentTensor = input;
        roi.inputRois.push_back(newRoi);
    }

    for (unsigned outputIdx = 0; outputIdx < n->getNumOutputs(); ++outputIdx)
    {
        auto output = n->getOutput(outputIdx);
        if (output->isShapeTensor()) continue;
        NodeROI outputRoi = n->getOutputROI(roi, outputIdx).value();
        TensorROI newRoi;
        SetFullROI(newRoi, output);
        memcpy(newRoi.getLayout().m_size.data(), outputRoi.size, sizeof(outputRoi.size));
        memcpy(newRoi.getLayout().m_baseOffset, outputRoi.baseOffset, sizeof(outputRoi.baseOffset));
        newRoi.m_parentTensor = output;
        roi.outputRois.push_back(newRoi);
    }
}

static void SetFullGenericROI(NodeROI&                         roi,
                              TensorROI&                       tRoi,
                              const TensorPtr&                 t,
                              bool                             updateRoiSize,
                              const TransposePermutationArray& permutation = {})
{
    TensorROILayout&  layout  = tRoi.getLayout();
    unsigned          dims    = t->getDim();
    TSize             roiSize =
        multiplyElements(std::begin(roi.size), std::begin(roi.size) + std::max<unsigned>(SYN_MAX_TENSOR_DIM, dims));
    if (roiSize == 0)
    {
        HB_ASSERT(false,
                  "NodeROI contains a 0 at {}",
                  std::find(std::begin(roi.size), std::end(roi.size), 0) - std::begin(roi.size));
    }
    tRoi.m_parentTensor = t;
    if (!permutation.empty())
    {
        applyPermutation(roi.size, permutation, layout.m_size.data());
    }
    else
    {
        memcpy(layout.m_size.data(), roi.size, sizeof(layout.m_size));
    }
    if (!permutation.empty())
    {
        applyPermutation(roi.baseOffset, permutation, layout.m_baseOffset);
    }
    else
    {
        memcpy(layout.m_baseOffset, roi.baseOffset, sizeof(layout.m_baseOffset));
    }
    layout.tensorDim = dims;
    // in case that the first dense dims are aggregated we need to skip the first strides
    unsigned* dimToStart = static_cast<unsigned*>(roi.additionalData.get());
    if (dimToStart != nullptr)
    {
        getNStridesInBytesNoFcd(t, layout.spatialStrides, *dimToStart);
    }
    else
    {
        getNStridesInBytesNoFcd(t, layout.spatialStrides);
    }
    if (updateRoiSize)
    {
        t->updateTensorROISize(safeBitsToByte(roiSize * t->getElementSizeInBits()));
    }
    auto sizeIterStart = std::begin(layout.m_size);
    auto sizeIterEnd   = sizeIterStart + layout.tensorDim;
    HB_ASSERT(std::find(sizeIterStart, sizeIterEnd, 0) == sizeIterEnd, "Can't have zero size layout");
}

void projectDmaRoi(NodeROI& roi, const Node& n)
{
    const auto& dmaNode    = static_cast<const DMANode&>(n);
    bool isPrefetch = dmaNode.isPrefetch();

    if (n.getNumInputsDataTensors() > 0 || isPrefetch)
    {
        TensorROI newRoi;
        if (dmaNode.isBroadcast())
        {
            NodeROI roiForBroadcast(roi);
            roiForBroadcast.baseOffset[0] = 0;

            // in case of broadcast if ROI size is not equal to input, it's needed to read from output
            const TensorPtr& t =
                (roi.size[0] != n.getInput(0)->getDenseSizeInElements() ? n.getOutput(0) : n.getInput(0));
            SetFullGenericROI(roiForBroadcast, newRoi, t, false);
        }
        else
        {
            // dma prefetch node has only one tensor (output) with both dram and sram addresses
            const TensorPtr& t = (isPrefetch ? n.getOutput(0) : n.getInput(0));
            SetFullGenericROI(roi, newRoi, t, dmaNode.getDmaType() == DMA_TYPE::DMA_TYPE_UPSTREAM);
        }
        roi.inputRois.push_back(std::move(newRoi));
    }
    if (n.getNumOutputs() > 0)
    {
        TensorROI newRoi;
        const TensorPtr& t = n.getOutput(0);
        if (dmaNode.isTranspose())
        {
            SetFullGenericROI(roi,
                              newRoi,
                              t,
                              dmaNode.getDmaType() == DMA_TYPE::DMA_TYPE_DOWNSTREAM,
                              static_cast<const DMATransposeNode&>(dmaNode).permutation());
        }
        else
        {
            SetFullGenericROI(roi, newRoi, t, dmaNode.getDmaType() == DMA_TYPE::DMA_TYPE_DOWNSTREAM);
        }
        roi.outputRois.push_back(std::move(newRoi));
    }
}

static void ProjectRotateRoi(NodeROI& roi, pNode n)
{
    // The rotator does not split the tensors. Therefore, we project each logical roi onto the
    // full tensors of both input and output
    std::shared_ptr<RotateNode> rotateNode = std::dynamic_pointer_cast<RotateNode>(n);
    pTensor inputTensor = n->getInput(0);
    pTensor outputTensor = n->getOutput(0);

    TensorROI inputRoi;
    SetFullROI(inputRoi, inputTensor);
    roi.inputRois.push_back(inputRoi);

    TensorROI outputRoi;
    SetFullROI(outputRoi, outputTensor);
    roi.outputRois.push_back(outputRoi);
}

static bool ProjectROI(NodeROI& roi, std::shared_ptr<Node> n)
{
    if (HabanaGraph::runsOnMME(n))
    {
        ProjectMMERoi(roi, n);
        return true;
    }
    if (HabanaGraph::runsOnTPC(n))
    {
        ProjectTPCRoi(roi, n);
        return true;
    }
    if (n->isDma())
    {
        projectDmaRoi(roi, *n);
        return true;
    }
    if (n->isRotate())
    {
        ProjectRotateRoi(roi, n);
        return true;
    }

    LOG_ERR(GC, "Unknown node type to project ROIs for");
    return false;
}

bool projectNodeROIs(HabanaGraph& g)
{
    for (pNode n : g.getExeSortedNodes())
    {
        if (n->isLogicalOperation()) continue;
        for (NodeROI& roi : *g.GetNodeROIs(n))
        {
            bool ret = ProjectROI(roi, n);
            if (!ret)
            {
                LOG_ERR(GC, "Could not project ROIs for {}", n->getNodeName());
                return false;
            }
        }
    }

    return true;
};
