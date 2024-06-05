#include "habana_nodes.h"
#include "node.h"
#include "mme_dim_controller.h"
#include "dedw_node.h"
#include "dedx_node.h"
#include "tensor.h"

MmeDimController::MmeDimController(const NodePtr& mmeNode)
{
    synConvolution3DParamsV2* pConvParams = nullptr;
    switch (mmeNode->getNodeType())
    {
        case Node::TYPE_CONVOLUTION:
            m_batchDim.push_back(mmeNode->getInput(TENSOR_IFM)->getDim() - 1);
            m_commonDimOperandA.push_back(DIM_C);
            m_nonCommonDimOperandB.push_back(WEIGHT_DIM_K);
            m_commonDimOperandB.push_back(WEIGHT_DIM_C);
            m_widthOutput.push_back(DIM_C);

            for (uint32_t dim = 0; dim <= mmeNode->getOutput(0)->getDim() - 2; ++dim)
            {
                m_nonCommonDimOperandA.push_back(DIM_W + dim);
                m_heightOutput.push_back(DIM_W + dim);
            }

            pConvParams = &std::static_pointer_cast<ConvolutionNode>(mmeNode)->getConvolutionParams();
            break;
        case Node::TYPE_DEDW:
            m_batchDim.push_back(mmeNode->getInput(TENSOR_IFM)->getDim() - 1);
            m_nonCommonDimOperandA.push_back(DIM_C);
            m_nonCommonDimOperandB.push_back(DIM_C);
            m_heightOutput.push_back(WEIGHT_DIM_K);
            m_widthOutput.push_back(WEIGHT_DIM_C);

            HB_ASSERT(mmeNode->getInput(0)->getDim() == mmeNode->getInput(1)->getDim(), "dimension mismatch");
            for (uint32_t dim = 0; dim <= mmeNode->getInput(0)->getDim() - 2; ++dim)
            {
                m_commonDimOperandB.push_back(DIM_W + dim);
                m_commonDimOperandA.push_back(DIM_W + dim);
            }

            pConvParams = &std::static_pointer_cast<DeToDwNode>(mmeNode)->getConvolutionParams();
            break;
        case Node::TYPE_DEDX:
        case Node::TYPE_TRANSPOSED_DEDX:
        {
            m_batchDim.push_back(mmeNode->getInput(TENSOR_IFM)->getDim() - 1);
            m_commonDimOperandA.push_back(DIM_C);
            m_widthOutput.push_back(DIM_C);
            // operand B layout is changed in TYPE_TRANSPOSED_DEDX from [K,C,S,R] to [C,K,S,R] but the role of the dim
            // is the same
            uint8_t weightDimK = (mmeNode->getNodeType() == Node::TYPE_DEDX) ? WEIGHT_DIM_K : WEIGHT_DIM_C;
            uint8_t weightDimC = (mmeNode->getNodeType() == Node::TYPE_DEDX) ? WEIGHT_DIM_C : WEIGHT_DIM_K;
            m_commonDimOperandB.push_back(weightDimK);
            m_nonCommonDimOperandB.push_back(weightDimC);
            for (uint32_t dim = 0; dim <= mmeNode->getOutput(0)->getDim() - 2; ++dim)
            {
                m_nonCommonDimOperandA.push_back(DIM_W + dim);
                m_heightOutput.push_back(DIM_W + dim);
            }

            pConvParams = &std::static_pointer_cast<DeToDxNode>(mmeNode)->getConvolutionParams();
            break;
        }
        case Node::TYPE_BATCH_GEMM:
        case Node::TYPE_BATCH_GEMM_DEDX:
        case Node::TYPE_BATCH_GEMM_DEDW:
        case Node::TYPE_MASKED_BATCH_GEMM:
            initGemmDims(mmeNode);
            for (uint32_t dim = DIM_GEMM_BATCH; dim < mmeNode->getOutput(0)->getDim(); ++dim)
            {
                m_batchDim.push_back(dim);
            }
            break;
        case Node::TYPE_GEMM:
        case Node::TYPE_GEMM_DEDX:
        case Node::TYPE_GEMM_DEDW:
        case Node::TYPE_FC:
            initGemmDims(mmeNode);
            m_batchDim.push_back(DIM_B);
            break;
        default:
            HB_ASSERT(false, "Unsupported mme node for controller");
            break;
    }
    if (pConvParams)
    {
        m_QRS.insert(m_QRS.end(), pConvParams->kernel, pConvParams->kernel + CONV_KERNEL_SIZE);
    }
}

void MmeDimController::initGemmDims(const NodePtr& mmeNode)
{
    auto        gemmNode   = std::static_pointer_cast<GEMMNode>(mmeNode);
    const auto& gemmParams = gemmNode->getGEMMParams();

    m_commonDimOperandA.push_back(gemmParams.transpose_a ? DIM_W : DIM_C);
    m_nonCommonDimOperandA.push_back(gemmParams.transpose_a ? DIM_C : DIM_W);
    m_commonDimOperandB.push_back(gemmParams.transpose_b ? WEIGHT_DIM_K : WEIGHT_DIM_C);
    m_nonCommonDimOperandB.push_back(gemmParams.transpose_b ? WEIGHT_DIM_C : WEIGHT_DIM_K);
    bool isDedw =
        mmeNode->getNodeType() == Node::TYPE_GEMM_DEDW || mmeNode->getNodeType() == Node::TYPE_BATCH_GEMM_DEDW;
    m_widthOutput.push_back(isDedw ? DIM_W : DIM_C);
    m_heightOutput.push_back(isDedw ? DIM_C : DIM_W);

    m_QRS.assign(3, 1);  // GEMM's QRS is always 1-s
}

const DimVector& MmeDimController::getNonCommonAxis(unsigned operandIndex) const
{
    return operandIndex == 0 ? nonCommonDimOperandA() : nonCommonDimOperandB();
}

const DimVector& MmeDimController::commonDimOperandA() const
{
    return m_commonDimOperandA;
}

const DimVector& MmeDimController::nonCommonDimOperandA() const
{
    return m_nonCommonDimOperandA;
}

const DimVector& MmeDimController::commonDimOperandB() const
{
    return m_commonDimOperandB;
}
const DimVector& MmeDimController::nonCommonDimOperandB() const
{
    return m_nonCommonDimOperandB;
}

const DimVector& MmeDimController::nonCommonDimsForOperand(unsigned inputIndex) const
{
    HB_ASSERT((inputIndex == 0 || inputIndex == 1), "Invalid MME node operand index {}", inputIndex);
    return (inputIndex == 0) ? nonCommonDimOperandA() : nonCommonDimOperandB();
}

const DimVector& MmeDimController::commonDimsForOperand(unsigned inputIndex) const
{
    HB_ASSERT((inputIndex == 0 || inputIndex == 1), "Invalid MME node operand index {}", inputIndex);
    return (inputIndex == 0) ? commonDimOperandA() : commonDimOperandB();
}

bool MmeDimController::isCommonDimForOperand(uint32_t dim, unsigned inputIndex) const
{
    HB_ASSERT((inputIndex == 0 || inputIndex == 1), "Invalid MME node operand index {}", inputIndex);
    return (inputIndex == 0) ? isCommonDimOperandA(dim) : isCommonDimOperandB(dim);
}

const DimVector& MmeDimController::widthOutput() const
{
    return m_widthOutput;
}

const DimVector& MmeDimController::heightOutput() const
{
    return m_heightOutput;
}

const DimVector& MmeDimController::qrsSizes() const
{
    return m_QRS;
}

const DimVector& MmeDimController::batchDim() const
{
    return m_batchDim;
}

bool MmeDimController::isCommonDimOperandA(uint32_t dim) const
{
    return (std::find(m_commonDimOperandA.begin(), m_commonDimOperandA.end(), dim) != m_commonDimOperandA.end());
}

bool MmeDimController::isCommonDimOperandB(uint32_t dim) const
{
    return (std::find(m_commonDimOperandB.begin(), m_commonDimOperandB.end(), dim) != m_commonDimOperandB.end());
}
