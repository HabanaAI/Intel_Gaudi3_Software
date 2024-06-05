#include "habana_nodes.h"

#include "access_pattern.h"
#include "access_pattern_generator.h"
#include "aggregate_fcd_node.h"
#include "compilation_hal_reader.h"
#include "data_type_utils.h"
#include "defs.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "hal_reader/hal_reader.h"
#include "infra/cpu_calculator.h"
#include "log_manager.h"
#include "logical_op_node.h"
#include "memcopy_node.h"
#include "node.h"
#include "node_factory.h"
#include "node_io_manager.h"
#include "physical_reshape_node.h"
#include "quantization_data.h"
#include "sif/shape_inference_metadata.h"
#include "slicing_utils.h"
#include "synapse_api.h"
#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "synapse_common_types.hpp"
#include "synapse_types_operators.h"
#include "tensor_shape.h"
#include "tpc_node.h"
#include "transpose_node.h"
#include "types.h"
#include "types_exception.h"
#include "utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>
#include <string_view>

using namespace gc;
using ReshapeOutputToInputMapping = SlicedOperandUtils::ReshapeOutputToInputMapping;

template<typename InputData,
         typename OutputData,
         typename StorageFormat     = int64_t,
         typename IntermediateClamp = int32_t>
static void DoGEMM_typed(TensorPtr A, TensorPtr B, TensorPtr bias, TensorPtr Cin, TensorPtr C)
{
    InputData* pA            = reinterpret_cast<InputData*>(A->map());
    InputData* pB            = reinterpret_cast<InputData*>(B->map());
    OutputData* pbias        = (bias == nullptr) ? nullptr : reinterpret_cast<OutputData*>(bias->map());
    OutputData* pCin         = (Cin == nullptr)  ? nullptr : reinterpret_cast<OutputData*>(Cin->map());
    OutputData* pC           = reinterpret_cast<OutputData*>(C->map());

    unsigned aW = A->getSizeInElements(0);
    unsigned aH = A->getSizeInElements(1);
    unsigned batch1 = A->getSizeInElements(2);
    unsigned batch2 = A->getSizeInElements(3);
    unsigned batch3 = A->getSizeInElements(4);
    unsigned bW = B->getSizeInElements(0);
    unsigned bH = B->getSizeInElements(1);
    unsigned cW = C->getSizeInElements(0);
    unsigned cH = C->getSizeInElements(1);
    UNUSED(cH);
    UNUSED(cW);

    HB_ASSERT(batch1 == B->getSizeInElements(2) && batch1 == C->getSizeInElements(2), "size mismatch");
    HB_ASSERT(batch2 == B->getSizeInElements(3) && batch2 == C->getSizeInElements(3), "size mismatch");
    HB_ASSERT(batch3 == B->getSizeInElements(4) && batch3 == C->getSizeInElements(4), "size mismatch");
    if (bias != nullptr)
    {
        HB_ASSERT(bias->getDenseSizeInElements() == C->getSizeInElements(0),
                  "Bias must have same size as the FCD of the GEMM output");
    }

    //Todo: why are zero points doubles?
    double zpA    = A ?  A->getZeroPoint() : 0;
    double zpB    = B ?  B->getZeroPoint() : 0;
    double zpC    = C ?  C->getZeroPoint() : 0;
    double scaleA = A ?  A->getScale()     : 1.;
    double scaleB = B ?  B->getScale()     : 1.;
    double scaleC = C ?  C->getScale()     : 1.;

    HB_ASSERT(aW == bH, "size mismatch");
    HB_ASSERT(aH == cH, "size mismatch");
    HB_ASSERT(bW == cW, "size mismatch");

    DoBatchGEMM_typed<InputData, OutputData, StorageFormat, IntermediateClamp>(pA, aW, aH, pB, bW, bH, pC, batch1*batch2*batch3, zpA, zpB, zpC, scaleA, scaleB, scaleC, pbias, pCin);
}

static void tranposeGemmInputOnCpu(TensorPtr& tensor)
{
    TensorPtr tensor_t = tensor->clone();
    TSize sizes[Tensor::c_tensorMaxDim];
    tensor_t->getAllSizesInElements(sizes, Tensor::c_tensorMaxDim);
    std::swap(sizes[DIM_C], sizes[DIM_W]);
    tensor_t->reshape(tensor_t->getDim(), sizes, nullptr);
    TransposePermutationArray perm{TPD_Width, TPD_Channel, TPD_Height, TPD_Depth, TPD_Batch};
    TransposeNode::transposeOnCpu(tensor, tensor_t, perm);
    tensor = tensor_t;
}

void GEMMNode::DoGEMM_polymorphic(TensorPtr A, TensorPtr B, TensorPtr bias, TensorPtr Cin, TensorPtr C)
{
    if (m_params.transpose_a)
    {
        tranposeGemmInputOnCpu(A);
    }
    if (m_params.transpose_b)
    {
        tranposeGemmInputOnCpu(B);
    }
    switch(A->getElementType())
    {
        case syn_type_bf16:
        {
            switch(C->getElementType())
            {
            case syn_type_float:
            {
                DoGEMM_typed<bfloat16, float, float, float>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_na:
            default:
            {
                HB_ASSERT(false, "Invalid data type");
            }
            }
            break;
        }
        case syn_type_fixed:
        {
            switch(C->getElementType())
            {
                case syn_type_fixed:
                {
                    DoGEMM_typed<int8_t, int8_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint8:
                {
                    DoGEMM_typed<int8_t, uint8_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_int16:
                {
                    DoGEMM_typed<int8_t, int16_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint16:
                {
                    DoGEMM_typed<int8_t, uint16_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_int32:
                {
                    DoGEMM_typed<int8_t, int32_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint32:
                {
                    DoGEMM_typed<int8_t, uint32_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_single:
                case syn_type_na:
                default:
                {
                    HB_ASSERT(false, "Invalid data type");
                }
            }
            break;
        }
        case syn_type_int16:
        {
            switch(C->getElementType())
            {
                case syn_type_fixed:
                {
                    DoGEMM_typed<int16_t, int8_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint8:
                {
                    DoGEMM_typed<int16_t, uint8_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_int16:
                {
                    DoGEMM_typed<int16_t, int16_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint16:
                {
                    DoGEMM_typed<int16_t, uint16_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_int32:
                {
                    DoGEMM_typed<int16_t, int32_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint32:
                {
                    DoGEMM_typed<int16_t, uint32_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_single:
                case syn_type_na:
                default:
                {
                    HB_ASSERT(false, "Invalid data type");
                }
            }
            break;
        }
        case syn_type_uint16:
        {
            switch(C->getElementType())
            {
            case syn_type_fixed:
            {
                DoGEMM_typed<uint16_t, int8_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_uint8:
            {
                DoGEMM_typed<uint16_t, uint8_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_uint16:
            {
                DoGEMM_typed<uint16_t, uint16_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_uint32:
            {
                DoGEMM_typed<uint16_t, uint32_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_int16:
            {
                DoGEMM_typed<uint16_t, int16_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_int32:
            {
                DoGEMM_typed<uint16_t, int32_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_single:
            case syn_type_na:
            default:
            {
                HB_ASSERT(false, "Invalid data type");
            }
            }
            break;
        }
        case syn_type_int32:
        {
            switch(C->getElementType())
            {
                case syn_type_fixed:
                {
                    DoGEMM_typed<int32_t, int8_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint8:
                {
                    DoGEMM_typed<int32_t, uint8_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_int16:
                {
                    DoGEMM_typed<int32_t, int16_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint16:
                {
                    DoGEMM_typed<int32_t, uint16_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_int32:
                {
                    DoGEMM_typed<int32_t, int32_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_uint32:
                {
                    DoGEMM_typed<int32_t, uint32_t>(A, B, bias, Cin, C);
                    break;
                }
                case syn_type_single:
                case syn_type_na:
                default:
                {
                    HB_ASSERT(false, "Invalid data type");
                }
            }
            break;
        }
        case syn_type_uint32:
        {
            switch(C->getElementType())
            {
            case syn_type_fixed:
            {
                DoGEMM_typed<uint32_t, int8_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_uint8:
            {
                DoGEMM_typed<uint32_t, uint8_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_uint16:
            {
                DoGEMM_typed<uint32_t, uint16_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_uint32:
            {
                DoGEMM_typed<uint32_t, uint32_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_int16:
            {
                DoGEMM_typed<uint32_t, int16_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_int32:
            {
                DoGEMM_typed<uint32_t, int32_t>(A, B, bias, Cin, C);
                break;
            }
            case syn_type_single:
            case syn_type_na:
            default:
            {
                HB_ASSERT(false, "Invalid data type");
            }
            }
            break;
        }
        case syn_type_single:
        case syn_type_na:
        default:
        {
            HB_ASSERT(false, "Invalid data type");
        }
    }
}

GEMMNode::GEMMNode(const TensorVector& inputs,
                   const TensorVector& outputs,
                   std::string_view    name,
                   Node::eNodeType     type,
                   ShapeFuncID         sifId)
: MmeNode(inputs, outputs, name, type, sifId)
{
}

SifNodeParams GEMMNode::getShapeInferenceFunctionUserParams()
{
    return reinterpret_cast<SifNodeParams>(&m_params);
}

size_t GEMMNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifGemmMetadata);
}

gc::access_pattern::NodeAccessPatternPtr GEMMNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternMmeNodeGenerator::generate(this);
}

GEMMDeToDwNode::GEMMDeToDwNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          params,
                               std::string_view    name,
                               Node::eNodeType     type)
: GEMMNode(inputs, outputs, name, Node::TYPE_GEMM_DEDW, SIF_GEMM_DEDW)
{
    setParams(params, sizeof(synGEMMParams));
}

GEMMDeToDxNode::GEMMDeToDxNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          params,
                               std::string_view    name,
                               Node::eNodeType     type)
: GEMMNode(inputs, outputs, name, Node::TYPE_GEMM_DEDX, SIF_GEMM_DEDX)
{
    setParams(params, sizeof(synGEMMParams));
}

NodePtr GEMMNode::createNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          userParams,
                             std::string_view    guid,
                             std::string_view    name)
{
    GEMMNode* gemm_node = new GEMMNode(inputs, outputs, name);
    gemm_node->setParams(userParams, sizeof(synGEMMParams));
    return NodePtr(gemm_node);
}

void GEMMNode::setParams(UserParams userParams, unsigned userParamsSize)
{
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(synGEMMParams))
        {
            LOG_ERR(HABANA_NODE, "GEMMNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synGEMMParams));
        }
        synGEMMParams params = *(synGEMMParams*)userParams;
        m_originalParams     = params;
        m_params             = params;
    }
    LOG_TRACE(HABANA_NODE,
              "GEMMNode name - {}, Node params - transpose_a={}, transpose_b={}",
              getNodeName(),
              m_params.transpose_a,
              m_params.transpose_b);
}

NodePtr GEMMDeToDwNode::createNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name)
{
    return NodePtr(new GEMMDeToDwNode(inputs, outputs, userParams, name));
}

void GEMMDeToDwNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    synGEMMParams params;
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(synGEMMParams))
        {
            LOG_ERR(HABANA_NODE, "GEMMDeToDwNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synGEMMParams));
        }
        params = *(synGEMMParams*)userParams;
    }
    else
    {
        // since we use the general GEMM node for gemmDeDw implementation, the 2D representation is:
        // tensor A (which is DeDy): height(a.k.a DIM0)=BHW, width(a.k.a DIM1, common dimension)=K
        // tensor B (which is X): height(a.k.a DIM0, common dimension)=BHW, width(a.k.a DIM1)=C
        // and since in GEMM the common dimension must have the same sizes between tensors A and B,
        // we need to transpose tensor A for that purpose, so it will have width=BHW (matching tensor B's height)
        // that's why we transpose tensor A
        params.transpose_a = true;
    }
    m_originalParams = params;
    m_params         = params;
    LOG_TRACE(HABANA_NODE,
              "GEMMDeToDwNode name - {}, Node params - transpose_a={}, transpose_b={}",
              getNodeName(),
              params.transpose_a,
              params.transpose_b);
}

NodePtr GEMMDeToDxNode::createNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name)
{
    return NodePtr(new GEMMDeToDxNode(inputs, outputs, userParams, name));
}

void GEMMDeToDxNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    synGEMMParams params;
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(synGEMMParams))
        {
            LOG_ERR(HABANA_NODE, "GEMMDeToDxNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synGEMMParams));
        }
        params = *(synGEMMParams*)userParams;
    }
    else
    {
        // since we use the general GEMM node for gemmDeDx implementation,
        // and since in DeDx we can look at the tensors as if they're 2D in the following way:
        // tensor A (which is DeDy): height(a.k.a DIM0)=BHW, width(a.k.a DIM1, common dimension)=K
        // tensor B (which is W): height(a.k.a DIM0, common dimension)=RSC, width(a.k.a DIM1)=K
        // and since in GEMM the common dimension must have the same sizes between tensors A and B,
        // we need to transpose tensor B for that purpose, so it will have height=K (matching tensor A's width)
        // that's why we transpose tensor B
        params.transpose_b = true;
    }
    m_originalParams = params;
    m_params         = params;
    LOG_TRACE(HABANA_NODE,
              "GEMMDeToDxNode name - {}, Node params - transpose_a={}, transpose_b={}",
              getNodeName(),
              params.transpose_a,
              params.transpose_b);
}

NodePtr GEMMNode::clone() const
{
    return NodePtr(new GEMMNode(*this));
}

NodePtr GEMMDeToDxNode::clone() const
{
    return NodePtr(new GEMMDeToDxNode(*this));
}

NodePtr GEMMDeToDwNode::clone() const
{
    return NodePtr(new GEMMDeToDwNode(*this));
}

Settable<NodeROI> GEMMNode::getInputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    Settable<NodeROI> ret;
    NodeROI inputRoi(roi);
    if (tensorIdx == TENSOR_IFM)
    {
        inputRoi.size[DIM_C] = getInput(tensorIdx)->getSizeInElements(DIM_C);
        ret.set(inputRoi);
    }
    else if (tensorIdx == TENSOR_WEIGHT)
    {
        inputRoi.size[DIM_W] = getInput(tensorIdx)->getSizeInElements(DIM_W);
        ret.set(inputRoi);
    }
    else
    {
        LOG_WARN(HABANA_NODE, "Can't produce model parameter roi for node {}, tensor index {}", getNodeName(), tensorIdx);
    }
    return ret;
}

bool GEMMNode::RunOnCpu()
{
    TensorPtr A    = getInput(TENSOR_IFM);
    TensorPtr B    = getInput(TENSOR_WEIGHT);
    TensorPtr bias = hasBias() ? getInput(TENSOR_BIAS) : nullptr;
    TensorPtr Cin  = hasCin() ? getInput(TENSOR_CIN) : nullptr;
    TensorPtr C    = getOutput(TENSOR_OFM);

    HB_ASSERT((A->getDim() <= B->getDim()) && (C->getDim() == A->getDim()),
        "the 2 main inputs and the output of GEMM need to be of the same dimensionality (but A can be lower than B)");
    if (Cin != nullptr)
    {
        HB_ASSERT(Cin->getDim() == C->getDim(), "Cin must have the same dimensionality as the output of GEMM");
        HB_ASSERT(Cin->getElementSizeInBytes() == C->getElementSizeInBytes(),
                  "Cin must be of the same type as the output of GEMM");
    }
    HB_ASSERT(A->getElementSizeInBytes() == B->getElementSizeInBytes(),
              "the inputs of GEMM must agree on the data type");

    DoGEMM_polymorphic(A, B, bias, Cin, C);
    return true;
}

unsigned GEMMNode::getMMEOperandAIndex() const
{
    if (m_type == Node::TYPE_GEMM_DEDW || m_type == Node::TYPE_BATCH_GEMM_DEDW)
    {
        return TENSOR_WEIGHT;
    }
    return TENSOR_IFM;
}

unsigned GEMMNode::getMMEOperandBIndex() const
{
    if (m_type == Node::TYPE_GEMM_DEDW || m_type == Node::TYPE_BATCH_GEMM_DEDW)
    {
        return TENSOR_IFM;
    }
    return TENSOR_WEIGHT;
}

bool GEMMNode::validateNodeLayout() const
{
    uint32_t cdAIdx = m_params.transpose_a ? DIM_W : DIM_C;
    uint32_t hAIdx = m_params.transpose_a ? DIM_C : DIM_W;
    uint32_t cdBIdx = m_params.transpose_b ? DIM_C : DIM_W;
    uint32_t wBIdx = m_params.transpose_b ? DIM_W : DIM_C;

    unsigned opA = getMMEOperandAIndex();
    unsigned opB = getMMEOperandBIndex();

    if (getInput(opA)->getSizeInElements(cdAIdx) != getInput(opB)->getSizeInElements(cdBIdx) ||
        getInput(opA)->getSizeInElements(hAIdx) != getOutput(TENSOR_OFM)->getSizeInElements(DIM_W) ||
        getOutput(TENSOR_OFM)->getSizeInElements(DIM_C) != getInput(opB)->getSizeInElements(wBIdx))
    {
        LOG_ERR(HABANA_NODE, "{} node, mismatch between common dimensions for gemm operation", getNodeName());
        return false;
    }

    return MmeNode::validateNodeLayout();
}

TensorSemanticType GEMMNode::getParamSemanticType(const TensorPtr& param) const
{
    if (param->isModelParameter() && param == getInput(TENSOR_WEIGHT) && param != getInput(TENSOR_IFM))
        return TYPE_WEIGHTS;
    return MmeNode::getParamSemanticType(param);
}

TensorShape GEMMNode::getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const
{
    TensorShape inputShape(output);

    if (inputIdx == TENSOR_IFM)
    {
        CoordArray baseCoord = inputShape.getBases();
        baseCoord[DIM_C] = 0;
        inputShape.setBase(baseCoord);
        SizeArray size = inputShape.getSizes();
        size[DIM_C] = getInput(TENSOR_IFM)->getSizeInElements(DIM_C);
        inputShape.setSize(size);
    }
    else if (inputIdx == TENSOR_WEIGHT)
    {
        CoordArray baseCoord = {0};
        baseCoord[WEIGHT_DIM_K] = inputShape.getBase(WEIGHT_DIM_K);
        inputShape.setBase(baseCoord);
        SizeArray size = {0};
        getInput(TENSOR_WEIGHT)->getAllSizesInElements(size);
        size[WEIGHT_DIM_K] = inputShape.getSize(WEIGHT_DIM_K);
        inputShape.setSize(size);
    }
    else if (inputIdx == TENSOR_CIN)
    {
        inputShape = output; // note: size of CIN can be different from CIN *after* the large images algorithm or packing
    }
    else
    {
        inputShape = MmeNode::getInputShape(output, outputIndex, inputIdx);
    }

    return inputShape;
}

unsigned GEMMNode::getKDimIndex()
{
    if (!getGEMMParams().transpose_b)
    {
        return WEIGHT_DIM_K;
    }

    return WEIGHT_DIM_C;
}

bool GEMMNode::equalTo(const Node& other) const
{
    const GEMMNode* pGemmNode = dynamic_cast<const GEMMNode*>(&other);
    if (pGemmNode == nullptr) return false; //Not a GEMM node
    if (m_params.transpose_a != pGemmNode->m_params.transpose_a ||
        m_params.transpose_b != pGemmNode->m_params.transpose_b)
    {
        return false;
    }
    return Node::equalTo(other);
}

void GEMMNode::print() const
{
    Node::print();
    LOG_DEBUG(GRAPH_DATA, "  Node params: {}, ", getNodeParametersStr());
}

std::string GEMMNode::getNodeParametersStr() const
{
    return fmt::format("transpose_a={}, transpose_b={}", m_params.transpose_a, m_params.transpose_b);
}

void GEMMNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

bool GEMMNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // Check if device supports transpose of both A and B simultaneously
    if (!g.getHALReader()->isMMETransposeAandTransposeBSupported() && m_params.transpose_a && m_params.transpose_b)
    {
        return false;
    }
    return BaseClass::validateNodeForGraph(g);
}

bool GEMMNode::isOperandTransposed(const TensorPtr& t) const
{
    if (t == getInput(0)) return m_params.transpose_a;
    if (t == getInput(1)) return m_params.transpose_b;
    return false;
}

bool GEMMDeToDwNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getTraits().trainingGraph())
    {
        return false;
    }
    return GEMMNode::validateNodeForGraph(g);
}

bool GEMMDeToDxNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getTraits().trainingGraph())
    {
        return false;
    }
    return GEMMNode::validateNodeForGraph(g);
}

BatchGemmNode::BatchGemmNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          params,
                             std::string_view    name,
                             Node::eNodeType     type,
                             ShapeFuncID         sifId)
: GEMMNode(inputs, outputs, name, type, sifId)
{
}

NodePtr BatchGemmNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    BatchGemmNode* batchGemmNode = new BatchGemmNode(inputs, outputs, nullptr, name);
    if (userParams != nullptr)
    {
        batchGemmNode->setParams(userParams, sizeof(synGEMMParams));
        LOG_TRACE(HABANA_NODE,
                  "BatchGEMMNode name - {}, Node params - transpose_a={}, tra81nspose_b={}",
                  name,
                  batchGemmNode->m_params.transpose_a,
                  batchGemmNode->m_params.transpose_b);
    }
    return NodePtr(batchGemmNode);
}

NodePtr BatchGemmNode::clone() const
{
    return NodePtr(new BatchGemmNode(*this));
}

bool BatchGemmNode::validateNodeLayout() const
{
    if (!isFullBroadcastLayout() && !isSymmetricLayout() && !isPartialBroadcastLayout())
    {
        LOG_ERR(HABANA_NODE, "Batch gemm node got tensors mismatch dimensions size");
        return false;
    }
    if (!isValidOutBatchSizes())
    {
        return false;
    }
    return GEMMNode::validateNodeLayout();
}

bool BatchGemmNode::isValidOutBatchSizes() const
{
    const SizeArray inputSizes  = getInput(TENSOR_IFM)->getAllSizesInElements();
    const SizeArray weightSizes = getInput(TENSOR_WEIGHT)->getAllSizesInElements();
    const SizeArray outputSizes = getOutput(TENSOR_OFM)->getAllSizesInElements();
    for (unsigned dim = DIM_GEMM_BATCH; dim < MAX_DIMENSIONS_NUM; dim++)
    {
        if (outputSizes[dim] != std::max(inputSizes[dim], weightSizes[dim]))
        {
            return false;
        }
    }
    return true;
}

bool BatchGemmNode::allBatchDimsDegenerated(const SizeVector& opSizes)
{
    auto iter = std::next(opSizes.begin(), DIM_GEMM_BATCH);
    return std::all_of(iter, opSizes.end(), [&](TSize dimSize) { return dimSize == 1; });
}

/** Full broadcast layout -
 * One of the operands' batch dims must all be equal to 1, and the other operand has dim > 1.
 */
bool BatchGemmNode::isFullBroadcastLayout(const SizeVector& input0Sizes, const SizeVector& input1Sizes)
{
    bool bcastOpA = allBatchDimsDegenerated(input0Sizes);
    bool bcastOpB = allBatchDimsDegenerated(input1Sizes);

    return (bcastOpA ^ bcastOpB);
}

/** Partial broadcast layout -
 * Constraints for the batch dimensions of the input matrices:
 * 1. Their sizes can be equal for some of the batch dims, but not all (otherwise it's symmetric layout).
 * 2. For corresponding dimension sizes, which are not equal, one of them must be equal to 1.
 * 3. Not all batch dimensions of an operand can be 1 (otherwise this is full broadcast).
 */
bool BatchGemmNode::isPartialBroadcastLayout(const SizeVector& input0Sizes, const SizeVector& input1Sizes)
{
    bool isPartial = false;
    auto input0Rank = input0Sizes.size();
    auto input1Rank = input1Sizes.size();

    for (uint32_t dim = DIM_GEMM_BATCH; dim < MAX_DIMENSIONS_NUM; ++dim)
    {
        TSize in1DimSize = (dim < input0Rank) ? input0Sizes[dim] : 1;
        TSize in2DimSize = (dim < input1Rank) ? input1Sizes[dim] : 1;
        if (in1DimSize == in2DimSize) continue;
        if (in1DimSize != 1 && in2DimSize != 1)
        {
            return false;
        }
        isPartial = true;
    }
    return isPartial && !isFullBroadcastLayout(input0Sizes, input1Sizes);
}

/* Symmetric layout -
 * Batch dimensions of the operands must be equal.
 */
bool BatchGemmNode::isSymmetricLayout(const SizeVector& input0Sizes, const SizeVector& input1Sizes)
{
    auto input0Rank = input0Sizes.size();
    auto input1Rank = input1Sizes.size();

    auto max_rank = std::max(input0Rank, input1Rank);

    for (uint32_t dim = DIM_GEMM_BATCH; dim < max_rank; ++dim)
    {
        unsigned in0DimSize = (dim < input0Rank) ? input0Sizes[dim] : 1;
        unsigned in1DimSize = (dim < input1Rank) ? input1Sizes[dim] : 1;
        if (in0DimSize != in1DimSize)
        {
            return false;
        }
    }
    return true;
}

bool BatchGemmNode::isPartialBroadcastLayout() const
{
    const TensorPtr input0 = getInput(0);
    const TensorPtr input1 = getInput(1);

    return isPartialBroadcastLayout(toSizeVector(input0), toSizeVector(input1));
}

bool BatchGemmNode::isFullBroadcastLayout() const
{
    const TensorPtr input0 = getInput(0);
    const TensorPtr input1 = getInput(1);

    return isFullBroadcastLayout(toSizeVector(input0), toSizeVector(input1));
}

bool BatchGemmNode::isBatchGemm() const
{
    return true;
}

bool BatchGemmNode::isSymmetricLayout() const
{
    const TensorPtr input0 = getInput(0);
    const TensorPtr input1 = getInput(1);

    return isSymmetricLayout(toSizeVector(input0), toSizeVector(input1));
}

bool BatchGemmNode::canBeConvertedToGEMM() const
{
    bool canConvert(false), allowDynamic(true);

    // Batch gemm can be converted to gemm only if one operand is full broadcast
    if (!isFullBroadcastLayout())
    {
        return false;
    }

    TensorPtr input0   = getInput(0);
    TensorPtr input1   = getInput(1);

    bool bcastOpB = allBatchDimsDegenerated(toSizeVector(input1));

    // Flatten for dynamic bgemm is possible when the only dynamic dim is the external dim
    auto in0Rank            = getInput(0)->getDim();
    auto in1Rank            = getInput(1)->getDim();
    auto in0FirstDynamicDim = getInput(0)->getFirstDynamicDimIndex();
    auto in1FirstDynamicDim = getInput(1)->getFirstDynamicDimIndex();

    // Check which operand is broadcast
    if (likely(bcastOpB))
    {
        if (in0FirstDynamicDim)
        {
            allowDynamic = (*in0FirstDynamicDim) == (in0Rank - 1);
        }

        canConvert = !m_params.transpose_a && allowDynamic;
    }
    else
    {
        // Enable flatten on operand B when https://jira.habana-labs.com/browse/SW-140671 is solved
        // if (in1FirstDynamicDim)
        // {
        //     allowDynamic = (*in1FirstDynamicDim) == (in1Rank - 1);
        // }
        // canConvert = !m_params.transpose_b && allowDynamic;
        UNUSED(in1FirstDynamicDim);
        UNUSED(in1Rank);
        canConvert = false;
    }

    if (!canConvert)
    {
        LOG_WARN(GC, "Can't flatten MME node {}", getNodeName());
    }
    return canConvert;
}

BatchGemmDeToDwNode::BatchGemmDeToDwNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          params,
                                         std::string_view    name)
: BatchGemmNode(inputs, outputs, params, name, Node::TYPE_BATCH_GEMM_DEDW, SIF_BATCH_GEMM_DEDW)
{
    setParams(params, sizeof(synGEMMParams));
}

NodePtr BatchGemmDeToDwNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new BatchGemmDeToDwNode(inputs, outputs, userParams, name));
}

void BatchGemmDeToDwNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    synGEMMParams params;
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(synGEMMParams))
        {
            LOG_ERR(HABANA_NODE, "BatchGemmDeToDwNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synGEMMParams));
        }
        params = *(synGEMMParams*)userParams;
    }
    else
    {
        params.transpose_a = !params.transpose_a;
    }
    m_originalParams = params;
    m_params         = params;
    LOG_TRACE(HABANA_NODE,
              "BatchGemmDeToDwNode name - {}, Node params - transpose_a={}, transpose_b={}",
              getNodeName(),
              params.transpose_a,
              params.transpose_b);
}

NodePtr BatchGemmDeToDwNode::clone() const
{
    return NodePtr(new BatchGemmDeToDwNode(*this));
}

bool BatchGemmDeToDwNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getTraits().trainingGraph())
    {
        return false;
    }
    return BatchGemmNode::validateNodeForGraph(g);
}

BatchGemmDeToDxNode::BatchGemmDeToDxNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          params,
                                         std::string_view    name)
: BatchGemmNode(inputs, outputs, params, name, Node::TYPE_BATCH_GEMM_DEDX, SIF_BATCH_GEMM_DEDX)
{
    setParams(params, sizeof(synGEMMParams));
}

NodePtr BatchGemmDeToDxNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    return NodePtr(new BatchGemmDeToDxNode(inputs, outputs, userParams, name));
}

void BatchGemmDeToDxNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    synGEMMParams params;
    if (userParams != nullptr)
    {
        if (userParamsSize != sizeof(synGEMMParams) && userParamsSize != 0)
        {
            LOG_ERR(HABANA_NODE, "BatchGemmDeToDxNode userParams size is incorrect");
            throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synGEMMParams));
        }
        params = *(synGEMMParams*)userParams;
    }
    else
    {
        params.transpose_b = !params.transpose_b;
    }
    m_originalParams = params;
    m_params         = params;
    LOG_TRACE(HABANA_NODE,
              "BatchGemmDeToDxNode name - {}, Node params - transpose_a={}, transpose_b={}",
              getNodeName(),
              params.transpose_a,
              params.transpose_b);
}

NodePtr BatchGemmDeToDxNode::clone() const
{
    return NodePtr(new BatchGemmDeToDxNode(*this));
}

bool BatchGemmDeToDxNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getTraits().trainingGraph())
    {
        return false;
    }
    return BatchGemmNode::validateNodeForGraph(g);
}

MaskedBatchGemmNode::MaskedBatchGemmNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         UserParams          params,
                                         std::string_view    name)
: BatchGemmNode(inputs, outputs, params, name, Node::TYPE_MASKED_BATCH_GEMM, SIF_NO_SUPPORT)
{
    setParams(params, sizeof(synGEMMParams));
}

NodePtr MaskedBatchGemmNode::createNode(const TensorVector& inputs,
                                        const TensorVector& outputs,
                                        UserParams          userParams,
                                        std::string_view    guid,
                                        std::string_view    name)
{
    MaskedBatchGemmNode* maskedBatchGemmNode = new MaskedBatchGemmNode(inputs, outputs, userParams, name);
    LOG_TRACE(HABANA_NODE,
              "MaskedBatchGemmNode name - {}, Node params - transpose_a={}, tra81nspose_b={}",
              name,
              maskedBatchGemmNode->m_params.transpose_a,
              maskedBatchGemmNode->m_params.transpose_b);
    return NodePtr(maskedBatchGemmNode);
}

NodePtr MaskedBatchGemmNode::clone() const
{
    return NodePtr(new MaskedBatchGemmNode(*this));
}

bool MaskedBatchGemmNode::validateNode() const
{
    // 4 inputs are expected - 2 gemm operands and 2 masks
    if (getInputs().size() != 4)
    {
        LOG_ERR(HABANA_NODE, "Masked Batch gemm node requires 4 inputs, got {}", getInputs().size());
        return false;
    }
    // Validate the first 4 inputs are not null
    for (unsigned i = 0; i < 4; i++)
    {
        if (!getInput(i))
        {
            LOG_ERR(HABANA_NODE, "Masked Batch gemm node input {} is null", i);
            return false;
        }
    }
    return BatchGemmNode::validateNode();
}

bool MaskedBatchGemmNode::validateNodeForGraph(const HabanaGraph& g) const
{
    if (!g.getTraits().trainingGraph())
    {
        return false;
    }
    // TODO [SW-83012] - for now only gaudi2 and gaudi3 supports this node type
    if ((g.getDeviceType() != synDeviceGaudi2) && (g.getDeviceType() != synDeviceGaudi3))
    {
        LOG_ERR(HABANA_NODE, "Masked Batch gemm is supported only on Gaudi2 and on Gaudi3");
        return false;
    }
    return BatchGemmNode::validateNodeForGraph(g);
}

bool MaskedBatchGemmNode::validateNodeLayout() const
{
    if (isDynamicShape())
    {
        LOG_ERR(HABANA_NODE, "Masked Batch gemm node doesn't support dynamic shape");
        return false;
    }

    if (!isSymmetricLayout())
    {
        LOG_ERR(HABANA_NODE, "Masked Batch gemm node supports only symmetric layout");
        return false;
    }

    TensorPtr in0    = getInput(TENSOR_IFM);
    TensorPtr in1    = getInput(TENSOR_WEIGHT);
    TensorPtr mask0  = getInput(TENSOR_AUX_BGEMM_MASK_A);
    TensorPtr mask1  = getInput(TENSOR_AUX_BGEMM_MASK_B);
    TensorPtr output = getOutput(TENSOR_OFM);
    // Validate the masks rank correspond to the inputs rank
    if (in0->getDim() != mask0->getDim() || in1->getDim() != mask1->getDim())
    {
        LOG_ERR(HABANA_NODE, "{} node, mismatch between ranks of gemm inputs and their masks", getNodeName());
        return false;
    }
    if (in0->getDim() != 4 || in1->getDim() != 4 || output->getDim() != 4)
    {
        LOG_ERR(HABANA_NODE, "{} node, input and output tensors are expected to be 4D", getNodeName());
        return false;
    }

    // Validate the masks and inputs agree on dims.
    // Validation of gemm inputs/output sizes is done in base class
    uint32_t nonCd0Idx     = m_params.transpose_a ? DIM_C : DIM_W;
    uint32_t nonCd1Idx     = m_params.transpose_b ? DIM_W : DIM_C;
    uint32_t nonCdMask0Idx = nonCd0Idx;
    uint32_t nonCdMask1Idx = nonCd1Idx;
    uint32_t cdMask0Idx    = m_params.transpose_a ? DIM_W : DIM_C;
    uint32_t cdMask1Idx    = m_params.transpose_b ? DIM_C : DIM_W;

    uint32_t externalBatchDim = output->getDim() - 1;
    uint32_t internalBatchDim = externalBatchDim - 1;

    // Non common dims of the gemm inputs and the corresponding mask should match
    // Common dims of the 2 masks should match
    // Output external batch dim should match the masks external batch dim (mask per batch)
    if (in0->getSizeInElements(nonCd0Idx) != mask0->getSizeInElements(nonCdMask0Idx) ||
        in1->getSizeInElements(nonCd1Idx) != mask1->getSizeInElements(nonCdMask1Idx) ||
        mask0->getSizeInElements(cdMask0Idx) != mask1->getSizeInElements(cdMask1Idx) ||
        mask0->getSizeInElements(externalBatchDim) != output->getSizeInElements(externalBatchDim) ||
        mask1->getSizeInElements(externalBatchDim) != output->getSizeInElements(externalBatchDim))
    {
        LOG_ERR(HABANA_NODE,
                "{} node, mismatch between equivalent dimensions of gemm inputs and their masks",
                getNodeName());
        return false;
    }
    // Masks internal batch dim should be 1
    if (mask0->getSizeInElements(internalBatchDim) != 1 || mask1->getSizeInElements(internalBatchDim) != 1)
    {
        LOG_ERR(HABANA_NODE, "{} node, masks internal batch dim should be 1", getNodeName());
        return false;
    }

    // Data type of the masks and the gemm inputs should match
    if (in0->getElementType() != mask0->getElementType() || in1->getElementType() != mask1->getElementType())
    {
        LOG_ERR(HABANA_NODE, "{} node, mismatch between gemm inputs and masks data type", getNodeName());
        return false;
    }

    return BatchGemmNode::validateNodeLayout();
}

TensorShape MaskedBatchGemmNode::getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const
{
    TensorShape inputShape(output);
    if (inputIdx == TENSOR_AUX_BGEMM_MASK_A)
    {
        uint32_t  cdMask0Idx = m_params.transpose_a ? DIM_W : DIM_C;
        SizeArray size       = inputShape.getSizes();
        // update the mask common dim size
        size[cdMask0Idx] = getInput(TENSOR_AUX_BGEMM_MASK_A)->getSizeInElements(cdMask0Idx);
        // update the internal batch dim to the input mask dim
        size[DIM_GEMM_BATCH] = getInput(TENSOR_AUX_BGEMM_MASK_A)->getSizeInElements(DIM_GEMM_BATCH);
        inputShape.setSize(size);
    }
    else if (inputIdx == TENSOR_AUX_BGEMM_MASK_B)
    {
        uint32_t  cdMask1Idx = m_params.transpose_b ? DIM_C : DIM_W;
        SizeArray size       = inputShape.getSizes();
        // update the mask common dim size
        size[cdMask1Idx] = getInput(TENSOR_AUX_BGEMM_MASK_B)->getSizeInElements(cdMask1Idx);
        // update the internal batch dim to the input mask dim
        size[DIM_GEMM_BATCH] = getInput(TENSOR_AUX_BGEMM_MASK_B)->getSizeInElements(DIM_GEMM_BATCH);
        inputShape.setSize(size);
    }
    else
    {
        inputShape = GEMMNode::getInputShape(output, outputIndex, inputIdx);
    }

    return inputShape;
}

MaxPoolNode::MaxPoolNode(const TensorVector& inputs,
                         const TensorVector& outputs,
                         const PoolParams&   params,
                         std::string_view    name)
: Node(inputs, outputs, name, Node::TYPE_POOL), m_params(params)
{
}

NodePtr MaxPoolNode::createNode(const TensorVector& inputs,
                                const TensorVector& outputs,
                                UserParams          userParams,
                                const char*         guid,
                                const std::string&  name)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "MaxPoolNode userParams is null");
        throw InvalidNodeParamsException(name, "userParams");
    }
    PoolParams poolParams = *(PoolParams*)userParams;
    LOG_TRACE(HABANA_NODE, "MaxPoolNode name - {}, params -  H={}, W={}, dH={}, dW={}, operation={}",
              name, poolParams.H, poolParams.W, poolParams.dH, poolParams.dW, poolParams.op);

    return NodePtr(new MaxPoolNode(inputs, outputs, poolParams, name));
}

bool MaxPoolNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    return Node::validateNode();
}

NodePtr MaxPoolNode::clone() const
{
    return NodePtr(new MaxPoolNode(*this));
}

bool MaxPoolNode::RunOnCpu()
{
    TensorPtr in   = getInput(TENSOR_IFM);
    TensorPtr out  = getOutput(TENSOR_OFM);

    unsigned inW  = in->getSizeInElements(1);
    unsigned inH  = in->getSizeInElements(2);
    UNUSED(inH);
    unsigned outW = out->getSizeInElements(1);
    unsigned outH = out->getSizeInElements(2);
    unsigned nFM  = out->getSizeInElements(0);
    //o = ((i - k + 2 * pad) / stride) + 1

    HB_ASSERT(outW == (((inW - m_params.W) / m_params.dW) + 1), "out size is incorrect");
    HB_ASSERT(outH == (((inH - m_params.H) / m_params.dH) + 1), "out size is incorrect");
    HB_ASSERT(out->getSizeInElements(0) == in->getSizeInElements(0), "size mismatch");
    HB_ASSERT(m_params.op == POOL_MAX, "op must be POOL_MAX");

    float* pOut = static_cast<float*>(out->map());
    float* pIn  = static_cast<float*>(in->map());

    for (unsigned fm = 0; fm < nFM; ++fm)
    {
        for (unsigned h = 0; h < outH; h += m_params.dH)
        {
            for (unsigned w = 0; w < outW; w += m_params.dW)
            {
                unsigned in_base = h * inW * nFM + w * nFM + fm;
                float res = 0;
                for (unsigned lh = 0; lh < m_params.H; ++lh)
                {
                    for (unsigned lw = 0; lw < m_params.W; ++lw)
                    {
                        res = std::max(res, pIn[in_base + lh * inW * nFM + lw * nFM]);
                    }
                }
                pOut[h * outW * nFM + w * nFM + fm] = res;
            }
        }
    }
    return true;
}

bool MaxPoolNode::validateNodeForGraph(const HabanaGraph& g) const
{
    // Legacy node. Valid for all graphs
    return true;
}

TPCMemcpyNode::TPCMemcpyNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             std::string_view    name,
                             UserParams          params,
                             unsigned            paramsSize)
: TPCNode(inputs, outputs, name, params, paramsSize)
{
}

NodePtr TPCMemcpyNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  unsigned            userParamsSize,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new TPCMemcpyNode(inputs, outputs, name, userParams, userParamsSize));
}

void TPCMemcpyNode::setGUID(const StringViewWithHash& guidAndHash)
{
    if (m_outputs.size() == 0)
    {
        Node::setGUID("memcpy_na");
        return;
    }
    switch(getOutput(TENSOR_OFM)->getElementType())
    {
    case syn_type_int32:
        Node::setGUID("memcpy_i32");
        break;
    case syn_type_single:
    case syn_type_hb_float:
    case syn_type_tf32:
        Node::setGUID("memcpy_f32");
        break;
    case syn_type_int16:
        Node::setGUID("memcpy_i16");
        break;
    case syn_type_uint16:
        Node::setGUID("memcpy_u16");
        break;
    case syn_type_fp16:
        Node::setGUID("memcpy_f16");
        break;
    case syn_type_bf16:
        Node::setGUID("memcpy_bf16");
        break;
    case syn_type_uint8:
        Node::setGUID("memcpy_u8");
        break;
    case syn_type_fixed:
        Node::setGUID("memcpy_i8");
        break;
    case syn_type_fp8_143:
        Node::setGUID("memcpy_hf8");
        break;
    case syn_type_fp8_152:
        Node::setGUID("memcpy_f8");
        break;
    case syn_type_uint32:
        Node::setGUID("memcpy_u32");
        break;
    case syn_type_int64:
        Node::setGUID("memcpy_i64");
        break;
    case syn_type_uint64:
        Node::setGUID("memcpy_u64");
        break;
    default:
        Node::setGUID("memcpy_na");
        break;
    }
}

bool TPCMemcpyNode::isSupportedNdimDataType(synDataType type)
{
    switch (type)
    {
        case syn_type_int8:
        case syn_type_uint8:
        case syn_type_fp16:
        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_int32:
        case syn_type_uint32:
        case syn_type_single:
        // int\uint64 are supported after precision reduction in GC
        case syn_type_int64:
        case syn_type_uint64:
            return true;
        default:
            return false;
    }
}

bool TPCMemcpyNode::validateNode() const
{
    if (getNumInputsDataTensors() - getNumInputsH2DTensors() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 data input and 1 output)");
        return false;
    }

    if (getGUID() == "memcpy_na")
    {
        LOG_ERR(HABANA_NODE, "unsupported data type for MemcpyNode");
        return false;
    }
    return TPCNode::validateNode();
}

NodePtr TPCMemcpyNode::clone() const
{
    return NodePtr(new TPCMemcpyNode(*this));
}

bool TPCMemcpyNode::RunOnCpu()
{
    unsigned inSize = 1, outSize = 1;
    TensorPtr in  = getInput(TENSOR_IFM);
    TensorPtr out = getOutput(TENSOR_OFM);
    for (unsigned i = 0; i < in->getDim(); ++i)
    {
        inSize *= in->getSizeInElements(i);
    }
    for (unsigned i = 0; i < out->getDim(); ++i)
    {
        outSize *= out->getSizeInElements(i);
    }
    HB_ASSERT(inSize == outSize, "size mismatch");

    memcpy(out->map(), in->map(), inSize);
    return true;
}

FlattenNode::FlattenNode(const TensorVector& inputs,
                         const TensorVector& outputs,
                         UserParams          userParams,
                         std::string_view    name,
                         eNodeType           type)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_FLATTEN)
{
    setParams(userParams, sizeof(synFlattenParams));
    // Flatten is a pragma, nothing actually executes (data does not change)
}

void FlattenNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParamsSize != sizeof(synFlattenParams))
    {
        LOG_ERR(HABANA_NODE, "FlattenNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synFlattenParams));
    }
    synFlattenParams params = *(synFlattenParams*)userParams;
    m_logicalWasForced = FlattenNode::getForceLogicalFlag(params.axis);
    FlattenNode::clearForceLogicalFlag(params.axis);
    m_flattenParams = params;
    LOG_TRACE(HABANA_NODE, "FlattenNode name - {}, params - flattenParams={}", getNodeName(), m_flattenParams.axis);
}

NodePtr FlattenNode::createNode(const TensorVector& inputs,
                                const TensorVector& outputs,
                                UserParams          userParams,
                                std::string_view    guid,
                                std::string_view    name)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "FlattenNode userParams is null");
        throw InvalidNodeParamsException(std::string {name}, "userParams");
    }
    synFlattenParams params = *(synFlattenParams*)userParams;
    bool             enforce_logical = FlattenNode::getForceLogicalFlag(params.axis);

    LOG_TRACE(HABANA_NODE, "FlattenNode name - {}, params - axis={}", name, params.axis);

    if (!enforce_logical && inputs[0]->isDynamicShape())
    {
        FlattenNode::clearForceLogicalFlag(params.axis);
        return NodeFactory::createNode(inputs, outputs, &params, NodeFactory::physicalFlattenNodeTypeName, name);
    }
    return NodePtr(new FlattenNode(inputs, outputs, userParams, name));
}

bool FlattenNode::validateNode() const
{

    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    const TensorPtr& in  = getInput(TENSOR_IFM);
    const TensorPtr& out = getOutput(TENSOR_IFM);

    if (!LogicalOpNode::validateNode())
    {
        return false;
    }

    // Verification - number-of-elements
    unsigned outTotalElem(out->getDenseSizeInElements());
    unsigned inTotalElem(in->getDenseSizeInElements());
    if ( outTotalElem != inTotalElem )
    {
        LOG_ERR(HABANA_NODE, "Output tensor and input tensor doesn't match in elements' count ( {} , {} )" , outTotalElem , inTotalElem );
        return false;
    }

    // If logical node was forced despite some dimensions being dynamic,
    // we should not disallow such node
    // because it is sandwiched between Serialize and Deserialize nodes
    // so flattening is well-defined.
    if (m_logicalWasForced)
    {
        return true;
    }

    // verification - flattened dimensions are not dynamic
    if (!in->isShapeTensor())
    {
        for (unsigned dim = 0; dim < in->getDim() - 1; dim++)  // dynamic SCD is allowed
        {
            if (in->isDynamicDim(dim))
            {
                LOG_ERR(HABANA_NODE,
                        "flatten on dynamic dimensions is not allowed. use reshape instead. node {}",
                        m_name);
                return false;
            }
        }
    }

    return true;
}

NodePtr FlattenNode::clone() const
{
    return NodePtr(new FlattenNode(*this));
}

unsigned FlattenNode::axis() const
{
    return m_flattenParams.axis;
}

void FlattenNode::runLogicalOperation() const
{
    AliasDirection aliasDirection = getAliasDirection();

    LOG_TRACE(OPT_LOGICAL_OPS, "{}: AliasDirection: {}", HLLOG_FUNC, aliasDirection);

    if (aliasDirection == OUTPUT_TO_INPUT)
    {
        m_outputs.front()->setAsFlattenSubTensor( m_inputs.front(), false, m_flattenParams.axis );
    }
    else
    {
        m_inputs.front()->setAsFlattenSubTensor( m_outputs.front(), true, m_flattenParams.axis );
    }
}

bool FlattenNode::RunOnCpu()
{
    TensorPtr  pInputTensor   = getInput(TENSOR_IFM);
    TensorPtr  pOutputTensor  = getOutput(TENSOR_OFM);
    char*    pInputMap      = nullptr;
    char*    pOutputMap     = nullptr;

    // Consistency verification
    {
        HB_ASSERT( pOutputTensor->getElementType() == pInputTensor->getElementType(),
                "Output tensor and input tensor have a different element type"      );
        HB_ASSERT( pOutputTensor->getDenseSizeInElements() == pInputTensor->getDenseSizeInElements(),
                "Output tensor and input tensor doesn't match in elements' count"   );
    }

    // Getting the input and output buffers
    pInputMap  = static_cast<char*>(pInputTensor->map());
    pOutputMap = static_cast<char*>(pOutputTensor->map());

    // Execute - copy the data "as is", as we only change "view"
    // Another (better) option that we may consider is to perform the copy to the output right
    // at the time that the execution to this node takes place
    // But as this is just a SimGraph code, the following is OK.
    memcpy(pOutputMap, pInputMap, pInputTensor->getTotalSizeInBytes());
    return true;
}

bool FlattenNode::isRedundantNode() const
{
    /* if input and output size is 2, and axis is param is 1 this node is redundant */
    return isRedundantNode(*getInput(TENSOR_IFM), *getOutput(TENSOR_OFM), m_flattenParams.axis);
}

bool FlattenNode::isRedundantNode(const Tensor& in, const Tensor& out, uint32_t axis)
{
    return (in.getDim() == 2 && out.getDim() == 2 && axis == 0);
}

void FlattenNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_flattenParams, sizeof(m_flattenParams));
}

SifNodeParams FlattenNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams)&m_flattenParams;
}

size_t FlattenNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_flattenParams);
}

bool FlattenNode::getForceLogicalFlag(unsigned axis)
{
    return !!(axis & 0x80000000);
}

void FlattenNode::setForceLogicalFlag(unsigned& axis)
{
    axis |= 0x80000000;
}

void FlattenNode::clearForceLogicalFlag(unsigned& axis)
{
    axis &= 0x7FFFFFFF;
}

void FlattenNode::permuteParams(const PermutationVector& inputPermutations)
{
    HB_ASSERT(inputPermutations.size() == 1,
              "'flatten' has more than one input permutation, taking the first one to convert params");
    m_flattenParams.axis = inputPermutations[0].permuteDim(m_flattenParams.axis);
}

SplitNode::SplitNode(const TensorVector& inputs,
                     const TensorVector& outputs,
                     UserParams          userParams,
                     std::string_view    name,
                     eNodeType           type)
: BaseClass(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_SPLIT, userParams)
{

}

NodePtr SplitNode::createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name)
{
    return createSplitNode(inputs, outputs, userParams, guid, name, false);
}

NodePtr SplitNode::createNodeInternal(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name)
{
    return createSplitNode(inputs, outputs, userParams, guid, name, true);
}

NodePtr SplitNode::createSplitNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name,
                                   bool                isInternalNode)
{
    unsigned dim = 0;

    if (userParams != nullptr)
    {
        dim = reinterpret_cast<synSplitParams*>(userParams)->axis;
        LOG_TRACE(HABANA_NODE, "SplitNode name - {}, params - dim={}", name, dim);
    }

    if (!isInternalNode && checkIfPhysicalSplit(inputs, outputs, dim))
    {
        return NodeFactory::createNode(inputs,
                                       outputs,
                                       &dim,
                                       sizeof(dim),
                                       NodeFactory::physicalSplitNodeTypeName,
                                       name);
    }

    TensorVector operands = inputs;
    operands.insert(operands.end(), outputs.begin(), outputs.end());
    bool isPartOfRmwSection = std::any_of(operands.begin(), operands.end(), [](const TensorPtr& t) {
        if (unlikely(!t)) return false;
        return t->isPartOfRMWSection();
    });

    bool enableFcdOptimization =
        (dim == 0) && !isInternalNode && !isPartOfRmwSection &&
        !is64BitOperands(inputs,
                         outputs);  // Don't optimize when operation is 64bit, not supported on physical operations

    if (enableFcdOptimization)
    {
        // Optimized version for split on FCD.
        return NodePtr(new SplitFcdNode(inputs, outputs, name));
    }

    return NodePtr(new SplitNode(inputs, outputs, userParams, name));
}

bool SplitNode::checkIfPhysicalSplit(const TensorVector& inputs, const TensorVector& outputs, unsigned dim)
{
    return inputs.size() == 2 && inputs[1]->isHostShapeTensor();
}

bool SplitNode::validateNode() const
{
    if ((m_inputs.size() != 1 && m_inputs.size() != 2) || (m_outputs.size() < 1))
    {
        LOG_ERR(HABANA_NODE,
                "Node {} has invalid number of operands: expecting 1 or 2 inputs and one or more outputs",
                getNodeName());
        return false;
    }
    if (m_inputs.size() == 2 && m_inputs[1] && !m_inputs[1]->isHostShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Node {} has invalid inputs: expecting host shape tensor at index 1", getNodeName());
        return false;
    }
    return true;
}

NodePtr SplitNode::clone() const
{
    return NodePtr(new SplitNode(*this));
}

bool SplitNode::RunOnCpu()
{
    unsigned nOutputChannels = 0;
    unsigned spSize          = 0;
    TensorPtr                               input = m_inputs.front();
    std::vector<std::pair<char*, unsigned>> outputPairs;
    unsigned scaleToFloat = 1;

    if (m_outputs.size() == 1)
    {
        // if the output size is 1, we don't need to split the tensor. Simply copy the data pointer to the output tensor.
        m_outputs.front()->setTensorBuffer(input->getData(), input->getBufferSizeInBytes(), input->getBufferDataType(), true);
        return true;
    }

    // when the input tensor is a static param and unquantized, the data is in float32,
    // even though the tensor's type may not be float32.
    // because of that we scale the data size to match float32 size, and rebind the output tensor to match float32 type.
    if (input->isStaticParam() && !input->isDataTypeMatchData())
    {
        scaleToFloat = sizeof(float_t) / input->getElementSizeInBytes();
        for (auto t : m_outputs)
        {
            unsigned output_size = t->getTotalSizeInBytes() * scaleToFloat;
            if (t->isBound())
            {
                t->rebind(output_size);
            }
        }
    }

    for (auto t : m_outputs)
    {
        nOutputChannels += t->getSizeInElements(m_aggDim);
        outputPairs.push_back(std::pair<char*, unsigned>(static_cast<char*>(t->map()), t->getSizeInBytes(m_aggDim) * scaleToFloat));

        unsigned SP = 1;
        for (unsigned dim = 0; dim < t->getDim(); dim++)
        {
            if (dim != m_aggDim)
            {
                SP *= t->getSizeInElements(dim);
            }
        }

        if (spSize == 0)
        {
            spSize = SP;
        }
        else
        {
            HB_ASSERT(spSize == SP, "size inconsistency between outputs");
        }
    }
    HB_ASSERT(nOutputChannels == input->getSizeInElements(m_aggDim),
              "channel inconsistency between outputs and input");
    unsigned inputSP = 1;
    for (unsigned dim = 0; dim < input->getDim(); dim++)
    {
        if (dim != m_aggDim)
        {
            inputSP *= input->getSizeInElements(dim);
        }
    }
    HB_ASSERT(spSize == inputSP, "spatial size inconsistency between outputs and input");
    char* inputData = static_cast<char*>(input->map());

    for (unsigned spIndex = 0; spIndex < spSize; ++spIndex)
    {
        for (auto& it : outputPairs)
        {
            memcpy(it.first, inputData, it.second);
            it.first += it.second;
            inputData  += it.second;
        }
    }
    return true;
}

SifNodeParams SplitNode::getShapeInferenceFunctionUserParams(std::vector<uint8_t>& metadataBuffer,
                                                             const size_t          bufferSize,
                                                             const unsigned        aggregationDim,
                                                             const TensorVector&   outputs)
{
    metadataBuffer.resize(bufferSize);

    auto metadata    = reinterpret_cast<SifSplitMetadata*>(metadataBuffer.data());
    metadata->header = {aggregationDim, outputs.size()};

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        metadata->splitDimSizes[i] = outputs[i]->getSizeInElements(aggregationDim);
    }

    return reinterpret_cast<SifNodeParams>(metadataBuffer.data());
}

SifNodeParams SplitNode::getShapeInferenceFunctionUserParams()
{
    return getShapeInferenceFunctionUserParams(m_sifMetadataBuffer,
                                               getShapeInferenceFunctionUserParamsSize(),
                                               m_aggDim,
                                               m_outputs);
}

size_t SplitNode::getShapeInferenceFunctionUserParamsSize(const unsigned numOutputs)
{
    uint32_t headerSize = sizeof(SifSplitHeader);
    uint32_t dataSize   = numOutputs * sizeof(TSize);
    return headerSize + dataSize;
}

size_t SplitNode::getShapeInferenceFunctionUserParamsSize() const
{
    return getShapeInferenceFunctionUserParamsSize(m_outputs.size());
}

//Out is in with "1" inserted in the tensor sizes in index "dim"
ExpandDimsNode::ExpandDimsNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    name,
                               eNodeType           type)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, type, SIF_EXPAND_DIMS)
{
    setParams(userParams, sizeof(unsigned));
}

NodePtr ExpandDimsNode::createNode(const TensorVector& inputs,
                                   const TensorVector& outputs,
                                   UserParams          userParams,
                                   std::string_view    guid,
                                   std::string_view    name)
{
    return NodePtr(new ExpandDimsNode(inputs, outputs, userParams, name));
}

void ExpandDimsNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "ExpandDimNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize != sizeof(unsigned))
    {
        LOG_ERR(HABANA_NODE, "ExpandDimsNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(unsigned));
    }
    unsigned dim = *(unsigned*)userParams;
    m_expandDim  = dim;
    LOG_TRACE(HABANA_NODE, "ExpandDimNode name - {}, params - dim={}", getNodeName(), m_expandDim);
}

gc::access_pattern::NodeAccessPatternPtr ExpandDimsNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternExpandDimsGenerator::generate(this, m_expandDim);
}

bool ExpandDimsNode::validateNode() const
{
    if (m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }

    const TensorPtr& in  = m_inputs[TENSOR_IFM];
    const TensorPtr& out = m_outputs[TENSOR_OFM];
    if (m_expandDim > in->getDim() || m_expandDim >= Tensor::c_tensorMaxNDim)
    {
        LOG_ERR(HABANA_NODE, "Invalid dimensions to expand");
        return false;
    }

    /* should be added after [SW-55206] is resolved
    if (in->getDim() + 1 != out->getDim())
    {
        LOG_ERR(HABANA_NODE, "dimensions mismatch for {}", getNodeName());
        return false;
    }
    */
    if (out->getDim() > in->getDim() + 1)  // WA - should be removed after [SW-55206] is resolved
    {
        HB_ASSERT(m_expandDim == 1, "");
        HB_ASSERT(in->getDim() == 1, "");
        for (unsigned dim = in->getDim() + 2; dim < out->getDim(); dim++)
        {
            HB_ASSERT(out->getSizeInElements(dim) == 1, "");
        }
    }

    bool valid = true;
    for (unsigned dim = 0; dim < m_expandDim; dim++)
    {
        valid &= in->getSizeInElements(dim) == out->getSizeInElements(dim);
    }
    valid &= out->getSizeInElements(m_expandDim) == 1;
    for (unsigned dim = m_expandDim; dim < in->getDim(); dim++)
    {
        valid &= in->getSizeInElements(dim) == out->getSizeInElements(dim + 1);
    }
    if (!valid)
    {
        LOG_ERR(HABANA_NODE, "dimensions mismatch for {}", getNodeName());
        return false;
    }
    return Node::validateNode();
}

bool ExpandDimsNode::isNode64BitCompatible() const
{
    return true;
}

NodePtr ExpandDimsNode::clone() const
{
    return NodePtr(new ExpandDimsNode(*this));
}

bool ExpandDimsNode::canHandleStridedRealTensor() const
{
    if ((getAliasDirection() == INPUT_TO_OUTPUT) && (m_expandDim == 0))  // input is the alias
    {
        // if the output is strided on dimension 1, then the resulting input will be strided on dimension 0.
        const TensorPtr& out = m_outputs.front();
        return out->getStrideInBytes(1) == out->getStrideInBytes(0);  // since sizeInElements(0) is 1
    }
    return true;
}

NStrideArray ExpandDimsNode::calculateAliasStrides(unsigned idx) const
{
    HB_ASSERT(idx == 0, "{}: Illegal alias idx {}", HLLOG_FUNC, idx);
    const TensorPtr& in                               = m_inputs.front();
    const TensorPtr& out                              = m_outputs.front();
    NStrideArray     strides                          = {0};
    unsigned         dim                              = 0;
    // set output tensor as alias to the input tensor
    if (this->getAliasDirection() == OUTPUT_TO_INPUT)
    {
        for (; dim < m_expandDim; dim++)  // strides before expanded dim
        {
            strides[dim] = in->getStrideInBytes(dim);
        }
        strides[m_expandDim] = in->getDenseStrideInElements(m_expandDim) * in->getElementSizeInBytes();
        for (; dim < in->getDim() + 1; dim++)  // strides after expanded dim
        {
            strides[dim + 1] = in->getStrideInBytes(dim);
        }

        if (out->getDim() > in->getDim() + 1)  // WA - should be removed after [SW-55206] is resolved
        {
            for (unsigned redundantDim = dim; redundantDim < out->getDim() + 1; redundantDim++)
            {
                strides[redundantDim] = strides[dim];
            }
        }
    }
    else  // set input tensor as alias to the output tensor
    {
        for (; dim < m_expandDim; dim++)
        {
            strides[dim] = out->getStrideInBytes(dim);
        }
        for (; dim < in->getDim() + 1; dim++)
        {
            strides[dim] = out->getStrideInBytes(dim + 1);
        }
    }
    return strides;
}

void ExpandDimsNode::runLogicalOperation() const
{
    NStrideArray     strides  = calculateAliasStrides(0);
    const TensorPtr real     = getRealTensor();
    const TensorPtr alias    = getAliasTensors().front();
    const auto&      sizes    = alias->getAllNSizesInElements();
    const auto&      minSizes = alias->getNMinimalSizesInElements();
    alias->reshape(alias->getDim(), sizes.data(), strides.data(), minSizes.data());
    alias->setAsAliasSubTensor(real);
}

bool ExpandDimsNode::isAliasStrided() const
{
    const TensorPtr& real = getRealTensor();
    return real->isDenseLayout();
}

void ExpandDimsNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_expandDim, sizeof(m_expandDim));
}

void ExpandDimsNode::permuteParams(const PermutationVector& inputPermutations)
{
    for (const auto& p : inputPermutations)
    {
        HB_ASSERT(p == inputPermutations[0], "Cannot convert params. All input permutations should be identical");
    }
    m_expandDim = inputPermutations[0].permuteDim(m_expandDim);
}

bool ExpandDimsNode::RunOnCpu()
{
    return constFoldingForReshape();
}

SifNodeParams ExpandDimsNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams)&m_expandDim;
}

size_t ExpandDimsNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_expandDim);
}

SliceAxisNode::SliceAxisNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          params,
                             unsigned            userParamsSize,
                             std::string_view    name)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, TYPE_SLICE_AXIS, SIF_SLICE_AXIS)
{
    setParams(params, userParamsSize);
}

NodePtr SliceAxisNode::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  unsigned            userParamsSize,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new SliceAxisNode(inputs, outputs, userParams, userParamsSize, name));
}

void SliceAxisNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "SliceAxisNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize == sizeof(synSliceAxisParams))
    {
        auto oldParams = static_cast<synSliceAxisParams*>(userParams);
        m_params.axis  = oldParams->axis;
        m_params.begin = oldParams->begin;
        m_params.end   = oldParams->end;
    }
    else if (userParamsSize == sizeof(synSliceAxisParamsV2))
    {
        m_params = *(synSliceAxisParamsV2*)userParams;
    }
    else
    {
        LOG_ERR(HABANA_NODE, "SliceAxisNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name);
    }
    LOG_TRACE(HABANA_NODE,
              "SliceAxisNode name - {}, params - axis={}, begin={}, end={}",
              m_name,
              m_params.axis,
              m_params.begin,
              m_params.end);
}

bool SliceAxisNode::validateNode() const
{
    if (m_inputs.size() > 2 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }
    return LogicalOpNode::validateNode();
}

NodePtr SliceAxisNode::clone() const
{
    return NodePtr(new SliceAxisNode(*this));
}

NStrideArray SliceAxisNode::calculateAliasStrides(unsigned idx) const
{
    const TensorPtr& real = getAliasDirection() == OUTPUT_TO_INPUT ? m_inputs[0] : m_outputs[0];
    NStrideArray     ret  = {1};
    real->getNStridesInBytes(ret.data());
    return ret;
}

void SliceAxisNode::runLogicalOperation() const
{
    // set output tensor as alias to the input tensor

    TensorPtr in  = m_inputs.front();
    TensorPtr out = m_outputs.front();

    unsigned concatDimStride = in->getElementSizeInBytes();
    if (m_params.axis > 0)
    {
        concatDimStride = in->getStrideInBytes(m_params.axis);
    }
    out->setAsConcatSubTensor(in, concatDimStride * m_params.begin, m_params.axis);
}

bool SliceAxisNode::RunOnCpu()
{
    //todo: Implement SliceAxisNode operator (SW-2370)
    return false;
}

bool SliceAxisNode::isRedundantNode() const
{
    if (isDynamicShape()) return false;

    SizeArray inputSizes = getInput(TENSOR_IFM)->getAllSizesInElements();
    return (m_params.begin == 0 && m_params.end == inputSizes[m_params.axis]);
}

void SliceAxisNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

void SliceAxisNode::permuteParams(const PermutationVector& inputPermutations)
{
    HB_ASSERT(inputPermutations.size() == 1,
              "'SliceAxis' has more than one input permutation, taking the first one to convert params");
    m_params.axis = inputPermutations[0].permuteDim(m_params.axis);
}

SifNodeParams SliceAxisNode::getShapeInferenceFunctionUserParams()
{
    return (SifNodeParams)&m_params;
}

size_t SliceAxisNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_params);
}

synDataType SliceAxisNode::getRequiredInputType(uint32_t tensorIdx) const
{
    // shape tensor (if exists) doesn't have to be in the same type as the real data tensor
    if (tensorIdx == 1)
    {
        return Node::getInput(tensorIdx)->getElementType();
    }
    return BaseClass::getRequiredInputType(tensorIdx);
}

ReshapeNode::ReshapeNode(const TensorVector& inputs,
                         const TensorVector& outputs,
                         std::string_view    name,
                         eNodeType           type,
                         ShapeFuncID         sifId)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, type, sifId)
{
}

NodePtr ReshapeNode::createNode(const TensorVector& inputs,
                                const TensorVector& outputs,
                                UserParams          userParams,
                                std::string_view    guid,
                                std::string_view    name)
{
    bool enforceLogical = false;
    if (userParams != nullptr)
    {
        enforceLogical = *reinterpret_cast<bool*>(userParams);
    }
    if (!enforceLogical && PhysicalReshapeNode::requiresPhysicalReshapeToHandleDynamicity(*inputs[0], *outputs[0]))
    {
        return NodeFactory::createNode(inputs, outputs, nullptr, NodeFactory::physicalReshapeNodeTypeName, name);
    }
    if (inputs.size() == 1 && inputs[0]->isShapeTensor())
    {
        return NodeFactory::createNode(inputs, outputs, nullptr, NodeFactory::staticReshapeShapeNodeTypeName, name);
    }

    return NodePtr(new ReshapeNode(inputs, outputs, name));
}

bool ReshapeNode::validateNode() const
{
    if ((m_inputs.size() != 1 && m_inputs.size() != 2) || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 or 2 inputs and 1 output)");
        return false;
    }
    if (m_inputs.size() == 2 && !m_inputs.back()->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "Invalid inputs, expecting shape tensor at index 1");
        return false;
    }

    const auto& in  = getInput(TENSOR_IFM);
    const auto& out = getOutput(TENSOR_OFM);

    unsigned inTotalElem(in->getDenseSizeInElements());
    unsigned outTotalElem(out->getDenseSizeInElements());

    if (outTotalElem != inTotalElem)
    {
        LOG_ERR(HABANA_NODE, "Output tensor and input tensor of {} doesn't match in elements' count ( {} , {} )",
                getNodeName(), outTotalElem , inTotalElem );
        return false;
    }
    return Node::validateNode();
}

bool ReshapeNode::validateDynamicShapes() const
{
    if (isDynamicShape())
    {
        // Dynamic reshape node must have 2 inputs
        // unless static reshape flag is set.
        // If the second input exists, it is verified to be a shape tensor
        // The verification is done in validateNode()
        if (m_inputs.size() != 2)
        {
            LOG_ERR(HABANA_NODE, "Shape tensor missing in node {}", getNodeName());
            return false;
        }
    }
    return true;
}

NodePtr ReshapeNode::clone() const
{
    return NodePtr(new ReshapeNode(*this));
}

void ReshapeNode::setMustBeDenseIfNeeded() const
{
    if (this->getAliasDirection() == OUTPUT_TO_INPUT)
    {
        m_inputs[0]->getTensorAnnotation().dataInfo.mustBeDense = true;
    }
    else
    {
        m_outputs[0]->getTensorAnnotation().dataInfo.mustBeDense = true;
    }
}

void ReshapeNode::runLogicalOperation() const
{
    TensorPtr real  = m_inputs.front();
    TensorPtr alias = m_outputs.front();
    if (this->getAliasDirection() != OUTPUT_TO_INPUT)
    {
        // The other way around
        std::swap(real, alias);
    }
    alias->setAsAliasSubTensor(real);
    correctStrides(real, alias);
}

// valid mapping satisfied the following conditions:
// (1) there are no empty ranges.
// (2) the ranges are linear partition of the tensor dims (if not it mean that we can't handle the strided real tensor)
//     Example: [(0,1), (2), (3,4,5)] is valid partition of {0,1,2,3,4,5}
// (3) if dynamic dims exists, all of them are in the end of ranges
static bool
validateMapping(const TensorShape& realDenseShape, const TensorPtr& alias, const ReshapeOutputToInputMapping& mapping)
{
    int last = -1;
    uint64_t elementsCounter = 1;
    uint64_t totalElements   = alias->getDenseSizeInElements();
    for (unsigned realDim = 0; realDim < mapping.size(); ++realDim)
    {
        const auto& range = mapping[realDim];
        if (range.empty()) return false;
        // Validate that the ranges are linear partition of {0, ..., tensor dim -1},
        // a special case is when there exists an outer dimensions of size 1.
        // However in that case the elements counter is equal to the total number of elements
        if ((elementsCounter != totalElements) && (last != static_cast<int>(range.at(0)) - 1)) return false;
        last = range.back();

        // validate that there is no dynamic dim that is not end of range
        for (unsigned i = 0; i < range.size() - 1; ++i)
        {
            if (alias->isDynamicDim(range[i]))
            {
                // dim that is not end of range allowed in case the remaining dimensions have size 1 for each.
                if (realDim != mapping.size() - 1)
                {
                    return false;
                }
                for (unsigned j = i + 1; j < range.size() - 1; j++)
                {
                    if (alias->getSizeInElements(range[j]) != 1)
                    {
                        return false;
                    }
                }
            }
        }
        elementsCounter *= realDenseShape.getSize(realDim);
    }
    return true;
}

static ReshapeOutputToInputMapping getRealDenseToAliasMapping(const TensorShape& denseShape, const TensorPtr& alias)
{
    return SlicedOperandUtils::getReshapeOutputToInputMapping(alias->getShape(), denseShape);
}

// Only relevant for non-dense real tensors.
// We try to merge all dense dimensions in real, and then check if we can map each dense dimension
// to 1 or more dimensions in the alias tensor. A matching will require that the multiplication
// of all the sizes of alias dimensions are the size of the dense dimension in real.
// The matching must preserve order, so that you must match the lower dimensions first. Both in real
// and alias.
// If we can't have that kind of matching we return false. Otherwise, returns the strides for the alias.
// Positive example:
//          real tensor:      maxSizes [2,3,4,12,4,3], minSizes [2,3,2,12,4,3],       strides [1,2,6,24,288,2000,6000]
//          aliased tensor:   maxSizes [2,12,3,16,3],  minSizes [2,6,3,16,3]
//          real dense shape: maxSizes [24,48,3],      minSizes [12,48,3], real dense strides [1,24,2000,6000]
//          mapping: [(0,1), (2,3), (4)]
//          new strides: [1,2,24,72,2000,6000]
// Negative example (static shape):
//          real tensor:      sizes [2,4,2], strides [1,10,40,80]
//          aliased tensor:   sizes [4,4]
//          real dense shape: sizes [2,8], real dense strides [1,10,80]
//          mapping: [(0), (0,1)] this is not a partition of {0,1} so it return invalid strides
std::pair<bool, NStrideArray> ReshapeNode::tryGetNewStrides(const TensorPtr& real, const TensorPtr& alias) const
{
    NStrideArray newStrides = {0};
    if (real->isStridedOnFCD() || alias->isStridedOnFCD()) return std::make_pair(false, newStrides);

    auto [realDenseShape, realDenseStrides] = mergeDenseDimensions(real);
    auto mapping                            = getRealDenseToAliasMapping(realDenseShape, alias);

    // in case that the mapping is invalid, return invalid strides.
    if (!validateMapping(realDenseShape, alias, mapping)) return std::make_pair(false, newStrides);

    newStrides[0]             = real->getElementSizeInBytes();
    uint64_t prevStrideInBits = real->getElementSizeInBits();
    for (unsigned aliasDim = 0, realDenseDim = 0; aliasDim < alias->getDim(); ++aliasDim)
    {
        newStrides[aliasDim + 1] = safeBitsToByte(prevStrideInBits * alias->getSizeInElements(aliasDim));
        if (aliasDim == mapping.at(realDenseDim).back())
        {
            newStrides[aliasDim + 1] = realDenseStrides[++realDenseDim];
        }
        prevStrideInBits = newStrides[aliasDim + 1] * BITS_PER_BYTE;
    }
    return std::make_pair(true, newStrides);
}

void ReshapeNode::correctStrides(const TensorPtr& real, const TensorPtr& alias) const
{
    HB_ASSERT(!real->isStridedOnFCD(), "real in reshape {} is strided on fcd", real->getName());
    HB_ASSERT(!alias->isStridedOnFCD(), "alias in reshape {} is strided on fcd", alias->getName());
    if (real->isDenseLayout()) return;
    bool success;
    NStrideArray newStrides;
    std::tie(success, newStrides) = tryGetNewStrides(real, alias);
    HB_ASSERT(success,
              "Cannot reshape: {} to {} withh strides",
              toString(real->getAllSizesInElements(), ','),
              toString(alias->getAllSizesInElements(), ','));
    alias->reshape(alias->getDim(), alias->getAllNSizesInElements().data(), newStrides.data());
}

NStrideArray ReshapeNode::calculateAliasStrides(unsigned idx) const
{
    auto [success, newStrides] = tryGetNewStrides(getRealTensor(), getAliasTensors().front());
    // if we cannot propagate strides from input to output, return the dense strides - base class method
    return success ? newStrides : BaseClass::calculateAliasStrides(idx);
}

bool ReshapeNode::isNode64BitCompatible() const
{
    return true;
}

bool ReshapeNode::isRehsapeOnFcd() const
{
    return m_inputs[TENSOR_IFM]->getSizeInElements(0) != m_outputs[TENSOR_OFM]->getSizeInElements(0);
}

bool ReshapeNode::RunOnCpu()
{
    return constFoldingForReshape();
}

bool ReshapeNode::isRedundantNode() const
{
    return isBasicRedundant();
}

bool ReshapeNode::canHandleStridedRealTensor() const
{
    // Check if the dense sizes of the tensor of which we want to alias could be projected upon
    // sizes of the new aliased tensor.
    TensorPtr real  = m_inputs.front();
    TensorPtr alias = m_outputs.front();
    if (this->getAliasDirection() != OUTPUT_TO_INPUT)
    {
        // The other way around
        std::swap(real, alias);
    }
    // If the strides manipulation did not happen yet
    // There's no way to determine - default false
    if (!real->isAliasedTensor() && real->isDenseLayout()) return false;
    bool success;
    std::tie(success, std::ignore) = tryGetNewStrides(real, alias);
    return success;
}

gc::access_pattern::NodeAccessPatternPtr ReshapeNode::generateNodeAccessPattern() const
{
    return gc::access_pattern::AccessPatternReshapeGenerator::generate(this);
}

void ReshapeNode::permuteParams(const PermutationVector& inputPermutations)
{
    return;
}

StaticReshapeNode::StaticReshapeNode(const TensorVector&       inputs,
                                     const TensorVector&       outputs,
                                     synStaticReshapeSifParams params,
                                     std::string_view          name,
                                     eNodeType                 type)
: ReshapeNode(inputs, outputs, name, type, SIF_STATIC_RESHAPE), m_sifParams(std::move(params))
{
}

StaticReshapeNode::StaticReshapeNode(const StaticReshapeNode& other)
: ReshapeNode(other), m_sifParams(other.m_sifParams)
{
}
StaticReshapeNode& StaticReshapeNode::operator=(const StaticReshapeNode& other)
{
    if (this != &other)
    {
        ReshapeNode::operator=(other);
        m_sifParams = other.m_sifParams;
    }
    return *this;
}

NodePtr StaticReshapeNode::createNode(const TensorVector& inputs,
                                      const TensorVector& outputs,
                                      UserParams          userParams,
                                      std::string_view    guid,
                                      std::string_view    name)
{
    HB_ASSERT(!inputs.empty(), "missing input for reshape {}", name);
    HB_ASSERT(!outputs.empty() > 0, "missing input for reshape {}", name);
    return NodePtr(new StaticReshapeNode(inputs, outputs, createParamsFromTensors(inputs[0], outputs[0]), name));
}

bool StaticReshapeNode::validateNode() const
{
    if(m_inputs.size() != 1 || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting 1 input and 1 output)");
        return false;
    }

    return ReshapeNode::validateNode();
}

bool StaticReshapeNode::validateDynamicShapes() const
{
    return true;
}

NodePtr StaticReshapeNode::clone() const
{
    return NodePtr(new StaticReshapeNode(*this));
}

void StaticReshapeNode::detectDynamicDims(const TensorPtr& operand, char* staticDims)
{
    unsigned maxDim = std::min(operand->getDim(), SIF_NUM_DIMS);
    if (!operand->isDynamicShape())
    {
        std::fill_n(staticDims, maxDim, 1);
    }
    else
    {
        for (unsigned i = 0; i < maxDim; ++i)
        {
            staticDims[i] = (char)!operand->isDynamicDim(i);
        }
    }
}

void StaticReshapeNode::updateSifMaxSizes()
{
    for (unsigned i = 0; i < SIF_NUM_DIMS; i++)
    {
        m_sifParams.outputMaxSizes[i] = m_outputs[0]->getSizeInElements(i);
        m_sifParams.inputMaxSizes[i]  = m_inputs[0]->getSizeInElements(i);
    }
}

synStaticReshapeSifParams StaticReshapeNode::createParamsFromTensors(const TensorPtr& input, const TensorPtr& output)
{
    synStaticReshapeSifParams params = {{0}};
    params.dimsNum = output->getDim();
    if (params.dimsNum > SIF_NUM_DIMS)
    {
        for (unsigned d = SIF_NUM_DIMS; d < params.dimsNum; d++)
        {
            HB_ASSERT(!output->isDynamicDim(d), "cannot support dynamic dim larger than {}", SIF_NUM_DIMS);
        }
        params.dimsNum = SIF_NUM_DIMS;
    }

    for (unsigned i = 0; i < SIF_NUM_DIMS; i++)
    {
        params.outputMaxSizes[i] = output->getSizeInElements(i);
        params.inputMaxSizes[i]  = input->getSizeInElements(i);
    }

    detectDynamicDims(input, params.inputStaticDims);
    detectDynamicDims(output, params.outputStaticDims);
    return params;
}

SifNodeParams StaticReshapeNode::getShapeInferenceFunctionUserParams()
{
    updateSifMaxSizes();
    return static_cast<SifNodeParams>(&m_sifParams);
}

size_t StaticReshapeNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(m_sifParams);
}

LoweringNode::LoweringNode(const TensorPtr& in, const TensorPtr& out, const std::string& name, unsigned int loweringFactor) :
    ReshapeNode({in}, {out}, name, Node::TYPE_INTERNAL_LOWERING),
    m_loweringFactor(loweringFactor)
{
}

NodePtr LoweringNode::clone() const
{
    return NodePtr(new LoweringNode(*this));
}

Settable<NodeROI> LoweringNode::getInputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    NodeROI ret(roi);
    ret.size[DIM_C] /= m_loweringFactor;

    return ret;
}

PackingNode::PackingNode(const TensorPtr& in, const TensorPtr& out, const std::string& name, unsigned int packingFactor)
: BaseClass({in}, {out}, createParamsFromTensors(in, out), name, Node::TYPE_INTERNAL_PACKING),
  m_packingFactor(packingFactor)
{

}

NodePtr PackingNode::clone() const
{
    return NodePtr(new PackingNode(*this));
}

Settable<NodeROI> PackingNode::getInputROI(const NodeROI& roi, uint32_t tensorIdx) const
{
    NodeROI ret(roi);
    bool forward = (getInput(TENSOR_IFM)->getSizeInElements(DIM_C) < getOutput(TENSOR_OFM)->getSizeInElements(DIM_C));
    if (forward)
    {
        ret.size[DIM_C]  = div_round_up(ret.size[DIM_C], m_packingFactor);
        ret.size[DIM_W] *= m_packingFactor;
        ret.baseOffset[DIM_W] *= m_packingFactor;
        ret.spatialOffset[0] *= m_packingFactor;
    }
    else
    {
        ret.size[DIM_C] *= m_packingFactor;
        ret.size[DIM_W]  = div_round_up(ret.size[DIM_W], m_packingFactor);
        ret.baseOffset[DIM_W]  /= m_packingFactor;
        ret.spatialOffset[0] /= m_packingFactor;
    }
    unsigned int numPixels = (roi.numIterations - 1) * roi.vectorSize + roi.spatialSizeMinus1 + 1;
    if (forward)
    {
        numPixels *= m_packingFactor;
    }
    else
    {
        numPixels = div_round_up(numPixels, m_packingFactor);
    }

    if (roi.vectorSize != 0)
    {
        ret.numIterations = div_round_up(numPixels, roi.vectorSize);
        ret.spatialSizeMinus1 = (numPixels - 1) % roi.vectorSize;
    }

    return ret;
}

TensorShape PackingNode::getInputShape(const TensorShape& output, uint32_t outputIndex, uint32_t inputIdx) const
{
    bool forward = (getInput(inputIdx)->getSizeInElements(DIM_C) < getOutput(outputIndex)->getSizeInElements(DIM_C));
    const TensorPtr& tensor = getInput(inputIdx);
    SizeArray size;

    if (tensor == nullptr)
    {
        LOG_ERR(HABANA_NODE, "Node has no input!");
        throw(NodeHasNoInput(getNodeName()));
    }

    tensor->getAllSizesInElements(size);
    if (forward)
    {
        size[DIM_C] *= m_packingFactor;
        size[DIM_W]  = div_round_up(size[DIM_W], m_packingFactor);
    }
    else
    {
        size[DIM_C]  = div_round_up(size[DIM_C], m_packingFactor);
        size[DIM_W] *= m_packingFactor;
    }
    TensorShape inputShape(tensor->getDim(), size);

    return inputShape;
}

RotateNode::RotateNode(const TensorVector& inputs,
                       const TensorVector& outputs,
                       UserParams          userParams,
                       std::string_view    name)
: Node(inputs, outputs, name, Node::TYPE_ROTATE, SIF_ROTATE),
  m_parallelLevel(1),
  m_coordinate_mode(0),     // coordinate mode (0 - fixed point, 1 - fp32) (default - fixed point)
  m_rotation_mode(0),       // rotation mode (0 - rotation, 1 - affine, 2 - projection, 3 - mesh) (default - rotation)
  m_interpolation_mode(0),  // interpolation  mode (0 - bilinear, 1 - nearest neigbour) (default - bilinear)
  m_input_pixel_width(8),
  m_output_pixel_width(8)
{
    setParams(userParams, sizeof(synRotateParams));
}

NodePtr RotateNode::createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    guid,
                               std::string_view    name)
{
    return std::make_shared<RotateNode>(inputs, outputs, userParams, name);
}

void RotateNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(GC, "Rotate userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    if (userParamsSize != sizeof(synRotateParams))
    {
        LOG_ERR(HABANA_NODE, "RotateNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synRotateParams));
    }
    synRotateParams params = *(synRotateParams*)userParams;
    m_angle                = params.m_angle;
    m_inputCenterX         = params.m_inputCenterX;
    m_inputCenterY         = params.m_inputCenterY;
    m_outputCenterX        = params.m_outputCenterX;
    m_outputCenterY        = params.m_outputCenterY;
    m_backgroundPixel      = params.m_background;
    m_isDumpDescriptors    = params.m_isDumpDescriptors;
    m_descFilePrefix       = params.m_descFilePrefix;
    LOG_TRACE(
        GC,
        "Rotate Node name - {}, Node params - angle={}, input center=({},{}), output center=({},{}, background={})",
        m_name,
        m_angle,
        m_inputCenterX,
        m_inputCenterY,
        m_outputCenterX,
        m_outputCenterY,
        m_backgroundPixel);
}

NodePtr RotateNode::clone() const
{
    return NodePtr(new RotateNode(*this));
}

bool RotateNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getHALReader()->getNumRotatorEngines() > 0);
}

bool RotateNode::validateNode() const
{
    if ((m_inputs.size() != 1 && m_inputs.size() != 2) || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "{}, Invalid number of operands (expecting 1 or 2 inputs and 1 output)", getNodeName());
        return false;
    }
    if (m_inputs.size() == 2 && !m_inputs.back()->isShapeTensor())
    {
        LOG_ERR(HABANA_NODE, "{}, Invalid inputs, expecting shape tensor at index 1", getNodeName());
        return false;
    }

    float rotation_angle = getRotationAngle();
    // Verify that the rotation angle is within 0 to 360
    if ( rotation_angle < 0.0 || rotation_angle > 360.0 )
    {
        LOG_ERR(GC, "Rotation angle must be between 0 to 360 degrees");
        return false;
    }
    return true;
}

void RotateNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void *)(&m_angle), sizeof(m_angle));
}

std::string_view RotateNode::getEngineTypeStr() const
{
    return "ROT";
}

WaitNode::WaitNode(std::string_view name, UserParams userParams) : Node({}, {}, name, TYPE_WAIT, SIF_WAIT)
{
    setParams(userParams, sizeof(synWaitParams));
}

NodePtr WaitNode::createNode(const TensorVector& inputs,
                             const TensorVector& outputs,
                             UserParams          userParams,
                             std::string_view    guid,
                             std::string_view    name)
{
    HB_ASSERT(inputs.empty(),  "wait node should not have inputs");
    HB_ASSERT(outputs.empty(), "wait node should not have outputs");
    return NodePtr(new WaitNode(name, userParams));
}

void WaitNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    HB_ASSERT_PTR(userParams);
    if (userParamsSize != sizeof(synWaitParams))
    {
        LOG_ERR(HABANA_NODE, "WaitNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synWaitParams));
    }
    synWaitParams params = *(synWaitParams*)userParams;
    m_waitCycles         = params.waitCycles;
    LOG_TRACE(HABANA_NODE, "WaitNode name - {}, Node params - waitCycles={}", m_name, m_waitCycles);
}

NodePtr WaitNode::clone() const
{
    return NodePtr(new WaitNode(*this));
}

bool WaitNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

void WaitNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_waitCycles, sizeof(m_waitCycles));
}

DebugNodeBase::DebugNodeBase(TensorVector inputs, TensorVector outputs, std::string_view name, Node::eNodeType type)
: Node(inputs, outputs, name, type, SIF_DEBUG)
{
}

NodePtr DebugNodeBase::createNode(const TensorVector& inputs,
                                  const TensorVector& outputs,
                                  UserParams          userParams,
                                  std::string_view    guid,
                                  std::string_view    name)
{
    return NodePtr(new DebugNodeBase(inputs, outputs, name));
}

NodePtr DebugNodeBase::clone() const
{
    return NodePtr(new DebugNodeBase(*this));
}

bool DebugNodeBase::validateNodeForGraph(const HabanaGraph& g) const
{
    return true;
}

DebugNode::DebugNode(const TensorPtr& opA, const TensorPtr& opB, const std::string& name)
: DebugNodeBase(TensorVector({ opA }), TensorVector{opB}, name)
{
}

Debug2Node::Debug2Node(const TensorPtr& opA, const TensorPtr& opB, const std::string& name)
: DebugNodeBase(TensorVector({ opA }), TensorVector{opB}, name, Node::TYPE_DEBUG2)
{
    opB->setAsAliasSubTensor(opA);
}

DebugForkNode::DebugForkNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name)
: DebugNodeBase(TensorVector({ opA }), TensorVector{opB, opC}, name)
{
}

DebugJoinNode::DebugJoinNode(const TensorPtr& opA, const TensorPtr& opB, const TensorPtr& opC, const std::string& name)
: DebugNodeBase(TensorVector({ opA, opB }), TensorVector{opC}, name)
{
}

LogicalRequantNode::LogicalRequantNode(const TensorVector& inputs, const TensorVector& outputs, std::string_view name)
: LogicalOpNode(inputs, outputs, name, OUTPUT_TO_INPUT, TYPE_LOGICAL_REQUANT, SIF_NO_SUPPORT)
{
}

NodePtr LogicalRequantNode::createNode(const TensorVector& inputs,
                                       const TensorVector& outputs,
                                       UserParams          userParams,
                                       std::string_view    guid,
                                       std::string_view    name)
{
    return std::make_shared<LogicalRequantNode>(inputs, outputs, name);
}

NodePtr LogicalRequantNode::clone() const
{
    return NodePtr(new LogicalRequantNode(*this));
}

void LogicalRequantNode::runLogicalOperation() const
{
    TensorPtr in  = m_inputs.front();
    TensorPtr out = m_outputs.front();
    if (getAliasDirection() == OUTPUT_TO_INPUT)
    {
        // set output tensor as alias to the input tensor
        out->setAsAliasSubTensor(in);
    }
    else //INPUT_TO_OUTPUT
    {
        // set input tensor as alias to the output tensor
        in->setAsAliasSubTensor(out);
    }
}

bool LogicalRequantNode::RunOnCpu()
{
    TensorPtr pInputTensor  = m_inputs[TENSOR_IFM];
    TensorPtr pOutputTensor = m_outputs[TENSOR_OFM];
    char*     pInputMap     = nullptr;
    char*     pOutputMap    = nullptr;

    // Consistency verification
    {
        HB_ASSERT( pOutputTensor->getElementType() == pInputTensor->getElementType(),
                "Output tensor and input tensor have a different element type"      );
        HB_ASSERT( pOutputTensor->getDenseSizeInElements() == pInputTensor->getDenseSizeInElements(),
                "Output tensor and input tensor doesn't match in elements' count"   );
    }

    // Getting the input and output buffers
    pInputMap  = static_cast<char*>(pInputTensor->map());
    pOutputMap = static_cast<char*>(pOutputTensor->map());

    memcpy(pOutputMap, pInputMap, pInputTensor->getTotalSizeInBytes());
    return true;
}

bool LogicalRequantNode::isRedundantNode() const
{
    return false;
}
