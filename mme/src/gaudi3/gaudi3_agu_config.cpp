#include "gaudi3_agu_config.h"
#include "gaudi3/mme.h"
#include "include/mme_common/mme_common_enum.h"

using namespace MmeCommon;

namespace gaudi3
{
//  consider moving this function to common class
void Gaudi3AguConfig::configureDescriptor(void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;

    for (auto operand : m_geoAttr.getOperands())
    {
        configTensor(desc, operand);
        configPorts(desc, operand);
        configTensorParams(desc, operand);
    }
}

//  set port offsets according to the portComplex struct
void Gaudi3AguConfig::configPorts(Mme::Desc* desc, EMmeInternalOperand operand)
{
    PortsComplex& logicPorts = getPortComplex(operand);
    Mme::MmeAguCoreDesc* aguDesc[4];

    switch (operand)
    {
        case e_mme_op_a:
            if (m_params.isNativeDmaOperation())
            {
                aguDesc[0] = &desc->aguIn[Mme::MME_MASTER][0];
                aguDesc[1] = &desc->aguIn[Mme::MME_SLAVE][0];
            }
            else
            {
                if (m_geoAttr.isTransposed(e_mme_op_a) && m_geoAttr.getMmeConcurrency() == 1)
                {
                    //  ports are interleaved between cores
                    aguDesc[0] = &desc->aguIn[Mme::MME_MASTER][0];
                    aguDesc[1] = &desc->aguIn[Mme::MME_SLAVE][0];
                    aguDesc[2] = &desc->aguIn[Mme::MME_MASTER][1];
                    aguDesc[3] = &desc->aguIn[Mme::MME_SLAVE][1];
                }
                else
                {
                    aguDesc[0] = &desc->aguIn[Mme::MME_MASTER][0];
                    aguDesc[1] = &desc->aguIn[Mme::MME_MASTER][1];
                    aguDesc[2] = &desc->aguIn[Mme::MME_SLAVE][0];
                    aguDesc[3] = &desc->aguIn[Mme::MME_SLAVE][1];
                }
            }
            break;
        case e_mme_op_b:
            aguDesc[0] = &desc->aguIn[Mme::MME_MASTER][2];
            aguDesc[1] = &desc->aguIn[Mme::MME_MASTER][3];
            aguDesc[2] = &desc->aguIn[Mme::MME_SLAVE][2];
            aguDesc[3] = &desc->aguIn[Mme::MME_SLAVE][3];
            break;
        case e_mme_op_c:
            aguDesc[0] = &desc->aguOut[Mme::MME_MASTER];
            aguDesc[1] = &desc->aguOut[Mme::MME_SLAVE];
            break;
        default:
            MME_ASSERT(0, "undefined operand");
    }

    //  consider using multi dim iterator or somehow simplify this loop
    unsigned coresPerMme = m_geoAttr.getCoresPerMmeNr();
    unsigned fcdPorts = m_geoAttr.getCoreFcdPorts(operand);
    unsigned spatialPorts = m_geoAttr.getCoreSpatialPorts(operand);
    unsigned batchPorts = m_geoAttr.getCoreBatchPorts(operand);
    unsigned cdPorts = m_geoAttr.getCoreCdPorts(operand);
    for (unsigned core = 0; core < coresPerMme; core++)  // gaudi3 mme isn't split into cores, this is for compatibility
    {
        unsigned portIdx = 0;
        for (unsigned cd = 0; cd < cdPorts; cd++)
        {
            for (unsigned batch = 0; batch < batchPorts; batch++)
            {
                for (unsigned fcd = 0; fcd < fcdPorts; fcd++)
                {
                    for (unsigned sp = 0; sp < spatialPorts; sp++)
                    {
                        PortAttr& curPort = logicPorts.at(core).at(cd).at(batch).at(fcd).at(sp);
                        memcpy(aguDesc[portIdx]->roiBaseOffset,
                               curPort.portOffset,
                               Mme::c_mme_max_tensor_dims * sizeof(unsigned));
                        portIdx++;
                    }
                }
            }
        }
    }
}

Mme::MmeTensorDesc& Gaudi3AguConfig::getDescTensor(Mme::Desc* desc, EMmeInternalOperand operand)
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return desc->tensorA;
        case e_mme_op_b:
            return desc->tensorB;
        case e_mme_op_c:
            return desc->tensorCOut;
    }
}

//  set a single tensor descriptor using the TensorAttr struct
void Gaudi3AguConfig::configTensor(Mme::Desc* desc, EMmeInternalOperand operand)
{
    const TensorAttr& tensor = getTensor(operand);
    Mme::MmeTensorDesc& tensorDesc = getDescTensor(desc, operand);

    if (m_params.strategy.dualGemm)
    {
        for (int gemm = 0; gemm < Mme::MME_PAIR_SIZE; gemm++)
        {
            for (int dim = 0; dim < 2; dim++)
            {
                tensorDesc.dualGemm.roiSize[gemm][dim] = tensor.roiSize[dim];
                tensorDesc.dualGemm.validElements[gemm][dim] = tensor.validElements[dim];
                if (m_geoAttr.isPortStartOffset(operand))
                {
                    tensorDesc.dualGemm.startOffset[gemm][dim] = tensor.baseOffset[dim];
                }
                else
                {
                    tensorDesc.dualGemm.startOffset[gemm][dim] = tensor.startOffset[dim];
                }

                if (dim > 0)
                {
                    tensorDesc.dualGemm.spatialStrides[gemm] = tensor.spatialStrides[dim];
                }
            }
        }
    }
    else
    {
        for (int dim = 0; dim < MAX_DIMENSION; dim++)
        {
            if (dim < MAX_DIMENSION - 1)
            {
                // descriptor doesnt hold a field for the last RoiSize because its not necessary
                tensorDesc.roiSize[dim] = tensor.roiSize[dim];
                // field used for spatial configuration doesnt hold a value for FCD dim.
                tensorDesc.spatialStrides[dim] = tensor.spatialStrides[dim + 1];
                if (m_geoAttr.isPortStartOffset(operand))
                {
                    tensorDesc.startOffset[dim] = tensor.baseOffset[dim + 1];
                }
                else
                {
                    tensorDesc.startOffset[dim] = tensor.startOffset[dim + 1];
                }
            }
            tensorDesc.validElements[dim] = tensor.validElements[dim];
            tensorDesc.loopStride[dim] = tensor.loopStride[dim];
        }
    }
}

//  configure periphery attribute to the tensor, mainly header bits.
void Gaudi3AguConfig::configTensorParams(Mme::Desc* desc, EMmeInternalOperand operand)
{
    const TensorAttr& tensor = getTensor(operand);

    switch (operand)
    {
        case e_mme_op_a:
            desc->spatialSizeMinus1A = tensor.lastSpatialStep - 1;
            desc->tensorA.roiSize[0] = tensor.lastFcdStep;
            desc->header.transA = m_geoAttr.isTransposed(operand);
            desc->header.sbTransA = m_geoAttr.isTransposed(operand);
            desc->header.advanceA = m_params.isFwdOrDedx();
            //  shuffling is required when ports are interleaved
            desc->header.shuffleA =
                m_geoAttr.isTransposed(operand) ? Mme::e_mme_shuffle_2ports : Mme::e_mme_shuffle_none;
            desc->header.swapBaseAndOffsetA = m_geoAttr.isPortStartOffset(operand);
            desc->header.teBypassA = 0;
            if (m_params.opType == MmeCommon::e_mme_memcpy)
            {
                // configure numStepRight
                unsigned numStepRight =
                    div_round_up((uint32_t)desc->tensorA.roiSize[0] * getElementSize(m_params.getOperand(e_mme_op_a).elementType),
                                 Mme::c_cl_size);
                desc->spatialSizeMinus1B = (tensor.lastSpatialStep * numStepRight) - 1;
            }
            break;
        case e_mme_op_b:
            desc->spatialSizeMinus1B = tensor.lastSpatialStep - 1;
            desc->tensorB.roiSize[0] = tensor.lastFcdStep;
            desc->header.transB = m_geoAttr.isTransposed(operand);
            desc->header.sbTransB = m_geoAttr.isTransposed(operand);
            desc->header.swapBaseAndOffsetB = m_geoAttr.isPortStartOffset(operand);
            desc->header.advanceB = 0;
            desc->header.teBypassB = 0;
            break;
        case e_mme_op_c:
            desc->spatialSizeMinus1Cout = tensor.lastSpatialStep - 1;
            desc->tensorCOut.roiSize[0] = tensor.lastFcdStep;
            desc->header.advanceC = m_params.isFwdOrDedx();
            desc->header.swapBaseAndOffsetOut = m_geoAttr.isPortStartOffset(operand);
            break;
        default:
            MME_ASSERT(0, "invalid mme internal operand");
            return;
    }
}

void Gaudi3AguConfig::setAssociatedDimAndSize(EMmeLoopMask mask,
                                              unsigned size,
                                              unsigned dimA,
                                              unsigned dimB,
                                              unsigned dimOut,
                                              void* descPtr)
{
    MME_ASSERT(size - 1 <= std::numeric_limits<uint8_t>::max(), "");
    auto* desc = static_cast<Mme::Desc*>(descPtr);

    Mme::MmeAssociatedDims* assocDim {};
    switch (mask)
    {
        case e_mme_conv_loop_0:
            desc->conv.kernelSizeMinus1.dim[0] = size - 1;
            assocDim = &desc->conv.associatedDims[0];
            break;
        case e_mme_conv_loop_1:
            desc->conv.kernelSizeMinus1.dim[1] = size - 1;
            assocDim = &desc->conv.associatedDims[1];
            break;
        case e_mme_conv_loop_2:
            desc->conv.kernelSizeMinus1.dim[2] = size - 1;
            assocDim = &desc->conv.associatedDims[2];
            break;
        case e_mme_conv_loop_3:
            desc->conv.kernelSizeMinus1.dim[3] = size - 1;
            assocDim = &desc->conv.associatedDims[3];
            break;
        case e_mme_outer_loop:
            desc->outerLoop.sizeMinus1 = size - 1;
            assocDim = &desc->outerLoop.associatedDims;
            break;
        default:
            MME_ASSERT(0, "unsupported EMmeLoopMask");
    }
    assocDim->dimA = dimA;
    assocDim->dimB = dimB;
    assocDim->dimOut = dimOut;
    assocDim->reserved = 0;
}

void Gaudi3AguConfig::setSpatialLoopSize(unsigned size, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->numIterationsMinus1 = size - 1;
}

void Gaudi3AguConfig::setPartialHeightLoopMaskA(unsigned mask, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->header.partialHeightLoopA = mask;
}
void Gaudi3AguConfig::setPartialHeightLoopMaskB(unsigned mask, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->header.partialHeightLoopB = mask;
}
}  // namespace gaudi3