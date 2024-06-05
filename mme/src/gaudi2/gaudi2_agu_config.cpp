#include "gaudi2_agu_config.h"
#include "gaudi2/mme.h"
#include "include/mme_common/mme_common_enum.h"

using namespace MmeCommon;

namespace Gaudi2
{
//  consider moving this function to common class
void Gaudi2AguConfig::configureDescriptor(void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;

    for (auto operand : m_geoAttr.getOperands())
    {
        configTensor(desc, operand);
        configPorts(desc, operand);
        configTensorParams(desc, operand);
    }
    configureRouting(desc);
    applyWorkarounds(desc);
}

void Gaudi2AguConfig::applyWorkarounds(Mme::Desc* desc)
{
    if (m_geoAttr.getHx2Bit() && m_geoAttr.getMmePortsNr(e_mme_op_a) == 4)
    {
        setFakeSpatialLoop(desc);
    }

    if (m_geoAttr.isMmeConcurrencyRoutingWorkAround())
    {
        addVirtualDim(desc);
    }
}

void Gaudi2AguConfig::setFakeSpatialLoop(Mme::Desc* desc)
{
    // in the specific case of Gaudi2 4x concurrency with steps, only half the EU height is utilized
    // that is because we turn on the 2xH bit, but we dont use the two upper A ports.
    // in this case when there is more than one step the MME will mistakenly think that that whole EU contains valid
    // data, to avoid it signal to the MME that every loop is the last spatial iteration, and configure the last spatial
    // step as half EU height. this will make sure that every step we read only 128 rows and never try to read the
    // invalid half of the EU in DEDW and BGEMM the spatial loop is never used to we will use it as the victim loop
    const auto& spView = m_recipe.curSp();
    const auto& convView = m_recipe.curNonSpatial();
    unsigned aSpatialSize = m_params.isDedwOperation() ? convView.sizes[1] : spView.viewSize;
    unsigned spatialStepsNr = div_round_up(aSpatialSize, m_geoAttr.getGeometryHeight());
    // this check is not needed as we can use this workaround even if we dont have any steps, but as a caution we will
    // use it only when necessary
    if (spatialStepsNr > 1)
    {
        desc->header.partialHeightLoopA = e_mme_tetris_loop;
        desc->spatialSizeMinus1Cout = 63; // half of original EU height
    }
}

void Gaudi2AguConfig::addVirtualDim(Mme::Desc* desc)
{
    // in this optimization the EU rows will be interleaved with rows with valid data and rows with junk data.
    // to handle that we will add a new loop sized 2, and valid element sized 1.
    // this will allow the MME to ignore all the invalid rows.
    // after two steps the loop finishes its work and we will have a regular spatial loop step.
    // first we are going to push all tensor dims one dimension above.
    // associated dims are treated here as they are already treated for this workaround in the common AGU config

    const TensorAttr& tensor = getTensor(e_mme_op_c);
    Mme::MmeTensorDesc& tensorDesc = getDescTensor(desc, e_mme_op_c);
    // if A is transposed we are forced to interleave the ports and introduce a virtual dimension, otherwise work in
    // blocks
    if (m_geoAttr.isTransposed(e_mme_op_a))
    {
        // push all dimensions above the FCD upwards one dim to introduce the new virtual dimension
        for (int dim = MAX_DIMENSION - 1; dim > GEMM_DIM_H; dim--)
        {
            if (dim < MAX_DIMENSION - 1)
            {
                tensorDesc.roiSize[dim] = tensorDesc.roiSize[dim - 1];
            }
            tensorDesc.validElements[dim] = tensorDesc.validElements[dim - 1];
            tensorDesc.loopStride[dim] = tensorDesc.loopStride[dim - 1];
            tensorDesc.spatialStrides[dim - 1] = tensorDesc.spatialStrides[dim - 2];
            tensorDesc.startOffset[dim - 1] = tensorDesc.startOffset[dim - 2];

            for (int i = 0; i < Mme::c_mme_wb_nr; i++)
            {
                for (int core = 0; core < Mme::MME_CORE_PAIR_SIZE; core++)
                {
                    desc->aguOut[i][core].roiBaseOffset[dim] = desc->aguOut[i][core].roiBaseOffset[dim - 1];
                }
            }
        }

        // after we pushed all the dimensions upwards we can introduce the new virtual dim.
        // this dim size will be 2, with only a single valid element.
        // this way only every other row will be written to memory, while the invalid rows will be discarded
        tensorDesc.roiSize[GEMM_DIM_H] = 2;
        tensorDesc.spatialStrides[GEMM_DIM_H - 1] = 1;
        tensorDesc.startOffset[GEMM_DIM_H - 1] = 0;
        tensorDesc.validElements[GEMM_DIM_H] = 1;
        tensorDesc.loopStride[GEMM_DIM_H] = 0;  // irrelevant
        // the slave gets a negative offset because we want it ot start outside the ROI as the first row is invalid
        // and only the second row (and every other even row) has valid data, unlike the master which only the first
        // (and any other odd row) is valid
        for (int i = 0; i < Mme::c_mme_wb_nr; i++)
        {
            desc->aguOut[i][Mme::MME_CORE_SLAVE].roiBaseOffset[GEMM_DIM_H] = -1;
        }

        // since we have twice as much data in the EU (half of it is invalid), each output port has to do twice the
        // amount of spatial steps.
        desc->spatialSizeMinus1Cout = ((desc->spatialSizeMinus1Cout + 1) * 2) - 1;
    }
    else
    {
        tensorDesc.roiSize[GEMM_DIM_H] = m_geoAttr.getEuHeight() * tensor.roiSize[GEMM_DIM_W];
        desc->spatialSizeMinus1Cout = m_geoAttr.getEuHeight() - 1;

        for (int i = 0; i < Mme::c_mme_wb_nr; i++)
        {
            for (int core = 0; core < Mme::MME_CORE_PAIR_SIZE; core++)
            {
                desc->aguOut[i][core].roiBaseOffset[GEMM_DIM_H] =
                    -1 * m_geoAttr.getTeHeight() * core * tensorDesc.roiSize[GEMM_DIM_W];
            }
        }
    }
}

const Gaudi2AguConfig::SbIndicesVec& Gaudi2AguConfig::getSbIndices(EMmeInternalOperand operand)
{
    // Get the indices of the relevant SBs
    static const SbIndicesVec SB_INDICES_VEC_OP_C = {0, 1};
    static const SbIndicesVec SB_INDICES_VEC_OP_A_NARROW = {0};
    static const SbIndicesVec SB_INDICES_VEC_OP_A_SYMMETRICAL = {0, 4};
    static const SbIndicesVec SB_INDICES_VEC_OP_B_SYMMETRICAL = {2, 3};
    static const SbIndicesVec SB_INDICES_VEC_OP_A_WIDE = {2, 4, 3, 1};
    static const SbIndicesVec SB_INDICES_VEC_OP_B_WIDE = {2, 3, 4, 1};

    if (operand == e_mme_op_c)
    {
        return SB_INDICES_VEC_OP_C;
    }

    unsigned portsNr = m_geoAttr.getCorePortsNr(operand);
    switch (portsNr)
    {
        case 1:
            return SB_INDICES_VEC_OP_A_NARROW;
        case 2:
            return (operand == e_mme_op_a) ? SB_INDICES_VEC_OP_A_SYMMETRICAL : SB_INDICES_VEC_OP_B_SYMMETRICAL;
        case 4:
            return (operand == e_mme_op_a) ? SB_INDICES_VEC_OP_A_WIDE : SB_INDICES_VEC_OP_B_WIDE;
        default:
            MME_ASSERT(0, "invalid number of ports");
            return SB_INDICES_VEC_OP_A_NARROW;  // never reached so we can return anything
    }
}

//  set port offsets according to the portComplex struct
void Gaudi2AguConfig::configPorts(Mme::Desc* desc, EMmeInternalOperand operand)
{
    PortsComplex& logicPorts = getPortComplex(operand);
    Mme::MmeAguCoreDesc* aguDesc[2][4];
    const SbIndicesVec& sbIndices = getSbIndices(operand);

    bool isInput = (operand != e_mme_op_c);
    if (isInput)
    {
        // set which AGU the operand reads from
        setAguReads(desc, operand, sbIndices);
    }

    // For each core, create a list of pointers to the relevant AGUs in the desc, in the order they should be
    // configured in.
    for (unsigned coreIdx = 0; coreIdx < Mme::MME_CORE_PAIR_SIZE; coreIdx++)
    {
        unsigned portIdx = 0;
        for (auto sbIdx : sbIndices)
        {
            aguDesc[coreIdx][portIdx] = &(isInput ? desc->aguIn : desc->aguOut)[sbIdx][coreIdx];
            portIdx++;
        }
    }

    // Configure desc AGUs from previously configured logicPorts. The ports are arranged in a 3d matrix (batch,fcd,sp).
    // AGUs are in a simple list but correspond to the same logic coordinates.
    // TODO: consider using multi dim iterator or somehow simplify this loop
    unsigned coresPerMme = m_geoAttr.getCoresPerMmeNr();
    unsigned fcdPorts = m_geoAttr.getCoreFcdPorts(operand);
    unsigned spatialPorts = m_geoAttr.getCoreSpatialPorts(operand);
    unsigned batchPorts = m_geoAttr.getCoreBatchPorts(operand);
    unsigned cdPorts = m_geoAttr.getCoreCdPorts(operand);
    for (unsigned core = 0; core < coresPerMme; core++)
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

                        bool swapCores = m_geoAttr.shouldSwapMasterAndSlave(operand);
                        unsigned actCore = swapCores ? coresPerMme - 1 - core : core;

                        memcpy(aguDesc[actCore][portIdx]->roiBaseOffset,
                               curPort.portOffset,
                               Mme::c_mme_max_tensor_dims * sizeof(unsigned));
                        portIdx++;
                    }
                }
            }
        }
    }
}

Mme::MmeTensorDesc& Gaudi2AguConfig::getDescTensor(Mme::Desc* desc, EMmeInternalOperand operand)
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
void Gaudi2AguConfig::configTensor(Mme::Desc* desc, EMmeInternalOperand operand)
{
    const TensorAttr& tensor = getTensor(operand);
    Mme::MmeTensorDesc& tensorDesc = getDescTensor(desc, operand);
    const bool isPortStartOffset = m_geoAttr.isPortStartOffset(operand);

    for (int dim = 0; dim < MAX_DIMENSION; dim++)
    {
        if (dim < MAX_DIMENSION - 1)
        {
            // descriptor doesnt hold a field for the last RoiSize because its not necessary
            tensorDesc.roiSize[dim] = tensor.roiSize[dim];
            // field used for spatial configuration doesnt hold a value for FCD dim.
            tensorDesc.spatialStrides[dim] = tensor.spatialStrides[dim + 1];
            if (isPortStartOffset)
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

//  configure periphery attribute to the tensor, mainly header bits.
void Gaudi2AguConfig::configTensorParams(Mme::Desc* desc, EMmeInternalOperand operand)
{
    const TensorAttr& tensor = getTensor(operand);

    switch (operand)
    {
        case e_mme_op_a:
            desc->spatialSizeMinus1A = tensor.lastSpatialStep - 1;
            desc->tensorA.roiSize[0] = tensor.lastFcdStep;
            desc->header.transA = m_geoAttr.isTransposed(operand);
            desc->header.advanceA = m_params.isFwdOrDedx();
            desc->header.swapBaseAndOffsetA = m_geoAttr.isPortStartOffset(operand);
            desc->header.teBypassA = 0;

            // if A is transposed there could be move A ports then C ports, set how to shuffle them
            if (m_geoAttr.isTransposed(operand))
            {
                Mme::EMmeShuffleAMode shuffleMode;
                switch (m_geoAttr.getCoreSpatialPorts(operand))
                {
                    default:
                        MME_ASSERT(0, "unexpected number of A ports");
                    case 1:
                        //  in 4xw geometry both cores share their two port, so each core sees 2 port that needs to be
                        //  shuffled, unless we are using the MME concurrency routing workaround, in this case the
                        //  second ports rows contain irrelevant data so we dont want to shuffle them, we want to keep
                        //  them in blocks so its easier to ignore them.
                        if (m_geoAttr.isPortSharedBetweenCores(operand)) shuffleMode = Mme::e_mme_shuffle_2ports;
                        else
                            shuffleMode = Mme::e_mme_shuffle_none;
                        break;
                    case 2:
                        shuffleMode = Mme::e_mme_shuffle_2ports;
                        break;
                    case 4:
                        shuffleMode = Mme::e_mme_shuffle_4ports;
                        break;
                }
                desc->brains.shuffleA = shuffleMode;
            }
            else
            {
                desc->brains.shuffleA = Mme::e_mme_shuffle_none;
            }
            break;
        case e_mme_op_b:
            desc->spatialSizeMinus1B = tensor.lastSpatialStep - 1;
            desc->tensorB.roiSize[0] = tensor.lastFcdStep;
            desc->header.transB = m_geoAttr.isTransposed(operand);
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

void Gaudi2AguConfig::setAssociatedDimAndSize(EMmeLoopMask mask,
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

void Gaudi2AguConfig::setSpatialLoopSize(unsigned size, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->numIterationsMinus1 = size - 1;
}

void Gaudi2AguConfig::setPartialHeightLoopMaskA(unsigned mask, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->header.partialHeightLoopA = mask;
}
void Gaudi2AguConfig::setPartialHeightLoopMaskB(unsigned mask, void* descPtr)
{
    Mme::Desc* desc = (Mme::Desc*) descPtr;
    desc->header.partialHeightLoopB = mask;
}

void Gaudi2AguConfig::setAguReads(Mme::Desc* desc,
                                  EMmeInternalOperand operand,
                                  const Gaudi2AguConfig::SbIndicesVec& sbIndices)
{
    // each one of the 5 bits in the mask represents an SB, the bit is on if the operand reads from this SB.
    uint8_t mask = 0b00000;
    for (auto sbIdx : sbIndices)
    {
        mask |= 1 << sbIdx;
    }

    if (operand == e_mme_op_a)
    {
        desc->header.aguReadsA = mask;
    }
    else
    {
        desc->header.aguReadsB = mask;
    }
}

enum aguCtrl
{
    e_mme_sb_sel_a0 = 0b000,
    e_mme_sb_sel_a1 = 0b001,
    e_mme_sb_sel_a2 = 0b010,
    e_mme_sb_sel_a3 = 0b011,
    e_mme_sb_sel_b0 = 0b100,
    e_mme_sb_sel_b1 = 0b101,
    e_mme_sb_sel_b2 = 0b110,
    e_mme_sb_sel_b3 = 0b111,
};

void Gaudi2AguConfig::configureRouting(Mme::Desc* desc)
{
    // find out which SBs are used
    const Gaudi2AguConfig::SbIndicesVec& sbIndicesA = getSbIndices(MmeCommon::e_mme_op_a);
    const Gaudi2AguConfig::SbIndicesVec& sbIndicesB = getSbIndices(MmeCommon::e_mme_op_b);
    std::array<bool, 5> sbIndiceUsed = {false};
    for (unsigned sbIndice : sbIndicesA)
    {
        sbIndiceUsed[sbIndice] = true;
    }
    for (unsigned sbIndice : sbIndicesB)
    {
        sbIndiceUsed[sbIndice] = true;
    }
    // find out which SBs are shared between cores
    bool aShared = m_geoAttr.isPortSharedBetweenCores(e_mme_op_a);
    bool bShared = m_geoAttr.isPortSharedBetweenCores(e_mme_op_b);
    MME_ASSERT((aShared && bShared) == 0, "cant share both input ports");
    EMmeInternalOperand sharedOperand = e_mme_op_nr;
    if (aShared) sharedOperand = e_mme_op_a;
    if (bShared) sharedOperand = e_mme_op_b;
    for (auto& type : {Mme::MME_CORE_MASTER, Mme::MME_CORE_SLAVE})
    {
        desc->ctrl.eus[type].sb0En = sbIndiceUsed[0];
        desc->ctrl.eus[type].sb1En = sbIndiceUsed[1];
        desc->ctrl.eus[type].sb2En = sbIndiceUsed[2];
        desc->ctrl.eus[type].sb3En = sbIndiceUsed[3];
        desc->ctrl.eus[type].sb4En = sbIndiceUsed[4];

        if (sharedOperand != e_mme_op_nr)
        {
            unsigned portsNr = m_geoAttr.getCorePortsNr(sharedOperand);
            if (portsNr == 1)
            {
                // configure sharing one port
                desc->ctrl.eus[type].in0En = 1;
                desc->ctrl.eus[type].in1En = 0;
                desc->ctrl.eus[type].sb0OutEn = 1;
                desc->ctrl.eus[type].sb2OutEn = 0;
                desc->ctrl.eus[type].sb3OutEn = 0;
            }
            else
            {
                // configure sharing two ports
                MME_ASSERT(portsNr == 2, "expected sharing only 2 ports");
                desc->ctrl.eus[type].in0En = 1;
                desc->ctrl.eus[type].in1En = 1;
                desc->ctrl.eus[type].sb0OutEn = 0;
                desc->ctrl.eus[type].sb2OutEn = 1;
                desc->ctrl.eus[type].sb3OutEn = 1;
            }
        }
        else
        {
            // configure no port sharing
            desc->ctrl.eus[type].in0En = 0;
            desc->ctrl.eus[type].in1En = 0;
            desc->ctrl.eus[type].sb0OutEn = 0;
            desc->ctrl.eus[type].sb2OutEn = 0;
            desc->ctrl.eus[type].sb3OutEn = 0;
        }

        // configure the routing itself - which SB is connected to which input of the EU.
        // this logic is hard coded according to the H6 MME spec
        switch (m_geoAttr.getCorePortsNr(e_mme_op_a))
        // SBs 0&1 share an input port, for that reason either only one of them is used
        // or they are routed to the last EU input (a3/b3) to reduce their usage as much as possible
        {
            case 4:
                desc->ctrl.eus[type].sb1Sel = e_mme_sb_sel_a3;
                desc->ctrl.eus[type].sb2Sel = e_mme_sb_sel_a0;
                desc->ctrl.eus[type].sb3Sel = e_mme_sb_sel_a2;
                desc->ctrl.eus[type].sb4Sel = e_mme_sb_sel_a1;
                break;
            case 2:
                desc->ctrl.eus[type].sb0Sel = e_mme_sb_sel_a0;
                desc->ctrl.eus[type].sb4Sel = e_mme_sb_sel_a1;
                break;
            case 1:
                if (aShared)
                {
                    desc->ctrl.eus[type].sb0Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_a0 : e_mme_sb_sel_a1;
                    desc->ctrl.eus[type].in0Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_a1 : e_mme_sb_sel_a0;
                }
                else
                {
                    desc->ctrl.eus[type].sb0Sel = e_mme_sb_sel_a0;
                }
                break;
        }
        switch (m_geoAttr.getCorePortsNr(e_mme_op_b))
        {
            case 4:
                desc->ctrl.eus[type].sb1Sel = e_mme_sb_sel_b3;
                desc->ctrl.eus[type].sb2Sel = e_mme_sb_sel_b0;
                desc->ctrl.eus[type].sb3Sel = e_mme_sb_sel_b1;
                desc->ctrl.eus[type].sb4Sel = e_mme_sb_sel_b2;
                break;
            case 2:
                if (bShared)
                {
                    desc->ctrl.eus[type].sb2Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_b0 : e_mme_sb_sel_b2;
                    desc->ctrl.eus[type].sb3Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_b1 : e_mme_sb_sel_b3;
                    desc->ctrl.eus[type].in0Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_b2 : e_mme_sb_sel_b0;
                    desc->ctrl.eus[type].in1Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_b3 : e_mme_sb_sel_b1;
                }
                else
                {
                    desc->ctrl.eus[type].sb2Sel = e_mme_sb_sel_b0;
                    desc->ctrl.eus[type].sb3Sel = e_mme_sb_sel_b1;
                }
                break;
            case 1:
                if (bShared)
                {
                    desc->ctrl.eus[type].sb0Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_b0 : e_mme_sb_sel_b1;
                    desc->ctrl.eus[type].in0Sel = type == Mme::MME_CORE_MASTER ? e_mme_sb_sel_b1 : e_mme_sb_sel_b0;
                }
                else
                {
                    desc->ctrl.eus[type].sb0Sel = e_mme_sb_sel_b0;
                }
                break;
        }
    }
}
}  // namespace Gaudi2