#include "common_geo_attr.h"
#include "mme_hal_reader.h"
#include "general_utils.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recurring_misalignment_opt.h"
#include "mme_common_utils.h"
#include "include/mme_common/recipe_generator.h"
#include <limits>

namespace MmeCommon
{
GeometryGrid GeometryGrid::idxToGrid(unsigned idx) const
{
    unsigned curIdx = idx;
    GeometryGrid gridIdxs;

    gridIdxs.cd = curIdx % cd;
    curIdx = curIdx / cd;

    gridIdxs.batch = curIdx % batch;
    curIdx = curIdx / batch;

    gridIdxs.fcd = curIdx % fcd;
    curIdx = curIdx / fcd;

    gridIdxs.spatial = curIdx % spatial;
    curIdx = curIdx / spatial;

    MME_ASSERT(curIdx == 0, "Error - received invalid index or grid init failed");
    return gridIdxs;
}

CommonGeoAttr::CommonGeoAttr(const MmeLayerParams& params, const MmeHalReader& mmeHal)
: m_params(params), m_mmeHal(mmeHal)
{
    transA = MmeCommon::isTransposed(params.opType, e_mme_in_a);
    transB = MmeCommon::isTransposed(params.opType, e_mme_in_b);
}

void CommonGeoAttr::resetGrids()
{
    mmeGrid = {};
    coreGrid = {};
    aGrid = {};
    bGrid = {};
    cGrid = {};
}

void CommonGeoAttr::init()
{
    validateParams();
    setGrids();
    setChipConcurrency();
    setSpInterleavingDim();

    if (supportsConcurrency())
        setConcurrentDim();
    else
        setDefaultConcurrentDim();
}

void CommonGeoAttr::validateParams()
{
    // Verify that concurrency enablers are not in undef state
    MME_ASSERT(!supportsConcurrency() || (m_params.strategy.batchConcurrencyEn != Undefined &&
                                          m_params.strategy.cdConcurrencyEn != Undefined),
               "Call to geo attr when concurrency flags are in undef state");
}

bool CommonGeoAttr::supportsConcurrency() const
{
    if (!m_params.isDedwOperation() && !m_params.isGemmOperation()) return false;
    if (m_params.opType == e_mme_reductionAdd || m_params.isGemmDmaOperation())
    {
        return false;   // No concurrency in reductionAdd
    }
    // Reduction is not supported in fp8, therefore cannot apply cd concurrency
    if (m_params.isDedwOperation() && (m_params.strategy.cdConcurrencyEn == TurnedOn) &&
        isTypeFp8(m_params.getOperand(e_mme_op_c).elementType))
    {
        return false;
    }

    return true;
}

MmeDimsIndex CommonGeoAttr::getSpInterleavingDim(EMmeInternalOperand operand) const
{
    if (operand == e_mme_op_a || operand == e_mme_op_b)
    {
        return m_interleavedCdDim;
    }
    return DIM_W;
}

bool CommonGeoAttr::isOperandBroadcasted(EMmeInternalOperand operand, MmeDimsIndex dim) const
{
    // operand B is fully broadcasted in DEDW
    if (m_params.isDedwOperation() && operand == e_mme_op_b) return true;
    //  and operand in BGEMM is broadcasted only if its not a regular GEMM
    if (m_params.isGemmOperation() && m_params.getOperand(operand, m_primaryTensors).sizes[dim] == 1 &&
        m_params.getOperand(e_mme_op_c).sizes[dim] != 1)
        return true;

    return false;
}

bool CommonGeoAttr::isOperandFullyBroadcasted(EMmeInternalOperand operand) const
{
    if (!m_params.isDedwOrGemm())
    {
        // broadcasting exists only in bgemm and dedw
        return false;
    }

    for (unsigned dim = getConcurrentDim(); dim < MME_MAX_TENSOR_DIMS; dim++)
    {
        if (!isOperandBroadcasted(operand, (MmeDimsIndex) dim))
        {
            return false;
        }
    }
    return true;
}

MmeDimsIndex CommonGeoAttr::getConcurrentDim() const
{
    if (!m_concurrentDimOpt.has_value())
        setDefaultConcurrentDim();

    return m_concurrentDimOpt.value();
}

unsigned CommonGeoAttr::getBatchDimsNr() const
{
    // generally we support up to c_batchDimNr batch dimensions
    unsigned maxBatch = c_batchDimNr;
    // the workaround below requires an extra MME loops, reducing the amount of batches we can support by 1.
    if (isMmeConcurrencyRoutingWorkAround() && isTransposed(e_mme_op_a)) maxBatch--;

    return maxBatch;
}

MmeDimsIndex CommonGeoAttr::getLastSpatialDim(EMmeInternalOperand operand) const
{
    switch (m_params.opType)
    {
        default:
            MME_ASSERT(0, "invalid operation");
        case e_mme_gemm_transpose:
        case e_mme_ab:
        case e_mme_atb:
        case e_mme_abt:
        case e_mme_atbt:
        case e_mme_reductionAdd:
            return GEMM_DIM_H;
        case e_mme_fwd:
        case e_mme_dedx:
        case e_mme_transposed_dedx:
        case e_mme_dedw:
        case e_mme_deterministic_dedw:
            switch (m_params.getExternalOperand(operand))
            {
                case e_mme_op_x:
                case e_mme_op_y:
                    return DIM_B;
                case e_mme_op_w:
                    return WEIGHT_DIM_C;
                default:
                    MME_ASSERT(0, "invalid operand");
            }
        case e_mme_memcpy:
        case e_mme_trans:
            return DIM_B;
    }
}

void CommonGeoAttr::setMmeConcurrency()
{
    if (m_params.isGemmOperation())
    {
        setBgemmConcurrency();
    }
    else
    {
        MME_ASSERT(m_params.isDedwOperation(), "unexpected op type");

        // Within the MME:
        // - If only cd concurrency is set - perform CD concurrency
        // - If only batch concurrency is set, perform batch concurrency
        // - If both are set, meaning hybrid mode, perform batch concurrency
        // - If none is set, skip
        if (m_params.strategy.batchConcurrencyEn == TurnedOn)
        {
            setDedwConcurrency();  // batch concurrency
        }
        else if (m_params.strategy.cdConcurrencyEn == TurnedOn)
        {
            setCdConcurrency();  // cd concurrency
        }
    }
}

bool CommonGeoAttr::shouldInterleaveOnSecondSpatialDim()
{
    // If the level of interleaving is unaligned to the second spatial dim, do not use it as it coast extra padding
    unsigned interleavingPorts = getInterleavedSpatialPortsNr(e_mme_op_a);
    unsigned spDim1 = m_params.y.sizes[2];
    if ((spDim1 % interleavingPorts) != 0)
    {
        return false;
    }

    // If the input data type is fp8, do not interleave on the second sp dim
    // todo SW-107828: mme cd concurrency: add support for interleaving on CD dim 1 in fp8
    if (isTypeFp8(m_params.getOperand(e_mme_op_a).elementType))
    {
        return false;
    }

    // Switching to the second CD dim increases the CD alignment requirements because
    // the alignment includes the first dim. If the alignment requirements exceed the SB size
    // do not switch
    if (m_params.strategy.sbReuse)
    {
        unsigned sbSize = SBReuse::calcSingleSBSize(m_mmeHal, m_params, getPortSize(e_mme_op_a));
        unsigned portSize = getPortSize(e_mme_op_a);
        unsigned sbRows = div_round_down(sbSize, portSize);

        unsigned dtAlignment =
            m_mmeHal.getNumElementsForCommonDimAlignment(m_params.getOperand(e_mme_op_a).elementType, m_params.opType);
        unsigned cdDim0Size = m_params.getOperand(e_mme_op_b).sizes[1];
        unsigned alignmentRequirement = dtAlignment * cdDim0Size;

        if (alignmentRequirement > sbRows)
        {
            return false;
        }
    }

    return true;
}

void CommonGeoAttr::setSpInterleavingDim()
{
    // By default, the spatial dim on which we interleave is the first dim.
    // But when this leads to plenty of misaligned accesses, we switch to the second dim.
    if ((m_params.isDedwOperation()) && (getGeometryCdConcurrency() > 1) &&
        (RecurringMisalignmentOptimization::isRecurringMisalignment(m_params, m_mmeHal, e_mme_op_a) ||
         RecurringMisalignmentOptimization::isMultipleAccessToSameCL(m_params, m_mmeHal, e_mme_op_a)) &&
        shouldInterleaveOnSecondSpatialDim())
    {
        // Use the second CD dim
        m_interleavedCdDim = DIM_H;
        return;
    }
    m_interleavedCdDim = DIM_W;
}

void CommonGeoAttr::setChipConcurrency()
{
    //  only bgemm and dedw can be multiplexed
    if (!supportsConcurrency()) return;

    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();
    auto& concurrentField = (m_params.strategy.cdConcurrencyEn == TurnedOn) ? mmeGrid.cd : mmeGrid.batch;

    //  convert unused MMEs on the to work on the next batch
    while ((getGeometryWidth() / 2) >= fcd && mmeGrid.fcd > 1)
    {
        mmeGrid.fcd /= 2;
        concurrentField *= 2;
    }
    while ((getGeometryHeight() / 2) >= sp && mmeGrid.spatial > 1)
    {
        mmeGrid.spatial /= 2;
        concurrentField *= 2;
    }

    // Try intra MME concurrency (if relevant, on top of the inter-MME concurrency)
    if (mmeGrid.spatial == 1 && mmeGrid.fcd == 1)
    {
        setMmeConcurrency();
    }
}

bool CommonGeoAttr::getDoubleAccumsBit() const
{
    if (m_params.strategy.dualGemm) return false;  // not supported in dualGemm mode
    if (m_params.isNativeDmaOperation()) return false;

    // make sure that we do not use only half the EU in case we are limited by interleaving port under utilization
    // as the combination of both can lead to the solution requiring steps which defeats the purpose of this
    // optimization.
    // notice here that we use C operand spatial dim as in conv operations the output defines the size of the dim, as the input can be smaller or larger
    EMmeInternalOperand interleavingOperand = isTransposed(e_mme_op_a) ? e_mme_op_a : e_mme_op_c;
    if (m_params.getOperand(e_mme_op_c).sizes[getSpInterleavingDim(e_mme_op_c)] >=
        getInterleavedSpatialPortsNr(interleavingOperand))
    {
        // return true whenever only half the ACC would be utilized, generally when sp size is less then 64.
        // Also, need to take into account the number of interleaving ports on operand c.
        return div_round_up(m_params.getSpatialSize(), getInterleavedSpatialPortsNr(e_mme_op_c)) <=
               (getAccHeight() / 2);
    }

    return false;
}

unsigned CommonGeoAttr::getCoreSpatialPorts(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return aGrid.spatial;
        case e_mme_op_b:
            return bGrid.spatial;
        case e_mme_op_c:
            return cGrid.spatial;
    }
}

unsigned CommonGeoAttr::getCoreFcdPorts(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return aGrid.fcd;
        case e_mme_op_b:
            return bGrid.fcd;
        case e_mme_op_c:
            return cGrid.fcd;
    }
}

unsigned CommonGeoAttr::getCoreBatchPorts(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return aGrid.batch;
        case e_mme_op_b:
            return bGrid.batch;
        case e_mme_op_c:
            return cGrid.batch;
    }
}

unsigned CommonGeoAttr::getCoreCdPorts(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return aGrid.cd;
        case e_mme_op_b:
            return bGrid.cd;
        case e_mme_op_c:
            return cGrid.cd;
    }
}

GeometryGrid CommonGeoAttr::getEffectiveCoreGrid(EMmeInternalOperand operand) const
{
    // this returns the amount of cores in each direction - starting from 0
    GeometryGrid grid = coreIdxToEffectiveGrid(operand, getCoresPerMmeNr() - 1);
    // regular grid indexing starts at 1 so we need to fix the grid above
    grid.fcd += 1;
    grid.spatial += 1;
    grid.batch += 1;
    return grid;
}

unsigned CommonGeoAttr::getEuFacingPortSize(EMmeInternalOperand operand) const
{
    MME_ASSERT(operand != e_mme_op_c, "expected an input operand");
    // this function returns the amount of contiguous elements that a port will push towards the EU.
    // it will be either the port width (also called portSize) if it is not transpose
    // or the port height (which is represented by teHeight) if it is transpose
    return isTransposed(operand) ? getTeHeight() : getPortSize(operand);
}

unsigned CommonGeoAttr::getMmeSpatialPorts(EMmeInternalOperand operand) const
{
    unsigned effectiveSpatialCoresNr = getEffectiveCoreGrid(operand).spatial;
    return getCoreSpatialPorts(operand) * effectiveSpatialCoresNr;
}

unsigned CommonGeoAttr::getChipSpatialPorts(EMmeInternalOperand operand) const
{
    return getMmeSpatialPorts(operand) * mmeGrid.spatial;
}

unsigned CommonGeoAttr::getMmeFcdPorts(EMmeInternalOperand operand) const
{
    unsigned effectiveFcdCoresNr = getEffectiveCoreGrid(operand).fcd;
    return getCoreFcdPorts(operand) * effectiveFcdCoresNr;
}

unsigned CommonGeoAttr::getChipFcdPorts(EMmeInternalOperand operand) const
{
    return getMmeFcdPorts(operand) * mmeGrid.fcd;
}

unsigned CommonGeoAttr::getMmeBatchPorts(EMmeInternalOperand operand) const
{
    unsigned effectiveBatchCoresNr = getEffectiveCoreGrid(operand).batch;
    return getCoreBatchPorts(operand) * effectiveBatchCoresNr;
}

unsigned CommonGeoAttr::getChipBatchPorts(EMmeInternalOperand operand) const
{
    return getMmeBatchPorts(operand) * mmeGrid.batch;
}

unsigned CommonGeoAttr::getMmePortsNr(EMmeInternalOperand operand) const
{
    return getMmeFcdPorts(operand) * getMmeSpatialPorts(operand) * getMmeBatchPorts(operand);
}

unsigned CommonGeoAttr::getChipPortsNr(EMmeInternalOperand operand) const
{
    return getChipFcdPorts(operand) * getChipSpatialPorts(operand) * getChipBatchPorts(operand);
}

unsigned CommonGeoAttr::getCorePortsNr(EMmeInternalOperand operand) const
{
    return getCoreFcdPorts(operand) * getCoreSpatialPorts(operand) * getCoreBatchPorts(operand);
}

//  return the number of interleaving spatial ports for a given operand within a single MME
unsigned CommonGeoAttr::getMmeInterleavedSpatialPortsNr(EMmeInternalOperand operand) const
{
    if (isSpatiallyInterleavedAcrossCores(operand)) return getMmeSpatialPorts(operand);
    if (isSpatiallyInterleavedInsideCore(operand)) return getCoreSpatialPorts(operand);
    //  no interleaving
    return 1;
}
//  return the number of interleaving spatial ports for a given operand in the entire chip
unsigned CommonGeoAttr::getInterleavedSpatialPortsNr(EMmeInternalOperand operand) const
{
    if (isSpatiallyInterleavedAcrossMmes(operand)) return getMmeSpatialPorts(operand) * getSpatialMmeNr(operand);

    return getMmeInterleavedSpatialPortsNr(operand);
}

unsigned CommonGeoAttr::getGeometryWidth() const
{
    return mmeGrid.fcd * getMmeWidth();
}

unsigned CommonGeoAttr::getGeometryHeight() const
{
    return mmeGrid.spatial * getMmeHeight();
}

unsigned CommonGeoAttr::getCoreConcurrency() const
{
    //  usually both input will have the same amount of batch ports
    //  but in case of broadcast one will be set to 1.
    return std::max(aGrid.batch, bGrid.batch);
}

unsigned CommonGeoAttr::getMmeConcurrency() const
{
    return getCoreConcurrency() * coreGrid.batch;
}

unsigned CommonGeoAttr::getGeometryConcurrency() const
{
    unsigned mmeConcurrency = getMmeConcurrency();
    //  make sure that if we have internal MME concurrency no other MME works on the same batch
    MME_ASSERT_DEBUG_ONLY(
        mmeConcurrency == 1 ||  // No concurrency
            mmeGrid.batch == getMmeNr() ||  // batch concurrency between the mmes
            ((mmeGrid.batch == 1) &&
             (coreGrid.batch == getCoresPerMmeNr())),  // batch concurrency within the mme is over all cores
        "expected concurrency over all MMEs");
    return mmeGrid.batch * mmeConcurrency;
}

unsigned CommonGeoAttr::getCoreCdConcurrency() const
{
    return cGrid.cd;
}

unsigned CommonGeoAttr::getMmeCdConcurrency() const
{
    return coreGrid.cd * getCoreCdConcurrency();
}

unsigned CommonGeoAttr::getGeometryCdConcurrency() const
{
    return mmeGrid.cd * getMmeCdConcurrency();
}

//  return the core location on the core grid according to its index
GeometryGrid CommonGeoAttr::coreIdxToGrid(unsigned coreIdx) const
{
    return coreGrid.idxToGrid(coreIdx);
}

GeometryGrid CommonGeoAttr::coreIdxToEffectiveGrid(EMmeInternalOperand operand, unsigned coreIdx) const
{
    GeometryGrid effectiveCoreGrid = coreIdxToGrid(coreIdx);

    //  translate core(output) movement to input port movement
    if (operand == e_mme_op_a || operand == e_mme_op_b)
    {
        // in some cases while the core advances on one axis, the ports advance on a different one.
        // this will happen in cases where a port from one core provides data to both cores.
        // this behavior changes from chip to chip.
        if (doPortAdvanceSpatially(operand))
        {
            // if operand is transposed or fills entire MME, it has no fcd offset between cores
            effectiveCoreGrid.spatial += effectiveCoreGrid.fcd;
            effectiveCoreGrid.fcd = 0;
        }
        else
        {
            // otherwise operand is non transposed, so the core offset is along the fcd and not the spatial dim
            effectiveCoreGrid.fcd += effectiveCoreGrid.spatial;
            effectiveCoreGrid.spatial = 0;
        }
        //  in CD concurrency the output "movement" translates into the inputs advancing on the CD
        if (coreGrid.cd > 1)
        {
            MME_ASSERT(!isTransposed(operand), "cd Concurrency for transposed input operands is not yet supported");
            effectiveCoreGrid.spatial += effectiveCoreGrid.cd;
            effectiveCoreGrid.cd = 0;
        }
        // when an operand is broadcasted the core batch offset doesnt affect it
        // instead the ports are shared between the two cores and they can continue to advance on the fcd/sp
        // [SW-100486] support A core broadcast
        if (operand == e_mme_op_b && isPortSharedBetweenCores(operand))
        {
            if (isTransposed(operand))
            {
                effectiveCoreGrid.spatial += effectiveCoreGrid.batch;
            }
            else
            {
                effectiveCoreGrid.fcd += effectiveCoreGrid.batch;
            }
            effectiveCoreGrid.batch = 0;
        }
    }

    return effectiveCoreGrid;
}

//  return the MME location on the mme grid according to its index
GeometryGrid CommonGeoAttr::mmeIdxToGrid(unsigned mmeIdx) const
{
    return mmeGrid.idxToGrid(mmeIdx);
}

bool CommonGeoAttr::isTransposed(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return transA;
        case e_mme_op_b:
            return transB;
    }
}

bool CommonGeoAttr::isPortStartOffset(EMmeInternalOperand operand) const
{
    //  check which kind of offset the ports require
    bool portOffsetIsBaseOffset = false;
    bool portOffsetIsStartOffset = false;

    //  port offset has to be startOffset if we are interleaved over multiple spatial dimensions
    //  because the accumulated offset needs to wrap around the different ROI sizes
    //  in this case we want an offset inside the ROI.
    if (getMmeInterleavedSpatialPortsNr(operand) > 1 && !m_params.isGemmOperation())
    {
        portOffsetIsStartOffset = true;
    }

    //  port offset has to be baseOffset for operand A in dedw in case there is concurrency
    //  because each port works on a different weight, and each weight has a different roi base for A.
    //  simply put - in this case we dont want offset inside the ROI but to move the ROI itself.
    if (m_params.isDedwOperation() && getMmeConcurrency() > 1 && operand == e_mme_op_a)
    {
        portOffsetIsBaseOffset = true;
    }

    MME_ASSERT((portOffsetIsBaseOffset && portOffsetIsStartOffset) == 0, "cant set both start and base port offset");
    return portOffsetIsStartOffset;
}

//  is spatially interleaved across multiple MMEs
bool CommonGeoAttr::isSpatiallyInterleavedAcrossMmes(EMmeInternalOperand operand) const
{
    if (m_params.isDmaOperation()) return false;

    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return true;
        case e_mme_op_c:
            return transA;
        case e_mme_op_b:
            return !transB;
    }
}

//  is spatially interleaved across all ports inside an MME
bool CommonGeoAttr::isSpatiallyInterleavedAcrossCores(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
        case e_mme_op_b:
            return isSpatiallyInterleavedInsideCore(operand);
        case e_mme_op_c:
            return (transA && m_params.opType != e_mme_trans) || m_params.opType == e_mme_memcpy;
    }
}

//  is spatially interleaved inside a core
bool CommonGeoAttr::isSpatiallyInterleavedInsideCore(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return m_params.opType != e_mme_trans;
        case e_mme_op_c:
            // In Gaudi2, either spatial c ports are interleaved inside core, or there's only 1 spatial c port per core.
            // We treat the 2nd case as interleaved for consistency with interleaving across cores, so we always return
            // true here. In Gaudi and Gaudi3 we treat each Mme as one core so this condition won't be met.
            if (getCoresPerMmeNr() > 1) return true;

            return (transA && m_params.opType != e_mme_trans) || m_params.opType == e_mme_memcpy;
        case e_mme_op_b:
            return !transB;
    }
}

bool CommonGeoAttr::isPortSharedBetweenCores(EMmeInternalOperand operand) const
{
    //  broadcast + concurrency is not yet implemented for bgemm so there is no sharing between cores in this mode
    if (m_params.isGemmOperation() && getMmeConcurrency() > 1)
    {
        return false;
    }
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return coreGrid.fcd > 1 || (coreGrid.batch > 1 && isOperandFullyBroadcasted(operand));
        case e_mme_op_b:
            return coreGrid.spatial > 1 || (coreGrid.batch > 1 && isOperandFullyBroadcasted(operand));
    }
}

unsigned CommonGeoAttr::getSpatialCoresInMmeNr() const
{
    return coreGrid.spatial;
}

unsigned CommonGeoAttr::getSpatialMmeNr(EMmeInternalOperand operand) const
{
    // C behaves like the mme grid while A & B need to translate the mme movement to their movement
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return isTransposed(operand) ? mmeGrid.spatial : mmeGrid.cd;
        case e_mme_op_b:
            return isTransposed(operand) ? mmeGrid.fcd : mmeGrid.cd;
        case e_mme_op_c:
            return mmeGrid.spatial;
    }
}

unsigned CommonGeoAttr::getFcdMmeNr(EMmeInternalOperand operand) const
{
    // C behaves like the mme grid while A & B need to translate the mme movement to their movement
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid mme internal operand");
        case e_mme_op_a:
            return isTransposed(operand) ? 1 : mmeGrid.fcd;
        case e_mme_op_b:
            return isTransposed(operand) ? 1 : mmeGrid.fcd;
        case e_mme_op_c:
            return mmeGrid.fcd;
    }
}

const CommonGeoAttr::EMmeInternalOperandVec& CommonGeoAttr::getOperands() const
{
    static const EMmeInternalOperandVec DMA_OPERANDS = {e_mme_op_a, e_mme_op_c};
    static const EMmeInternalOperandVec COMPUTE_OPERANDS = {e_mme_op_a, e_mme_op_b, e_mme_op_c};

    if (m_params.isNativeDmaOperation())
    {
        return DMA_OPERANDS;
    }
    else
    {
        return COMPUTE_OPERANDS;
    }
}

bool CommonGeoAttr::isDimConcurrencyOptimizationSupported() const
{
    return true;
}

void CommonGeoAttr::setConcurrentDim()
{
    if (m_concurrentDimOpt.has_value())
        return;

    m_concurrentDimOpt = m_params.isGemmOperation() ? calculateConcurrentDimForGemm() :
        calculateConcurrentDimForNonGemm();
}

void CommonGeoAttr::setDefaultConcurrentDim() const
{
    if (m_concurrentDimOpt.has_value())
        return;

    m_concurrentDimOpt = m_params.isGemmOperation() ? calculateDefaultConcurrentDimForGemm() : calculateConcurrentDimForNonGemm();
}

MmeDimsIndex CommonGeoAttr::calculateDefaultConcurrentDimForGemm() const
{
    return m_params.canFlatten() ? GEMM_DIM_B2 : GEMM_DIM_B1;
}

MmeDimsIndex CommonGeoAttr::calculateConcurrentDimForGemm() const
{
    const unsigned geometryConcurrency = getGeometryConcurrency();
    const MmeDimsIndex firstBatchConcurDim = m_params.canFlatten() ? GEMM_DIM_B2 : GEMM_DIM_B1;
    if (!isDimConcurrencyOptimizationSupported())
        return firstBatchConcurDim;

    const unsigned lastBatchConcurDim = getBatchDimsNr() + MmeCommon::GEMM_DIM_B1 - 1;
    MmeDimsIndex bestBatchConcurDim = firstBatchConcurDim;
    unsigned bestTotalBatchSteps = std::numeric_limits<unsigned int>::max();
    for (unsigned batchDim = firstBatchConcurDim; batchDim <= lastBatchConcurDim; ++batchDim)
    {
        unsigned currentTotalBatchSteps = div_round_up(m_params.y.sizes[batchDim], geometryConcurrency);
        for (unsigned b = firstBatchConcurDim; b <= lastBatchConcurDim; ++b)
        {
            if (b != batchDim)
                currentTotalBatchSteps *= m_params.y.sizes[b];
        }

        if (currentTotalBatchSteps < bestTotalBatchSteps)
        {
            bestBatchConcurDim = static_cast<MmeDimsIndex>(batchDim);
            bestTotalBatchSteps = currentTotalBatchSteps;
        }
    }

    return bestBatchConcurDim;
}

MmeDimsIndex CommonGeoAttr::calculateConcurrentDimForNonGemm() const
{
    return m_params.isDmaOperation() ? GEMM_DIM_B3 :
            (m_params.canLower() ? DIM_R : DIM_S);
}

}  // namespace MmeCommon
