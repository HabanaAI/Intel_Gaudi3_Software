#include "gaudi2_geo_attr.h"
#include "gaudi2/mme.h"
#include "general_utils.h"
#include "include/mme_common/mme_common_enum.h"

using namespace MmeCommon;

namespace Gaudi2
{
void Gaudi2GeoAttr::setGrids()
{
    resetGrids();
    m_asymPortConfigForNoPortConstraint = false;
    unsigned mmeLimit = m_params.strategy.mmeLimit;
    switch (m_params.getGeometry())
    {
        default:
            MME_ASSERT(0, "Geometry not supported");
        case e_mme_geometry_4xw:
            mmeGrid.fcd = mmeLimit;
            mmeGrid.spatial = 1;
            set2xwPorts();
            break;
        case e_mme_geometry_2xw:
            mmeGrid.fcd = mmeLimit;
            mmeGrid.spatial = 1;
            setSymPorts();
            break;
        case e_mme_geometry_2xh:
            mmeGrid.fcd = 1;
            mmeGrid.spatial = mmeLimit;
            setSymPorts();
            break;
        case e_mme_geometry_4xh:
            mmeGrid.fcd = 1;
            mmeGrid.spatial = mmeLimit;
            set2xhPorts();
            break;
    }
}

void Gaudi2GeoAttr::setGridsForDedwCdConcurrency2x()
{
    setGrids();
    mmeGrid = {};
    mmeGrid.cd = 2;
}

void Gaudi2GeoAttr::setGridsForFp8DedwCdConcurrency4x()
{
    // When this optimization is applied for dedw, all cores read spatial rows from
    // a and b in interleaved. So they can be seen as placed on top of each other vertically.
    // No batch.
    resetGrids();
    mmeGrid.cd = 2;
    coreGrid.cd = 2;
    aGrid.fcd = 1;
    aGrid.spatial = 2;
    aGrid.batch = 1;
    bGrid.fcd = 1;
    bGrid.spatial = 2;
    bGrid.batch = 1;
    cGrid.fcd = 1;
    cGrid.spatial = 2;
    cGrid.batch = 1;
}

bool Gaudi2GeoAttr::getBgemmBit() const
{
    //  bgemm 8x MME concurrency requires a special bit in the descriptor.
    return cGrid.batch > 1 || cGrid.cd > 1;
}

bool Gaudi2GeoAttr::getHx2Bit() const
{
    return getMmeFcdPorts(e_mme_op_c) == 1;
}

bool Gaudi2GeoAttr::getDoubleAccumsBit() const
{
    if (aGrid.batch > 1 && cGrid.batch == 1)
    {
        //  this can only happen in case B is broadcasted and we work concurrently on different batches/filters in the
        //  same EU. in this case the effective spatialSize can be larger than half the EU.
        return false;
    }
    //  in Gaudi2 Bgemm mode the spatial size of each gemm is up to 64.
    if (getBgemmBit()) return true;

    // [SW-101193] turn on doubleAccums for short output
    return false; /*CommonGeoAttr::getDoubleAccumsBit()*/
}

void Gaudi2GeoAttr::setBgemm4x()
{
    //  2 gemms per EU
    aGrid.spatial = 1;
    aGrid.fcd = 1;
    aGrid.batch = 2;
    bGrid.spatial = 1;
    bGrid.fcd = 1;
    bGrid.batch = 2;
    cGrid.spatial = 1;
    cGrid.fcd = 1;  // This will turn on 2xh mode
    cGrid.batch = 2;
    coreGrid.spatial = 1;
    coreGrid.fcd = 1;
    coreGrid.batch = 2;
}

void Gaudi2GeoAttr::setBgemm2x()
{
    //  1 gemm per EU
    cGrid.fcd = 1;
    cGrid.spatial = 2;
    coreGrid.fcd = 1;
    coreGrid.spatial = 1;
    coreGrid.batch = 2;
}

bool Gaudi2GeoAttr::isPortSharedBetweenCores(EMmeInternalOperand operand) const
{
    // in 4xw concurrency mode the MME concurrency is higher than 1 and it shouldn't share ports between cores
    // but due to HW limitations we are forced to share the A port and work around ignoring the useless data.
    if (operand == e_mme_op_a && m_4xwConcurrencyMode) return true;

    return CommonGeoAttr::isPortSharedBetweenCores(operand);
}

void Gaudi2GeoAttr::setBgemmConcurrency()
{
    // We first check the hybrid mode of 2x inter-mme and 2x intra-mme which is currently
    // supported in specific conditions.
    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();
    if (mmeGrid.batch == 2)  // Already 2x, where each mme works on different gemms
    {
        if ((getGeometryHeight() == 512) &&  // 4xh
            0 &&  //  [SW-99258] reneable 4xGemmPerCore for sizes 256-384
            (fcd <= 64) && (sp >= 256))
        {
            // Each MME works on 2 gemms, which means a total of 4x
            setBgemm2x();
            return;
        }
    }

    if (mmeGrid.fcd != 1 || mmeGrid.spatial != 1)
    {
        MME_ASSERT(0, "expected only batch or CD MMEs");
    }

    //  2 gemms per EU, which means 4 gemms per MME and a total of 8x
    if (fcd <= getPortSize(e_mme_op_b) && sp <= getPortSize(e_mme_op_b))
    {
        //  due to HW limitations this mode cant operate if the input is FP8 or if it is not dense.
        const auto& bView = m_params.getOperand(e_mme_op_b);
        const bool isFirstDimDense = bView.sizes[0] == bView.strides[1];
        if (!isFp8 && isFirstDimDense)
        {
            setBgemm4x();
            return;
        }
    }

    // general check for MME concurrency of 2
    if (isPortSharedBetweenCores(e_mme_op_a))
    {
        // make sure the FCD and SP fit the core size, and that we do not have a third batch dimension as we use an
        // extra loop for the workaround so we need to make sure it wont be needed
        if (fcd <= getEuWidth() && sp <= getEuHeight() / 2 && m_params.getOperand(e_mme_op_c).sizes[GEMM_DIM_B3] == 1)
        {
            MME_ASSERT(m_params.strategy.geometry == e_mme_geometry_4xw, "logic assumes this is 4xW geometry");
            if (!isFp8 || isTransposed(e_mme_op_a))
            {
                coreGrid.fcd = 1;
                coreGrid.batch = 2;
                m_4xwConcurrencyMode = true;
                return;
            }
        }
    }
    else if (isPortSharedBetweenCores(e_mme_op_b))
    {
        if (m_params.strategy.geometry == e_mme_geometry_4xh)
        {
            if (fcd <= getPortSize(e_mme_op_b) && sp <= getEuHeight())
            {
                if (!isFp8 || isTransposed(e_mme_op_b))
                {
                    setBgemm2x();
                    return;
                }
            }
        }
        else if (fcd <= (getEuWidth() / 2) && sp <= getEuHeight())
        {
            setBgemm2x();
            return;
        }
    }

    // check for 4xHybrid mode
    if (fcd <= (getEuWidth() / 2) && sp <= getEuHeight() && 0)
    {
        // [SW-78758] fix bug with 4xh bgemm4x fp8 test fail if we try fp8 with 4xHybrid
        if (isFp8) setSymPorts();
        // Each MME works on 2 gemms, which means a total of 4x
        setBgemm2x();
        return;
    }
    // if we reached here we failed to get any internal MME concurrency, it will remain 2.
}

// Asymmetric port configuration is used to overcome port constraint penalty when the actual number of ports
// enables that.
// The hw has 10 ports per MME, but only 8 SBs. Which means that two pairs of ports share the same SB. Therefore,
// when both ports that are connected to the same SB are actually used, they will conflict and data read time
// is doubled (a.k.a. port constraint panelty.). The ports are symmetric for each MME code (5 ports with 4 SBs
// per core).
// In asymmtric geometries each MME core has 4 ports on the long dim of the geometry and 1 port on the short dim.
// Ports SB0 and SB1 share an actual single SB.
// Symmetric geometries use up to 8 ports and therefore never conflict.
//
// There are cases where we can avoid the port constraint penalty in non-symmetric geometries. This requires that
// the total of port that are actually used does not exceed the actual number of SBs, and that they can be
// configured such only one port out of SB0 and SB1 is used.
//
bool Gaudi2GeoAttr::canApplyasymPortConfigForNoPortConstraint()
{
    if (mmeGrid.cd == 1)
    {
        // The asymmetric port config mode is currently supported in hybrid mode
        return false;
    }
    if (getMmePortsNr(e_mme_op_a) + getMmePortsNr(e_mme_op_b) <= c_inputPort)
    {
        return false;  // geometry is not port constraint
    }
    if (getCorePortsNr(e_mme_op_b) == 1)  // 4xh
    {
        if (isTransposed(e_mme_op_a))
        {
            return false;  // currently only the non-transposed case is supported
        }

        unsigned sp = m_params.getSpatialSize();
        unsigned fcd = m_params.getFcdSize();
        unsigned numPortsPerSingleBatch = div_round_up(sp, getPortSize(e_mme_op_a));
        unsigned totalFcdPortsPerCore = getCorePortsNr(e_mme_op_a);
        unsigned batchSize = m_params.getOperand(e_mme_op_c).sizes[getConcurrentDim()];
        unsigned effectiveBatchConcurrency = std::min(getGeometryConcurrency(), batchSize);

        // When the last A port per MME core is unused, it means that there is no port constraint anyway
        if (numPortsPerSingleBatch * effectiveBatchConcurrency <= (aGrid.fcd - 1))
        {
            return false;
        }

        // When more than 1 B port is required, than at least one core uses all 4 A ports + 1 B port
        // so port constraint cannot be avoided
        if (fcd > getPortSize(e_mme_op_b))
        {
            return false;
        }

        unsigned effectiveTotalNumberOfFcdPorts = numPortsPerSingleBatch * effectiveBatchConcurrency;
        if (effectiveTotalNumberOfFcdPorts >= 2 * totalFcdPortsPerCore)
        {
            // The effective total number of ports is equal or larger than the available fcd ports per mme
            return false;
        }

        // First case: effective number of ports used leaves 1 port per core not used.
        if (effectiveTotalNumberOfFcdPorts <= 2 * (totalFcdPortsPerCore - 1))
        {
            return true;
        }

        // Second case: the number of ports per single batch is more than the number of ports per single
        // core, but leaves 2 ports unused
        if ((numPortsPerSingleBatch > totalFcdPortsPerCore) &&
            (numPortsPerSingleBatch <= 2 * (totalFcdPortsPerCore - 1)))
        {
            return true;
        }
    }
    if (getCorePortsNr(e_mme_op_a) == 1)  // 4xw
    {
        return false;  // asymmetric port constraint on 4xw is not supported yet
    }

    // Default
    return false;
}

void Gaudi2GeoAttr::setDedwConcurrency()
{
    if (isFp8)
    {
        return;  // Gaudi2 does not support batch concurrency
    }
    // If there are steps, return to the original geometry
    if (mmeGrid.fcd != 1 || mmeGrid.spatial != 1)
    {
        setGrids();
        return;
    }

    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();

    while (getMmeFcdPorts(e_mme_op_a) > 1)
    {
        if (fcd <= getEuWidth() && sp <= ((getMmeFcdPorts(e_mme_op_a) / 2) * getPortSize(e_mme_op_a)))
        {
            if (coreGrid.spatial > 1 || coreGrid.fcd > 1)
            {
                //  1 gemm per EU
                coreGrid.fcd = 1;
                coreGrid.spatial = 1;
                coreGrid.batch = 2;
            }
            else
            {
                aGrid.fcd /= 2;
                aGrid.batch *= 2;
            }
        }
        else
            break;
    }

    if (canApplyasymPortConfigForNoPortConstraint())
    {
        m_asymPortConfigForNoPortConstraint = true;
    }
}

void Gaudi2GeoAttr::setCdConcurrency()
{
    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();

    if (isPortSharedBetweenCores(MmeCommon::e_mme_op_b))
    {
        // since B is shared each core FCD is larger than what can be utilized in a seperated core geometry
        // we need to make sure that the size is at most half the full core width
        if (fcd <= getEuWidth() / 2)
        {
            // also we need to check that either the output SP is small enough to fit in a single core
            // or that the reduced granularity will result in better performance.

            // for example, if the output SP is 384 and the current single MME sizes are 256x256 it will be
            // done in 2 steps, if we increase the cd concurrency by 2x we will work in 2x128x128. now it will take 3
            // steps, but each one of them will be twice as fast. so instead of 2 steps each taking X cycles. we will
            // get 3 steps each taking X/2 cycles. overall a 25% speedup.

            // TODO this logic currently causes wrong outputs and requires an investigation and fixing SW-127191
            unsigned reminder = sp % getMmeHeight();
            if (sp <= getEuHeight() || (reminder <= getEuHeight() && reminder > 0))
            {
                // Reset output grids except mmeGrid
                cGrid = {};
                coreGrid = {};
                // set core cd concurrency
                coreGrid.cd = getCoresPerMmeNr();
                // not sharing B anymore, need to set the second output port to a seperated core geometry
                cGrid.spatial = 2;
            }
        }

        // try to further increase the concurrency to 2 per core
        if (!isFp8 && fcd <= getPortSize(e_mme_op_b) && sp <= getPortSize(e_mme_op_a))
        {
            aGrid = {};
            aGrid.spatial = 2;
            bGrid = {};
            bGrid.spatial = 2;
            cGrid = {};
            cGrid.cd = 2;
        }
    }
}

void Gaudi2GeoAttr::set2xwPorts()
{
    aGrid.fcd = 1;
    aGrid.spatial = 1;

    if (isTransposed(e_mme_op_b))
    {
        bGrid.fcd = 1;
        bGrid.spatial = 4;
    }
    else
    {
        if (isFp8)
        {
            bGrid.fcd = 2;
            bGrid.spatial = 2;
        }
        else
        {
            bGrid.fcd = 4;
            bGrid.spatial = 1;
        }
    }

    cGrid.fcd = 2;
    cGrid.spatial = 1;

    coreGrid.fcd = 2;
    coreGrid.spatial = 1;
}

void Gaudi2GeoAttr::set2xhPorts()
{
    if (isTransposed(e_mme_op_a))
    {
        aGrid.fcd = 1;
        aGrid.spatial = 4;
    }
    else
    {
        if (isFp8)
        {
            aGrid.fcd = 2;
            aGrid.spatial = 2;
        }
        else
        {
            aGrid.fcd = 4;
            aGrid.spatial = 1;
        }
    }

    bGrid.fcd = 1;
    bGrid.spatial = 1;

    cGrid.fcd = 1;
    cGrid.spatial = 2;

    coreGrid.fcd = 1;
    coreGrid.spatial = 2;
}

void Gaudi2GeoAttr::setSymPorts()
{
    if (isTransposed(e_mme_op_a))
    {
        aGrid.fcd = 1;
        aGrid.spatial = 2;
    }
    else
    {
        if (isFp8)
        {
            aGrid.fcd = 1;
            aGrid.spatial = 2;
        }
        else
        {
            aGrid.fcd = 2;
            aGrid.spatial = 1;
        }
    }

    if (isTransposed(e_mme_op_b))
    {
        bGrid.fcd = 1;
        bGrid.spatial = 2;
    }
    else
    {
        if (isFp8)
        {
            bGrid.fcd = 1;
            bGrid.spatial = 2;
        }
        else
        {
            bGrid.fcd = 2;
            bGrid.spatial = 1;
        }
    }

    cGrid.fcd = 2;
    cGrid.spatial = 1;

    coreGrid.fcd = 1;
    coreGrid.spatial = 2;
}

unsigned Gaudi2GeoAttr::getPortSize(EMmeInternalOperand operand) const
{
    switch (operand)
    {
        default:
            MME_ASSERT(0, "invalid operand");
        case e_mme_op_a:
        case e_mme_op_b:
            if (isFp8)
                // since fp8 is only 1 byte each port will bring 128 elements instead of 64
                return Mme::c_cl_size;
            else
                //  for larger data types each ports brings only 64 elements by bringing 1 or 2 cache lines.
                return Mme::c_cl_size / 2;
        case e_mme_op_c:
            // a single output port handles 128 elements.
            return Mme::c_cl_size;
    }
}

unsigned Gaudi2GeoAttr::getAccHeight() const
{
    return Mme::c_cl_size;
}

unsigned Gaudi2GeoAttr::getEuWidth() const
{
    // TODO: [SW-78758] test all bgemm and dedw concurrency cases, maybe move up to common
    if (getMmeConcurrency() == 1 || isOperandFullyBroadcasted(e_mme_op_b))
    {
        return cGrid.fcd * Mme::c_cl_size;
    }

    // since concurrency is at least 2 and we have at most 2 cores and we are not broadcasted - we know there is no
    // sharing between cores, and the size is only determined by a single core port config
    if (isTransposed(e_mme_op_b))
    {
        return bGrid.spatial * getTeHeight();
    }
    else
    {
        return bGrid.fcd * getPortSize(e_mme_op_b);
    }
}

unsigned Gaudi2GeoAttr::getEuHeight() const
{
    // in case doulbleAccums bit is on only hald the EU will be used.
    if (getDoubleAccumsBit())
    {
        return Mme::c_cl_size / 2;
    }

    // generally the EU height is 128 rows per spatial output port
    // in 4xw concurrency mode, even though the MME concurrency is larger than 1, it is configured
    if ((getMmeConcurrency() == 1 || isOperandFullyBroadcasted(e_mme_op_a) || m_4xwConcurrencyMode) &&
        getMmeCdConcurrency() == 1)
    {
        return cGrid.spatial * Mme::c_cl_size;
    }

    // since concurrency is at least 2 and we have at most 2 cores and we are not broadcasted - we know there is no
    // sharing between cores, and the size is only determined by a single core port config
    if (isTransposed(e_mme_op_a))
    {
        return aGrid.spatial * getTeHeight();
    }
    else
    {
        return aGrid.fcd * getPortSize(e_mme_op_a);
    }
}

unsigned Gaudi2GeoAttr::getMmeWidth() const
{
    return coreGrid.fcd * getEuWidth();
}

unsigned Gaudi2GeoAttr::getMmeHeight() const
{
    return coreGrid.spatial * getEuHeight();
}

unsigned Gaudi2GeoAttr::getTeHeight() const
{
    return Mme::c_cl_size / 2;
}

unsigned Gaudi2GeoAttr::getCoresPerMmeNr() const
{
    return 2;
}

unsigned Gaudi2GeoAttr::getCoreSpatialEuPort(EMmeInternalOperand operand) const
{
    if (isFp8 && !isTransposed(operand))
    {
        return getCoreSpatialPorts(operand);
    }
    return 1;
}

bool Gaudi2GeoAttr::isGeometryPortConstrained() const
{
    EMmeDataType dt = m_params.getOperand(MmeCommon::e_mme_op_a).elementType;
    //  in fp32 the MME is already compute bound, reducing the BW by 2x wont have perf effect.
    //  in other DTs the BW and compute are balanced so we use more SBs then the amount of port we are constrained
    if (dt == e_type_fp32 || dt == MmeCommon::e_type_fp32_ieee) return false;
    // even though the geometry itself is port constrained, there is a chance not all SBs in the geometry are actually
    // used
    if (getMmePortsNr(e_mme_op_a) + getMmePortsNr(e_mme_op_b) > c_inputPort)
    {
        // currently only supporting the very specific case of single B FCD port and 3 A SP ports
        if (getCorePortsNr(e_mme_op_b) == 1 && getCorePortsNr(e_mme_op_a) == aGrid.fcd)
        {
            // the forth A port is unused, meaning overall each core uses only 4 ports.
            if (m_params.getSpatialSize() <= (aGrid.fcd - 1) * getPortSize(e_mme_op_a))
            {
                return false;
            }
        }

        // Case where ports are configured in an asymmetric way to avoid port constraint:
        if (m_asymPortConfigForNoPortConstraint)
        {
            return false;
        }

        // indeed port constrained
        return true;
    }

    return false;
}

unsigned Gaudi2GeoAttr::getEffectiveBatchConcurrency() const
{
    unsigned batchGeometryConcurrency = getGeometryConcurrency();
    if (!m_asymPortConfigForNoPortConstraint)
    {
        return batchGeometryConcurrency;
    }
    unsigned batchSize = m_params.getOperand(e_mme_op_c).sizes[getConcurrentDim()];
    return std::min(batchGeometryConcurrency, batchSize);
}

bool Gaudi2GeoAttr::isPortValid(EMmeInternalOperand operand,
                                unsigned core,
                                unsigned cdIdx,
                                unsigned batchIdx,
                                unsigned fcdIdx,
                                unsigned spIdx) const
{
    if (m_asymPortConfigForNoPortConstraint)
    {
        // In this mode we want to invalidate the two fcd ports of the slave.
        MME_ASSERT(cdIdx == 0, "In asymmetric port config, number of CD ports is expected to be 1");
        if (operand == MmeCommon::e_mme_op_a)
        {
            MME_ASSERT(spIdx == 0, "In dedw asymmetric port config, number of A spatial ports is expected to be 1");
        }
        if (operand == MmeCommon::e_mme_op_c)
        {
            MME_ASSERT(fcdIdx == 0, "In dedw asymmetric port config, number of C spatial ports is expected to be 1");
        }

        if ((core == Mme::MME_CORE_SLAVE) &&  // slave
            ((batchIdx == 1) ||  // second filter out of two of the slave
             (fcdIdx >= 2)))  // last two fcd ports of the slave
        {
            return false;
        }
    }
    return true;
}
bool Gaudi2GeoAttr::shouldSwapMasterAndSlave(MmeCommon::EMmeInternalOperand operand) const
{
    if (m_asymPortConfigForNoPortConstraint)
    {
        // Only the Master can read B0.
        // In addition, the 4x flow assigns the first 2x batches to Master and the next 2x batches
        // to the Slave.
        // In this particular mode, we disable the last 1x batch. So unless we swap Master and Slave,
        // Master will read B0 and also perform 2x which leads to port constraint. Swap solves that
        // because now the Slave performs 2x while Master 1x in addition to reading B0.
        if (operand == MmeCommon::e_mme_op_a || operand == MmeCommon::e_mme_op_c)
        {
            return true;
        }
        return false;
    }
    return false;
}

// In hybrid pattern, data is written interleaved between ports of the same dcore,
// but non-interleaved between different dcores.
bool Gaudi2GeoAttr::isHybridPattern() const
{
    return (m_params.getGeometry() == e_mme_geometry_4xh) && (!isTransposed(e_mme_op_a));
}

bool Gaudi2GeoAttr::doPortAdvanceSpatially(EMmeInternalOperand operand) const
{
    // in gaudi2 when ports are transposed they always advance spatially between cores.
    // otherwise they always advance on the FCD
    if (isTransposed(operand)) return true;

    // specifically in Fp8 in the narrow geoemtries the ports also advance spatillay between cores due to
    // port size limitations
    if (isFp8)
    {
        if (operand == e_mme_op_a && m_params.strategy.geometry == e_mme_geometry_4xw) return true;
        if (operand == e_mme_op_b && m_params.strategy.geometry == e_mme_geometry_4xh) return true;
    }

    return false;
}

}  // namespace Gaudi2