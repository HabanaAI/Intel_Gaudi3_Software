#include "gaudi3_geo_attr.h"
#include "gaudi3/mme.h"

using namespace MmeCommon;

namespace gaudi3
{
void Gaudi3GeoAttr::setGrids()
{
    if (m_params.isNativeDmaOperation())
    {
        setGridsDma();
    }
    else
    {
        setGridsMme();
    }
}

void Gaudi3GeoAttr::setGridsMme()
{
    unsigned mmeLimit = m_params.strategy.mmeLimit;
    if (mmeLimit > 1)
    {
        switch (m_params.getGeometry())
        {
            default:
                MME_ASSERT(0, "Geometry not supported");
            case e_mme_geometry_2xw:
                mmeGrid.fcd = mmeLimit / 2;
                mmeGrid.spatial = 2;
                break;
            case e_mme_geometry_4xw:
                mmeGrid.fcd = mmeLimit;
                mmeGrid.spatial = 1;
                break;
            case e_mme_geometry_2xh:
                mmeGrid.fcd = 2;
                mmeGrid.spatial = mmeLimit / 2;
                break;
            case e_mme_geometry_4xh:
                mmeGrid.fcd = 1;
                mmeGrid.spatial = mmeLimit;
                break;
        }
    }

    //  in Gaudi3 ports are always split across the spatial dimension since
    //  each ports handles the entirety of the FCD.
    aGrid.fcd = 1;
    aGrid.spatial = 4;
    bGrid.fcd = 1;
    bGrid.spatial = 4;
    cGrid.fcd = 1;
    cGrid.spatial = 2;
}

void Gaudi3GeoAttr::setGridsDma()
{
    unsigned mmeLimit = m_params.strategy.mmeLimit;
    switch (m_params.getGeometry())
    {
        default:
            MME_ASSERT(0, "Geometry not supported");
        case e_mme_geometry_2xw:
        case e_mme_geometry_4xw:
            mmeGrid.fcd = mmeLimit;
            mmeGrid.spatial = 1;
            break;
        case e_mme_geometry_2xh:
        case e_mme_geometry_4xh:
            mmeGrid.fcd = 1;
            mmeGrid.spatial = mmeLimit;
            break;
    }

    bGrid.fcd = 0;
    bGrid.spatial = 0;
    //  dma operations use only 2 A and the 2 C ports
    switch (m_params.getGeometry())
    {
        default:
            MME_ASSERT(0, "Geometry not supported");
        case e_mme_geometry_4xw:
        case e_mme_geometry_2xh:
            if (isTransposed(MmeCommon::e_mme_op_a))
            {
                aGrid.fcd = 1;
                aGrid.spatial = 2;
            }
            else
            {
                aGrid.fcd = 2;
                aGrid.spatial = 1;
            }
            cGrid.fcd = 2;
            cGrid.spatial = 1;
            break;
        case e_mme_geometry_2xw:
        case e_mme_geometry_4xh:
            if (isTransposed(MmeCommon::e_mme_op_a))
            {
                aGrid.fcd = 2;
                aGrid.spatial = 1;
            }
            else
            {
                aGrid.fcd = 1;
                aGrid.spatial = 2;
            }
            cGrid.fcd = 1;
            cGrid.spatial = 2;
            break;
    }
}

bool Gaudi3GeoAttr::getNonShareABit() const
{
    //  special case in which A works on a batch per EU, but B is shared between EUs
    return aGrid.batch > 1 && bGrid.batch == 1;
}

bool Gaudi3GeoAttr::getBgemmBit() const
{
    //  bgemm MME concurrency requires a special bit in the descriptor.
    return aGrid.batch > 1 && bGrid.batch > 1;
}

void Gaudi3GeoAttr::setBgemmConcurrency()
{
    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();
    MmeDimsIndex concurrentDim = getConcurrentDim();

    //  1 Gemm per EU, 2 per MME, B is broadcasted
    if (isOperandBroadcasted(MmeCommon::e_mme_op_b, concurrentDim) && fcd <= getEuWidth() && sp <= getEuHeight())
    {
        aGrid.spatial /= 2;
        cGrid.spatial /= 2;
        aGrid.batch = 2;
        cGrid.batch = 2;
        return;
    }

    //  1 Gemm per EU, 2 per MME
    if (fcd <= (getEuWidth() / 2) && sp <= getEuHeight())
    {
        aGrid.spatial /= 2;
        bGrid.spatial /= 2;
        cGrid.spatial /= 2;
        aGrid.batch = 2;
        bGrid.batch = 2;
        cGrid.batch = 2;
        return;
    }
}

void Gaudi3GeoAttr::setDedwConcurrency()
{
    // If there are steps, return to the original geometry
    if (mmeGrid.fcd != 1 || mmeGrid.spatial != 1)
    {
        setGrids();
        return;
    }

    //  configuring 2 filters per MME requires both start and roiBase offsets per port
    //  since we can only have one of them this optimization is canceled.
    return;

    unsigned sp = m_params.getSpatialSize();
    //  1 Gemm per EU, 2 per MME
    if (sp <= getEuHeight())
    {
        aGrid.spatial /= 2;
        cGrid.spatial /= 2;
        aGrid.batch = 2;
        cGrid.batch = 2;
    }
}

void Gaudi3GeoAttr::setCdConcurrency()
{
    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();

    // If output spatial dims fit within 128x128, we can split the CD among the 4 cores. We cannot use
    // the full 256x128 due to lack of ports.
    if ((fcd <= 128) && (sp <= 128))
    {
        // Reset all grids except mmeGrid
        aGrid = {};
        bGrid = {};
        cGrid = {};

        aGrid.spatial = 4;
        bGrid.spatial = 4;
        cGrid.spatial = 2;
    }
}

unsigned Gaudi3GeoAttr::getPortSize(EMmeInternalOperand operand) const
{
    if (m_params.opType == MmeCommon::e_mme_memcpy)
    {
        //  this is a bit misleading, when DT size > 1 we will access more then one memory CL.
        return m_mmeHal.getMemoryClSize();
    }
    else if (m_params.opType == e_mme_trans)
    {
        //  only in trans mode the amount of contiguous fcd elements handled by a port is less than 256
        return getTeHeight();
    }
    else
    {
        if (operand == e_mme_op_a)
        {
            unsigned mmeSize = getMmeHeight();
            if (getDoubleAccumsBit())
            {
                // even though the EU size was reduced by half A port still needs to pad to the size of the whole EU.
                mmeSize *= 2;
            }
            return mmeSize;
        }
        else
        {
            //  in Gaudi3 each port feeds the whole width of the MME
            return getMmeWidth();
        }
    }
}

unsigned Gaudi3GeoAttr::getAccHeight() const
{
    return Mme::c_cl_size;
}

unsigned Gaudi3GeoAttr::getEuWidth() const
{
    if (getBgemmBit())
    {
        return Mme::c_cl_size;
    }
    else
    {
        return 2 * Mme::c_cl_size;
    }
}

unsigned Gaudi3GeoAttr::getEuHeight() const
{
    if (getDoubleAccumsBit())
    {
        return Mme::c_cl_size / 2;
    }

    return Mme::c_cl_size;
}

unsigned Gaudi3GeoAttr::getMmeWidth() const
{
    if (m_params.isNativeDmaOperation())
    {
        //  dma operations bypasses the EU and can be wider or narrower.
        return cGrid.fcd * getPortSize(e_mme_op_c);
    }
    else
    {
        //  Gaudi3 doesnt have internal geometries, simply use EU size
        return getEuWidth();
    }
}

unsigned Gaudi3GeoAttr::getMmeHeight() const
{
    if (m_params.isNativeDmaOperation())
    {
        //  dma operations bypasses the EU and can be wider or narrower.
        return cGrid.spatial * getPortSize(e_mme_op_c);
    }
    else
    {
        //  Gaudi3 doesnt have internal geometries, simply use EU size
        if (getBgemmBit() || getNonShareABit())
        {
            return getEuHeight();
        }
        else
        {
            return 2 * getEuHeight();
        }
    }
}

unsigned Gaudi3GeoAttr::getTeHeight() const
{
    if (m_params.opType == MmeCommon::e_mme_trans && m_params.y.elementType == MmeCommon::e_type_fp32)
    {
        //  only in this specific configuration the TE works on fewer elements.
        return Mme::c_cl_size / 4;
    }
    else
    {
        //  regardless of dataType the TE always works as if there are 64 elements.
        return Mme::c_cl_size / 2;
    }
}

unsigned Gaudi3GeoAttr::getCoresPerMmeNr() const
{
    // in Gaudi3 each MME counts as a single core
    return 1;
}

unsigned Gaudi3GeoAttr::getCoreSpatialEuPort(EMmeInternalOperand operand) const
{
    if (!isTransposed(operand))
    {
        return getCoreSpatialPorts(operand);
    }
    return 1;
}

bool Gaudi3GeoAttr::doPortAdvanceSpatially(EMmeInternalOperand operand) const
{
    // in Gaudi3 the cores cannot be arranged on the FCD axis, and ports are always arranged on the spatial axis
    // so in all cases the cores spatial arrangement is translated into ports spatial arrangement
    return true;
}
}  // namespace gaudi3