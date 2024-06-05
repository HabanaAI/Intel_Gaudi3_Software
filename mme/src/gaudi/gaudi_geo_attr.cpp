#include "gaudi_geo_attr.h"
#include "gaudi/mme.h"

using namespace MmeCommon;

namespace gaudi
{
void GaudiGeoAttr::setGrids()
{
    switch (m_params.getGeometry())
    {
        default:
            MME_ASSERT(0, "Geometry not supported");
        case e_mme_geometry_4wx1h:
            mmeGrid.fcd = 2;
            mmeGrid.spatial = 1;
            set2xwPorts();
            break;
        case e_mme_geometry_2wx2h:
            mmeGrid.fcd = 1;
            mmeGrid.spatial = 2;
            set2xwPorts();
            break;
        case e_mme_geometry_1wx4h:
            mmeGrid.fcd = 1;
            mmeGrid.spatial = 2;
            set2xhPorts();
            break;
    }

    //  batch 2X optimization is not implemented yet
    mmeGrid.batch = 1;
    aGrid.batch = 1;
    bGrid.batch = 1;
    cGrid.batch = 1;
}

//  Gaudi behaves a bit differently then newer chips.
//  instead of batch concurrency it performs unroll and its bgemm2x is also different.
//  for those reasons it has its own concurrency flow
void GaudiGeoAttr::setChipConcurrency()
{
    if (!m_params.isGemmOperation()) return;

    // 2x is supported only in symmetric geometry.  added to align to current H3 implementation.
    if (getGeometryWidth() != getGeometryHeight()) return;

    unsigned fcd = m_params.getFcdSize();
    unsigned sp = m_params.getSpatialSize();

    //  convert unused MME to work on the next batch
    if (sp <= getEuHeight() && fcd <= 2 * getEuWidth())
    {
        set2xwPorts();
        mmeGrid.spatial = 1;
        mmeGrid.fcd = 1;
        mmeGrid.batch = 2;
    }
    else if (fcd <= getEuWidth() && sp <= 2 * getEuHeight())
    {
        set2xhPorts();
        mmeGrid.spatial = 1;
        mmeGrid.fcd = 1;
        mmeGrid.batch = 2;
    }

    return;
}

void GaudiGeoAttr::setBgemmConcurrency()
{
    MME_ASSERT(0, "should never get here");
    return;
}

void GaudiGeoAttr::setDedwConcurrency()
{
    MME_ASSERT(0, "should never get here");
    return;
}

void GaudiGeoAttr::setCdConcurrency()
{
    MME_ASSERT(0, "should never get here");
    return;
}

void GaudiGeoAttr::set2xhPorts()
{
    if (isTransposed(e_mme_op_a))
    {
        aGrid.fcd = 1;
        aGrid.spatial = 2;
    }
    else
    {
        aGrid.fcd = 2;
        aGrid.spatial = 1;
    }

    bGrid.fcd = 1;
    bGrid.spatial = 1;

    cGrid.fcd = 1;
    cGrid.spatial = 2;

    transC = true;
}

void GaudiGeoAttr::set2xwPorts()
{
    if (isTransposed(e_mme_op_b))
    {
        bGrid.fcd = 1;
        bGrid.spatial = 2;
    }
    else
    {
        bGrid.fcd = 2;
        bGrid.spatial = 1;
    }

    aGrid.fcd = 1;
    aGrid.spatial = 1;

    cGrid.fcd = 2;
    cGrid.spatial = 1;

    transC = false;
}

unsigned GaudiGeoAttr::getEuSize() const
{
    if (m_params.getOperand(e_mme_op_a).elementType == e_type_bf16)
    {
        return 64;
    }
    else
    {
        return 32;
    }
}

unsigned GaudiGeoAttr::getPortSize(EMmeInternalOperand operand) const
{
    //  in Gaudi each port feeds the whole width of the EU
    return getEuWidth();
}

unsigned GaudiGeoAttr::getAccHeight() const
{
    return getEuSize();
}

unsigned GaudiGeoAttr::getEuWidth() const
{
    return getEuSize();
}

unsigned GaudiGeoAttr::getEuHeight() const
{
    return getEuSize();
}

unsigned GaudiGeoAttr::getMmeWidth() const
{
    switch (m_params.getGeometry())
    {
        default:
            MME_ASSERT(0, "Geometry not supported");
        case e_mme_geometry_4wx1h:
            return 2 * getEuWidth();
        case e_mme_geometry_2wx2h:
            return 2 * getEuWidth();
        case e_mme_geometry_1wx4h:
            return getEuWidth();
    }
}

unsigned GaudiGeoAttr::getMmeHeight() const
{
    switch (m_params.getGeometry())
    {
        default:
            MME_ASSERT(0, "Geometry not supported");
        case e_mme_geometry_4wx1h:
            return getEuHeight();
        case e_mme_geometry_2wx2h:
            return getEuHeight();
        case e_mme_geometry_1wx4h:
            return 2 * getEuHeight();
    }
}

unsigned GaudiGeoAttr::getTeHeight() const
{
    return getEuSize();
}

unsigned GaudiGeoAttr::getCoresPerMmeNr() const
{
    return 1;
}

bool GaudiGeoAttr::doPortAdvanceSpatially(EMmeInternalOperand operand) const
{
    // no cores in Gaudi1
    return false;
}

bool GaudiGeoAttr::isDimConcurrencyOptimizationSupported() const
{
    return false;
}

}  // namespace gaudi