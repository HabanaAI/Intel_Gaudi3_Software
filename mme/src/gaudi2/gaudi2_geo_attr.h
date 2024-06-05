#ifndef MME__GAUDI2_GEO_ATTR_H
#define MME__GAUDI2_GEO_ATTR_H

#include "gaudi2_mme_hal_reader.h"
#include "src/mme_common/common_geo_attr.h"

namespace Gaudi2
{
class Gaudi2GeoAttr : public MmeCommon::CommonGeoAttr
{
public:
    Gaudi2GeoAttr(const MmeCommon::MmeLayerParams& params)
    : CommonGeoAttr(params, MmeHalReader::getInstance()),
      isFp8(isTypeFp8(params.getOperand(MmeCommon::e_mme_op_a).elementType))
    {
        init();
    };
    virtual ~Gaudi2GeoAttr() = default;

    virtual unsigned getAccHeight() const override;  //  ACC height in rows
    virtual unsigned getEuWidth() const override;  //  EU width in elements
    virtual unsigned getEuHeight() const override;  //  EU height in elements
    virtual unsigned getMmeWidth() const override;  //  MMME width in elements, geometry dependant
    virtual unsigned getMmeHeight() const override;  //  MME height in elements, geometry dependant
    virtual unsigned getPortSize(MmeCommon::EMmeInternalOperand operand) const override;  //  port size in elements
    virtual unsigned getTeHeight() const override;  //  number of rows in a single TE
    virtual unsigned getCoresPerMmeNr() const override;  // number of cores in a single MME
    virtual unsigned getCoreSpatialEuPort(MmeCommon::EMmeInternalOperand operand) const override;
    virtual bool doPortAdvanceSpatially(MmeCommon::EMmeInternalOperand operand) const override;
    virtual bool isGeometryPortConstrained() const override;
    unsigned getEffectiveBatchConcurrency() const override;

    virtual bool getBgemmBit() const override;
    virtual bool getHx2Bit() const override;
    virtual bool getDoubleAccumsBit() const override;
    bool isHybridPattern() const override;

protected:
    virtual void setGrids() override;
    void setGridsForDedwCdConcurrency2x();
    void setGridsForFp8DedwCdConcurrency4x();
    virtual void setBgemmConcurrency() override;
    void setBgemm4x();
    void setBgemm2x();
    virtual void setDedwConcurrency() override;
    virtual void setCdConcurrency() override;
    virtual bool isPortValid(MmeCommon::EMmeInternalOperand operand,
                             unsigned core,
                             unsigned cd,
                             unsigned batch,
                             unsigned fcd,
                             unsigned sp) const override;
    virtual bool shouldSwapMasterAndSlave(MmeCommon::EMmeInternalOperand operand) const override;
    virtual bool isAsymPortConfigMode() const override { return m_asymPortConfigForNoPortConstraint; }
    virtual bool isMmeConcurrencyRoutingWorkAround() const override { return m_4xwConcurrencyMode; }
    virtual bool isPortSharedBetweenCores(MmeCommon::EMmeInternalOperand operand) const override;

private:
    unsigned c_inputPort = 8;  //  Gaudi2 has 8 input ports
    const bool isFp8;
    void setSymPorts();
    void set2xhPorts();
    void set2xwPorts();
    bool canApplyasymPortConfigForNoPortConstraint();
    bool m_asymPortConfigForNoPortConstraint = false;
    bool m_4xwConcurrencyMode = false;
};
}  // namespace Gaudi2

#endif //MME__GAUDI2_GEO_ATTR_H
