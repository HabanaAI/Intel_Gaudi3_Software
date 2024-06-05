#ifndef MME__GAUDI_GEO_ATTR_H
#define MME__GAUDI_GEO_ATTR_H

#include "gaudi_mme_hal_reader.h"
#include "src/mme_common/common_geo_attr.h"

namespace gaudi
{

class GaudiGeoAttr : public MmeCommon::CommonGeoAttr
{
public:
    GaudiGeoAttr(const MmeCommon::MmeLayerParams& params) : CommonGeoAttr(params, MmeHalReader::getInstance())
    {
        init();
    };
    virtual ~GaudiGeoAttr() = default;

    virtual unsigned getAccHeight() const override;  //  ACC height in rows
    virtual unsigned getEuWidth() const override;  //  EU width in elements
    virtual unsigned getEuHeight() const override;  //  EU height in elements
    virtual unsigned getMmeWidth() const override;  //  MMME width in elements, geometry dependant
    virtual unsigned getMmeHeight() const override;  //  MME height in elements, geometry dependant
    virtual unsigned getPortSize(MmeCommon::EMmeInternalOperand operand) const override;  //  port size in elements
    virtual unsigned getTeHeight() const override;  //  number of rows in a single TE
    virtual unsigned getCoresPerMmeNr() const override;  // number of cores in a single MME
    virtual bool doPortAdvanceSpatially(MmeCommon::EMmeInternalOperand operand) const override;
    virtual bool getDoubleAccumsBit() const override {return false; };  //  gaudi doesnt support double accums

protected:
    virtual void setGrids() override;
    virtual void setChipConcurrency() override;
    virtual void setBgemmConcurrency() override;
    virtual void setDedwConcurrency() override;
    virtual void setCdConcurrency() override;

private:
    virtual bool isDimConcurrencyOptimizationSupported() const override;

    bool transC = false;
    void set2xwPorts();
    void set2xhPorts();
    unsigned getEuSize() const;
};

}  // namespace gaudi

#endif //MME__GAUDI_GEO_ATTR_H
