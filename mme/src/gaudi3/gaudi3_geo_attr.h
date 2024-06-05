#ifndef MME__GAUDI3_GEO_ATTR_H
#define MME__GAUDI3_GEO_ATTR_H

#include "gaudi3_mme_hal_reader.h"
#include "src/mme_common/common_geo_attr.h"

namespace gaudi3
{
class Gaudi3GeoAttr : public MmeCommon::CommonGeoAttr
{
public:
    Gaudi3GeoAttr(const MmeCommon::MmeLayerParams& params) : CommonGeoAttr(params, MmeHalReader::getInstance())
    {
        init();
    };
    virtual ~Gaudi3GeoAttr() = default;

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
    virtual bool getBgemmBit() const override;
    virtual bool getNonShareABit() const override;

protected:
    virtual void setGrids() override;
    void setGridsMme();
    void setGridsDma();
    virtual void setBgemmConcurrency() override;
    virtual void setDedwConcurrency() override;
    virtual void setCdConcurrency() override;
};
}  // namespace gaudi3

#endif //MME__GAUDI3_GEO_ATTR_H
