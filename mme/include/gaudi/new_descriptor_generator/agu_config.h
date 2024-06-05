#ifndef MME__GAUDI_AGU_CONFIG_H
#define MME__GAUDI_AGU_CONFIG_H

#include <memory>
#include "include/gaudi/new_descriptor_generator/mme_common.h"

namespace gaudi
{
class GeoAttr;
class GaudiGeoAttr;

class AguConfig
{
public:
    AguConfig(const MmeCommon::MmeRecipe& recipe,
              const MmeCommon::MmeLayerParams& params,
              const gaudiReuseAttr& reuseAttr,
              GeoAttr::UseDataTypeForAttributeCalculation useDataType =
                  GeoAttr::UseDataTypeForAttributeCalculation::InputDataType);

    virtual ~AguConfig() = default;
    virtual void setAgu(DescGroup& descGroup) = 0;

protected:
    static const unsigned c_operand_max_agu = 4;
    static const unsigned c_padded_conv_dim = Mme::c_mme_max_conv_dims - 1;

    virtual void setNumIterationsMinus1(Mme::Desc& localDesc) = 0;
    virtual void setPartialHeightLoops(Mme::Desc* descPtr,
                                       const std::array<unsigned, c_operand_max_agu>& loopS,
                                       const std::array<unsigned, c_operand_max_agu>& loopL,
                                       const std::array<unsigned, c_operand_max_agu>& loopO,
                                       unsigned descIdx);

    const MmeCommon::MmeLayerParams& getParams() { return m_params; }
    const std::shared_ptr<GeoAttr> getGeoAttr() { return m_geoAttrSPtr; }
    const gaudiReuseAttr getReuseAttr() { return m_reuseAttr; }
    const MmeCommon::MmeRecipe& getRecipe() { return m_recipe; }

    const MmeCommon::MmeLayerParams& m_params;
    const MmeCommon::MmeRecipe& m_recipe;
    const std::shared_ptr<GeoAttr> m_geoAttrSPtr;
    const std::shared_ptr<GaudiGeoAttr> m_newGeoAttrSPtr;
    const gaudiReuseAttr& m_reuseAttr;
};

// AguConfig classes configure the tensor part of the descriptor (in firstOfAgu methods)
// and AGU part of the descriptor (firstOfAgu & restOfAgu methods).

class AguBGemmConfig : public AguConfig
{
public:
    AguBGemmConfig(const MmeCommon::MmeRecipe& recipe,
                   const MmeCommon::MmeLayerParams& params,
                   const gaudiReuseAttr& reuseAttr,
                   bool isZeroCD)
    : AguConfig(recipe, params, reuseAttr), m_zeroCD(isZeroCD)
    {
    }
    void setAgu(DescGroup& descGroup) override;

protected:
    void setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& localDesc);
    void setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& localDesc);
    void setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, bool isSouth, Mme::Desc& localDesc);
    void setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, bool isSouth, Mme::Desc& localDesc);

    void setNumIterationsMinus1(Mme::Desc& localDesc) override { localDesc.numIterationsMinus1 = 0; }

    void setDescFirstAguATranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc* desc);
    void setDescFirstAguANonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc* desc);
    void
    setDescRestAguATranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, bool isSouth, Mme::Desc* desc);
    void setDescRestAguANonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                     bool isSouth,
                                     Mme::Desc* desc);
    void setDescFirstAguBNonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc* desc);
    void setDescFirstAguBTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc* desc);
    void
    setDescRestAguBTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, bool isSouth, Mme::Desc* desc);
    void setDescRestAguBNonTranspose(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                     bool isSouth,
                                     Mme::Desc* desc);
    void setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, bool isSouth, Mme::Desc* desc);
    void setDescFirstOfAguCoutAndBatchLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                            std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                            std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                            Mme::Desc* desc);

    void doubleWorkForBatchMode(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                bool isSouth,
                                Mme::Desc* desc);
    void doubleWorkForBatchMode2xw(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                                   std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                                   std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                                   bool isSouth,
                                   Mme::Desc* desc);
    void setAssociatedDims(Mme::Desc& desc);
    void setDescAgusBgemm(Mme::Desc* descPtr,
                          const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                          const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                          const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                          unsigned descIdx);

    bool getZeroCD() { return m_zeroCD; }

private:
    const bool m_zeroCD = false;
};

class AguConvConfig : public AguConfig
{
public:
    AguConvConfig(const MmeCommon::MmeRecipe& recipe,
                  const MmeCommon::MmeLayerParams& params,
                  const gaudiReuseAttr& reuseAttr,
                  MmeRoi roi,
                  GeoAttr::UseDataTypeForAttributeCalculation dataType =
                      GeoAttr::UseDataTypeForAttributeCalculation::InputDataType)
    : AguConfig(recipe, params, reuseAttr, dataType), m_roi(roi)
    {
    }
    void setAgu(DescGroup& descGroup) override;

protected:
    void setNumIterationsMinus1(Mme::Desc& localDesc) override;

    virtual void setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& desc) = 0;
    virtual void setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& desc) = 0;
    virtual void setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& desc) = 0;
    virtual void setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& desc) = 0;
    virtual void setDescFirstOfAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, Mme::Desc& desc) = 0;
    virtual void setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, Mme::Desc& desc) = 0;
    virtual void setLoopDimsAndSizes(Mme::Desc& desc) = 0;

    virtual void setSpatialSize(Mme::Desc& localDesc, bool isMemsetDesc) = 0;
    virtual void setLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                          std::array<unsigned, c_operand_max_agu>& loopS,
                          std::array<unsigned, c_operand_max_agu>& loopL,
                          std::array<unsigned, c_operand_max_agu>& loopO) = 0;
    void setDescAgus(Mme::Desc* descPtr,
                     const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS,
                     const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL,
                     const std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                     unsigned descIdx);
    uint32_t* getSpPos(int dim) { return m_spPos[dim]; }
    uint32_t getSpPosB(int dim) { return m_spPosB[dim]; }
    MmeRoi getRoi() { return m_roi; }
    void setSpPos();
    void setSpPosB();

private:
    uint32_t m_spPos[c_operand_max_agu][Mme::c_mme_max_tensor_dims - 1];
    unsigned m_spPosB[Mme::c_mme_max_tensor_dims];  // index 0 is ignored

    MmeRoi m_roi;
};

class AguConvFwdDedxConfig : public AguConvConfig
{
public:
    AguConvFwdDedxConfig(const MmeCommon::MmeRecipe& recipe,
                         const MmeCommon::MmeLayerParams& params,
                         const gaudiReuseAttr& reuseAttr,
                         const MmeCommon::ConvSubProblem* convSubProblem,
                         MmeRoi roi,
                         GeoAttr::UseDataTypeForAttributeCalculation dataType =
                             GeoAttr::UseDataTypeForAttributeCalculation::InputDataType)
    : AguConvConfig(recipe, params, reuseAttr, roi, dataType), m_convSubProblem(convSubProblem)
    {
    }

protected:
    void setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& desc) override;
    void setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& desc) override;
    void setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& desc) override;
    void setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& desc) override;
    void setDescFirstOfAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, Mme::Desc& desc) override;
    void setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, Mme::Desc& desc) override;

    void setLoopDimsAndSizes(Mme::Desc& desc) override;
    void setSpatialSize(Mme::Desc& localDesc, bool isMemsetDesc) override;
    void setLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                  std::array<unsigned, c_operand_max_agu>& loopS,
                  std::array<unsigned, c_operand_max_agu>& loopL,
                  std::array<unsigned, c_operand_max_agu>& loopO) override;

private:
    const MmeCommon::ConvSubProblem* m_convSubProblem;
};

class AguConvDedwConfig : public AguConvConfig
{
public:
    AguConvDedwConfig(const MmeCommon::MmeRecipe& recipe,
                      const MmeCommon::MmeLayerParams& params,
                      const gaudiReuseAttr& reuseAttr,
                      MmeRoi roi,
                      GeoAttr::UseDataTypeForAttributeCalculation dataType =
                          GeoAttr::UseDataTypeForAttributeCalculation::InputDataType)
    : AguConvConfig(recipe, params, reuseAttr, roi, dataType)
    {
    }

protected:
    void setDescFirstAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& desc) override;
    void setDescRestAguA(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguS, Mme::Desc& desc) override;
    void setDescFirstAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& desc) override;
    void setDescRestAguB(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguL, Mme::Desc& desc) override;
    void setDescFirstOfAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, Mme::Desc& desc) override;
    void setDescRestAguCout(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO, Mme::Desc& desc) override;

    void setLoopDimsAndSizes(Mme::Desc& desc) override;
    void setSpatialSize(Mme::Desc& localDesc, bool isMemsetDesc) override;
    void setLoops(std::array<Mme::MmeAguCoreDesc, c_operand_max_agu>& aguO,
                  std::array<unsigned, c_operand_max_agu>& loopS,
                  std::array<unsigned, c_operand_max_agu>& loopL,
                  std::array<unsigned, c_operand_max_agu>& loopO) override;
    void setPartialHeightLoops(Mme::Desc* descPtr,
                               const std::array<unsigned, c_operand_max_agu>& loopS,
                               const std::array<unsigned, c_operand_max_agu>& loopL,
                               const std::array<unsigned, c_operand_max_agu>& loopO,
                               unsigned descIdx) override;
};

class AguMemsetConfig : public AguConvFwdDedxConfig
{
public:
    AguMemsetConfig(const MmeCommon::MmeRecipe& recipe,
                    const MmeCommon::MmeLayerParams& params,
                    const gaudiReuseAttr& reuseAttr,
                    const MmeCommon::ConvSubProblem* convSubProblem,
                    MmeRoi roi)
    : AguConvFwdDedxConfig(recipe,
                           params,
                           reuseAttr,
                           convSubProblem,
                           roi,
                           GeoAttr::UseDataTypeForAttributeCalculation::OutputDataType)
    {
    }
    void setAgu(DescGroup& descGroup) override;

protected:
    void resetTensorAgus(Mme::Desc& localDesc);
    void setTensorLRoiSize(Mme::Desc& localDesc);
};

} // namespace gaudi

#endif //MME__GAUDI_AGU_CONFIG_H
