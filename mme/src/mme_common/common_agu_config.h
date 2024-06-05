#pragma once
#include "common_geo_attr.h"
#include "include/general_utils.h"
#include "include/mme_common/conv_sub_problems.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/mme_common/recipe.h"
#include "src/mme_common/mme_hal_reader.h"
#include "llvm/small_vector.h"

namespace MmeCommon
{
// parameters that describe a praticular tensor
// size of memory footprint
// amount of data that will be produced for the tensor
// this parameters are the same for all ports
struct TensorAttr
{
    unsigned roiSize[MAX_DIMENSION] = {0, 0, 0, 0, 0};
    unsigned validElements[MAX_DIMENSION] = {0, 0, 0, 0, 0};
    int loopStride[MAX_DIMENSION] = {0, 0, 0, 0, 0};
    unsigned spatialStrides[MAX_DIMENSION] = {0, 0, 0, 0, 0};
    int baseOffset[MAX_DIMENSION] = {0, 0, 0, 0, 0};
    int startOffset[MAX_DIMENSION] = {0, 0, 0, 0, 0};
    unsigned lastSpatialStep;
    unsigned lastFcdStep;
};

// parameters that are per port.
// basically only port offset
struct PortAttr
{
    int portOffset[MAX_DIMENSION] = {0};

    PortAttr& operator=(const PortAttr& other)
    {
        for (unsigned dim = 0; dim < MAX_DIMENSION; dim++)
        {
            this->portOffset[dim] = other.portOffset[dim];
        }
        return *this;
    }
};

//  the sizes of each dimension here will match the appropriate port Grid from geoAttr
template <class T>
using PortContainer = llvm_vecsmall::SmallVector<T, 3>;
using PortsComplex = PortContainer<PortContainer<PortContainer<PortContainer<PortContainer<PortAttr>>>>>;

class CommonAguConfig
{
public:
    CommonAguConfig(const MmeLayerParams& params,
                    const CommonGeoAttr& geoAttr,
                    const MmeHalReader& mmeHal,
                    unsigned mmeIdx,
                    const MmeCommon::MmeRecipe& recipe)
    : m_params(params), m_geoAttr(geoAttr), m_mmeHal(mmeHal), m_mmeIdx(mmeIdx), m_recipe(recipe)
    {
    }
    virtual ~CommonAguConfig() = default;

    //  configure the TensorAttr struct
    void config(void* descPtr);
    void setDescOffset(const OffsetArray& descAddrOffset) { m_descAddrOffset = descAddrOffset; }

protected:
    unsigned m_mmeIdx;  //  index of current MME unit

    TensorAttr tensorA;
    TensorAttr tensorB;
    TensorAttr tensorC;

    PortsComplex portsA;
    PortsComplex portsB;
    PortsComplex portsC;

    void configureDma();
    void configureTranspose();
    void configureGemmTranspose();
    void configureBgemm();
    void configureDedw();
    void configureFwd();
    void configureDedx();
    //  configure the portOffsets for all available ports that share the same work
    void setPortOffsets(EMmeInternalOperand operand);
    //  multiply spatial and loop strides according to number of reader and 2x optimizations
    void multiplyStrides(EMmeInternalOperand operand);
    //  multiply all sizes by the appropriate stride
    void finalizeSizes(EMmeInternalOperand operand);
    // incase we have sub-problems - need to adjust the agu offsets between descriptors.
    void fixDescOffsets(EMmeInternalOperand operand, TensorAttr& tensorAttr);
    //  confgure the chip specific descriptor according to the agu structs
    virtual void configureDescriptor(void* descPtr) = 0;

    void setAssociatedDimsDma(void* descPtr);
    void setAssociatedDimsBgemmDedw(void* descPtr);
    void setAssociatedDimsFwdDedx(void* descPtr);
    virtual void setAssociatedDimAndSize(MmeCommon::EMmeLoopMask mask,
                                         unsigned size,
                                         unsigned dimA,
                                         unsigned dimB,
                                         unsigned dimOut,
                                         void* descPtr) = 0;
    virtual void setSpatialLoopSize(unsigned size, void* descPtr) = 0;
    virtual void setPartialHeightLoopMaskA(unsigned mask, void* descPtr) = 0;
    virtual void setPartialHeightLoopMaskB(unsigned mask, void* descPtr) = 0;

    PortsComplex& getPortComplex(EMmeInternalOperand operand);
    TensorAttr& getTensor(EMmeInternalOperand operand);
    inline bool isMemsetDesc() const { return ConvSubProblemContainer::isMemsetDesc(m_params); }
    const MmeCommon::MmeRecipe& m_recipe;
    const MmeLayerParams& m_params;
    const CommonGeoAttr& m_geoAttr;
    const MmeHalReader& m_mmeHal;

private:
    unsigned getSinglePortFcdSize(EMmeInternalOperand operand);
    unsigned calcLastStepSize(EMmeInternalOperand operand, bool isSpatial, unsigned lastStepSize);
    void setPortSpatialOffset(EMmeInternalOperand operand, unsigned spatialIdx, PortAttr& curPort);
    void setPortBatchOffset(EMmeInternalOperand operand, unsigned spatialIdx, PortAttr& curPort);
    void setMmeOffset(EMmeInternalOperand operand);
    void setCoreOffset(EMmeInternalOperand operand, unsigned coreIdx, PortAttr& coreBasePort);
    void unitMatrixOffsets(PortsComplex& ports);
    void invalidateLogicalPort(PortAttr& coreBasePort);
    PortAttr getBasePort(EMmeInternalOperand operand);
    unsigned fixForTeAcceleration(EMmeInternalOperand operand, unsigned size, bool isSpatial = true);
    unsigned getPaddedCommonDim(const unsigned originalCommonDim, const EMmeDataType dataType);
    // transpose
    void configureUnitMatrix();
    // bGemm
    void configureBgemmTransposedA();
    void configureBgemmNonTransposedA();
    void configureBgemmTransposedB();
    void configureBgemmNonTransposedB();
    void configureBgemmOutput();
    void configureBatchLoops(EMmeInternalOperand operand);
    // conv
    void configureNonConvDimsTransposedB(unsigned paddedCommonDim);
    void configureNonConvDimsNonTransposedB(unsigned paddedCommonDim);
    bool isFilterDimReversed(MmeDimsIndex dim) const;
    OffsetArray m_descAddrOffset;
};
}  // namespace MmeCommon