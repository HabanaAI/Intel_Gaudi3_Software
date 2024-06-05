#ifndef MME__GAUDI_MME_COMMON_H
#define MME__GAUDI_MME_COMMON_H

#include "utils.h"
#include "include/gaudi/mme_agu_stat.h"
#include "include/mme_common/mme_descriptor_generator_base.h"

#define GAUDI_MME_C_CL_SIZE 128
#define swap_bf(a, b)                                                                                                  \
    {                                                                                                                  \
        auto ___t___ = (a);                                                                                            \
        (a) = (b);                                                                                                     \
        (b) = ___t___;                                                                                                 \
    }  // swap bit field
namespace gaudi
{
struct MmeSubTensorView
{
    MmeCommon::SizeArray bases = {0};  // the sub-view base coord in the tensor relative to tensor view B.
    MmeCommon::SizeArray sizes = {0};  // the actual size of the sub-view in elements.
    bool lower = false;
};

struct MmeRoi
{
    uint32_t size[Mme::c_mme_max_tensor_dims];
    uint32_t denseBase;
    uint32_t spBase;
    uint32_t spSize;
};

struct MmeSpatialSlice
{
    uint32_t spBase;
    uint32_t spSize;
};

union MmeDW_New
{
    float f32;
    uint32_t u32;
    uint16_t u16[2];
    uint8_t u8[4];
};

struct MmeSpatialSubTensorView
{
    unsigned viewBase = 0;
    unsigned viewSize = 0;
};
struct gaudiReuseAttr
{
    unsigned denseStepsNr = 0;  // Number of steps in the dense direction
    unsigned denseLoopSelector = 0;  // Bitmap in which bit that is set corresponds to the dense loop movement
    unsigned lastDenseStepSize = 0;  // Size of last step in the dense direction

    unsigned aPartialHeightStepsNr = 0;  // Number of full steps in the spatial (c/s) direction
    unsigned aPartialHeightLoopSelector = 0;  // Bitmap indicates the spatial movement that has a tail
    unsigned lastAPartialHeightStepSize = 0;  // Size of the last step in the spatial (c/s) direction

    unsigned spatialLoopSelector = 0;  // Bitmap in which bit that is set corresponds to the vertical loop movement
    unsigned spatialStepsNr = 0;  // Number of steps in the spatial direction for reuse
    // spatialStepsNr is expected to be equal aPartialHeightStepsNr * (steps on the filter)
    unsigned accumDim = 0;  // Bitmap indications all the loops that are summed in the EU
    // In Conv, all first 3 loops are summed up
    // In Bgemm, no loop is summed up (som the bitmap is 0)

    bool reuseA = false;
    bool reuseB = false;
};

struct MmeBGemmRecipe
{
    bool raster;
    MmeCommon::MmeTensorView aView;
    MmeCommon::MmeTensorView bView;
    MmeCommon::MmeTensorView cView;
    unsigned roiSizes[Mme::c_mme_max_tensor_dims];
    std::vector<MmeSubTensorView> batchSubviews;
    std::vector<MmeSpatialSubTensorView> fcdDimSubviews;
    std::vector<MmeSpatialSubTensorView> spSubviews;

    MmeCommon::EMmeOpType opType;
    MmeCommon::EMmePattern pattern;
    MmeCommon::EMmeGeometry geometry;
};

enum EPhysOperand
{
    OP_S = 0x0,
    OP_L = 0x1,
    OP_O = 0x2,
};

enum MmeBatchMode
{
    mme_batch_none = 0,
    mme_batch_2xw = 1,
    mme_batch_2xh = 2
};
//===== agu related =============
enum associatedDimPattern
{
    pattern_sp_kfc = 0x0104ff00,
    pattern_sp_fkc = 0x0201ff00,
    pattern_sp_fck = 0x0200ff01,
    pattern_sp_cfk = 0x0100ff04,
    pattern_sp_kcf = 0x0004ff03,
    pattern_sp_ckf = 0x0003ff04,
    pattern_z_ksf = 0x000403ff,
    pattern_z_skf = 0x000304ff,
};

enum LOOP
{
    LOOP_C = 0,
    LOOP_SPATIAL = 1,
    LOOP_K = 2,
    LOOP_FILTER = 3,
    LOOP_FILTER0 = LOOP_FILTER + 0,
    LOOP_FILTER1 = LOOP_FILTER + 1,
    LOOP_FILTER2 = LOOP_FILTER + 2,
};

// TODO Should be eventually removed, use MmeLayerParams instead
struct ExecuteParams
{
    MmeCommon::EMmeOpType opType;
    ptrToInt aPtr;
    ptrToInt bPtr;
    ptrToInt cPtr;
    const MmeCommon::MmeConv* conv = nullptr;
    const MmeCommon::MmeTensorView* a = nullptr;
    const MmeCommon::MmeTensorView* b = nullptr;
    const MmeCommon::MmeTensorView* c = nullptr;
    const MmeCommon::MmeControls* controls = nullptr;
    const MmeCommon::MmeStrategy* strategy = nullptr;
    const MmeRoi* roi = nullptr;  // for fwd and dedx
    const MmeSpatialSlice* spatialSlice = nullptr;  // for dedw only
    const MmeCommon::MmeTracing* tracing = nullptr;

    ExecuteParams(MmeCommon::EMmeOpType opType) : opType(opType) { aPtr.p = bPtr.p = cPtr.p = nullptr; }
};

struct DescGroup
{
    Mme::Desc desc[Mme::MME_MASTERS_NR];
};
typedef std::vector<DescGroup> DescList;

//========== Class definitions =================================

class GeoAttr
{
public:
    // which data type to use in the setting of the geo attributes
    enum class UseDataTypeForAttributeCalculation : bool
    {
        InputDataType,
        OutputDataType
    };

    GeoAttr(MmeCommon::MmeLayerParams params,
            const MmeCommon::MmeHalReader& mmeHalReader,
            UseDataTypeForAttributeCalculation useDataType = UseDataTypeForAttributeCalculation::InputDataType);

    unsigned m_subMatrixHeight = 0;
    unsigned m_subMatrixWidth = 0;
    unsigned m_matrixWidth = 0;
    unsigned m_matrixHeight = 0;

    unsigned m_cPortElemWidth = 0;
    unsigned m_portElemHeight = 0;
    unsigned m_inputPortElemWidth = 0;

    unsigned m_geoElemWidthPerPair = 0;
    unsigned m_geoElemHeightPerPair = 0;
    unsigned m_geoTotalElemWidth = 0;
    unsigned m_geoTotalElemHeight = 0;
    unsigned m_geoAportsTotalHeight = 0;

    unsigned m_totalAports = 0;
    unsigned m_totalBports = 0;
    unsigned m_euElemWidth = 0;
    unsigned m_euElemHeight = 0;

    unsigned m_concurrentLevel = 0;
    MmeBatchMode m_batchMode = mme_batch_none;

    bool m_isAInterleaved = true;

    bool isTransO() const { return m_transO; }
    const MmeCommon::MmeHalReader& getMmeHalReader() { return m_mmeHalReader; }

private:
    bool m_transO = false;
    const MmeCommon::MmeHalReader& m_mmeHalReader;
};

using MmeActivation = MmeCommon::MmeActivation<Mme::Desc>;
using ActivationList = std::list<MmeActivation>;
//========== Common function headers ==========================
unsigned countSignals(const Mme::Desc* desc);
uint8_t getLoopFromLoopMask(Mme::EMmeLoopMask mask);
void setOpStartAndEndEvents(const MmeCommon::MmeLayerParams* params,
                            std::list<MmeActivation>& activations,
                            const MmeCommon::EMmeOperand userOperand);
EPhysOperand mmeOperand2PhysOperand(const MmeCommon::EMmeOperand userOperand,
                                    const MmeCommon::EMmeOpType opType,
                                    const bool transposed);
void aguRanges2Rois(const unsigned signalBase, std::vector<AguRanges>* ranges, OverlapRoi& roi);
void generateRoi(MmeActivation& act,
                 MmeCommon::EMmeInternalOperand internalOperand,
                 MmeCommon::EMmeOpType opType,
                 std::shared_ptr<MmeCommon::CommonRoiCalculator<Mme::gaudi::_Desc>> roiCalc,
                 bool squashIORois);
Mme::EMmeDataType ConvertToGaudiDataType(MmeCommon::EMmeDataType dt);

void spPosToCoord(const unsigned spPos, const unsigned* spSizes, unsigned* spCoord);
unsigned spCoordToPos(const unsigned* spCoord, const unsigned* spSizes);

associatedDimPattern getConfigPattern(MmeCommon::EMmePattern pattern);

void setLoopDim(Mme::Desc* desc, MmeCommon::EMmePattern pattern, unsigned loop, EPhysOperand operand, unsigned dim);
void setLoopSize(Mme::Desc* desc, MmeCommon::EMmePattern pattern, unsigned loop, unsigned sizeMinus1);
unsigned getLoopSize(const Mme::Desc* desc, const MmeCommon::EMmePattern pattern, const unsigned loop);
unsigned getLoopMask(const MmeCommon::EMmePattern pattern, const unsigned loop);
void padRoi(const unsigned spatialAGUs, const MmeRoi* roiIn, MmeRoi* roiOut);

} // namespace gaudi

#endif //MME__GAUDI_MME_COMMON_H
