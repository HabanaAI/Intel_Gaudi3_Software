//#define NDEBUG
#include <list>
#include <cmath>
#include <stdio.h>
#include "general_utils.h"
#include "include/mme_common/mme_common_enum.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "utils.h"
#include "mme_assert.h"
#include "include/gaudi/new_descriptor_generator/mme_common.h"

//#define MME_STACK_TRACE
#ifdef MME_STACK_TRACE
static unsigned __indent__ = 0;
#define TRACE_ENTER for (unsigned i=0; i<__indent__; i++) printf("  "); printf("Enter (%05u) - %s\n", __LINE__, __FUNCTION__); __indent__++
#define TRACE_EXIT __indent__--; for (unsigned i=0; i<__indent__; i++) printf("  ");printf("Exit  (%05u) - %s\n",  __LINE__, __FUNCTION__)
#else
#define TRACE_ENTER
#define TRACE_EXIT
#define TRACE_EXIT
#endif

#define mme_div_round_up(a, b) (((a) + (b) - 1) / (b))

using namespace MmeCommon;

namespace gaudi
{

#define set_bf_to_all_ones(bf)  {(bf)=1; (bf)=-(bf);}

static const unsigned c_operand_max_agu = 4;
static const unsigned c_padded_conv_dim = Mme::c_mme_max_conv_dims - 1;

typedef std::vector<DescGroup> DescList;

struct GeoParams
{
    unsigned matrixWidth;
    unsigned matrixHeight;
    unsigned aguAnr;
    unsigned aguBnr;
    bool     transO;
    unsigned subMatrixHeight;
    unsigned subMatrixWidth;
};

enum EReduction
{
    REDUCTION_NONE  = 0x000,
    REDUCTION_RN_BF = 0x00d,
    REDUCTION_RN_SP = 0x00f,
    REDUCTION_RZ_BF = 0x08d,
    REDUCTION_RZ_SP = 0x08f,
    REDUCTION_RU_BF = 0x10d,
    REDUCTION_RU_SP = 0x10f,
    REDUCTION_RD_BF = 0x18d,
    REDUCTION_RD_SP = 0x18f,
};

union MmeDW
{
    float f32;
    uint32_t u32;
    uint16_t u16[2];
    uint8_t  u8[4];
};

static void transposeDesc(Mme::Desc *desc)
{
    TRACE_ENTER;
    desc->header.transO = !desc->header.transO;

    std::swap(desc->baseAddrHighS, desc->baseAddrHighL);
    std::swap(desc->baseAddrLowS, desc->baseAddrLowL);
    swap_bf(desc->header.transS, desc->header.transL);
    swap_bf(desc->header.advanceS, desc->header.advanceL);
    swap_bf(desc->header.lowerS, desc->header.lowerL);
    for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        swap_bf(desc->conv.associatedDims[i].dimS, desc->conv.associatedDims[i].dimL);
    }
    swap_bf(desc->outerLoop.associatedDims.dimS, desc->outerLoop.associatedDims.dimL);
    std::swap(desc->tensorS, desc->tensorL);
    std::swap(desc->paddingValueS, desc->paddingValueL);
    TRACE_EXIT;
}

static void geo2Params(const EMmeGeometry geo, const EMmeDataType inDataType, GeoParams* params)
{
    TRACE_ENTER;
    switch (geo)
    {
        case e_mme_geometry_4wx1h:
            params->aguAnr = 1;
            params->aguBnr = 4;
            params->transO = false;
            break;
        case e_mme_geometry_2wx2h:
            params->aguAnr = 2;
            params->aguBnr = 2;
            params->transO = false;
            break;
        case e_mme_geometry_1wx4h:
            params->aguAnr = 4;
            params->aguBnr = 1;
            params->transO = true;
            break;
        default:
            MME_ASSERT(0, "invalid geometry");
            params->aguAnr = 0;
            params->aguBnr = 0;
            params->transO = false;
            break;  // Initialize fields to silent compile time warning
    }

    unsigned shiftAmount = getDataTypeShiftAmount(inDataType);
    params->subMatrixHeight = Mme::c_mme_matrix_size >> shiftAmount;
    params->subMatrixWidth = Mme::c_mme_matrix_size >> shiftAmount;

    params->matrixHeight = params->aguAnr * params->subMatrixHeight;
    params->matrixWidth = params->aguBnr * params->subMatrixWidth;
    TRACE_EXIT;
}

unsigned getUserDataVal(const bool reductionEn, const EMmeDataType dt, const RoundingMode rm)
{
    TRACE_ENTER;
    Mme::MmeUserData ud = {0};
    ud.first = REDUCTION_NONE;
    if (reductionEn)
    {
        switch (dt)
        {
            case EMmeDataType::e_type_bf16:
            {
                switch (rm)
                {
                    case RoundToNearest:
                        ud.first = REDUCTION_RN_BF;
                        break;
                    case RoundToZero:
                        ud.first = REDUCTION_RZ_BF;
                        break;
                    case RoundUp:
                        ud.first = REDUCTION_RU_BF;
                        break;
                    case RoundDown:
                        ud.first = REDUCTION_RD_BF;
                        break;
                    default:
                        MME_ASSERT(0, "invalid rounding");
                }
            }
            break;
            case EMmeDataType::e_type_fp32:
            {
                switch (rm)
                {
                    case RoundToNearest:
                        ud.first = REDUCTION_RN_SP;
                        break;
                    case RoundToZero:
                        ud.first = REDUCTION_RZ_SP;
                        break;
                    case RoundUp:
                        ud.first = REDUCTION_RU_SP;
                        break;
                    case RoundDown:
                        ud.first = REDUCTION_RD_SP;
                        break;
                    default:
                        MME_ASSERT(0, "invalid rounding");
                }
            }
            break;
            default:
                MME_ASSERT(0, "invalid data type");
        }
    }
    ud.steady = ud.first;
    TRACE_EXIT;
    return ud.dw;
}

#define MME_CMD_CLONE(_class_name_)                  \
    _class_name_ *ret = new _class_name_(*this);     \
    if (ret->m_next)                                 \
    {                                                \
        ret->m_next = ret->m_next->clone();          \
    }                                                \
    return ret;

class MmeStackCmd
{
public:
    MmeStackCmd() : m_next(0) {}

    virtual ~MmeStackCmd()
    {
        if (m_next) delete m_next;
    }

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd); }

    virtual void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT_PTR(m_next);
        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        m_next->execute(params, descList);
        TRACE_EXIT;
    }

    MmeStackCmd * setNext(MmeStackCmd *next)
    {
        TRACE_ENTER;
        if (next)
        {
            next->getLast()->m_next = m_next;
        }
        m_next = next;
        TRACE_EXIT;
        return next;
    }

    MmeStackCmd * getNext() { return m_next; }

    MmeStackCmd * setLast(MmeStackCmd *cmd)
    {
        TRACE_ENTER;
        getLast()->setNext(cmd);
        TRACE_EXIT;
        return cmd;
    }

    MmeStackCmd *getLast()
    {
        TRACE_ENTER;
        MmeStackCmd *last;
        for (last = this; last->m_next; last = last->m_next)
        {
            // do nothing
        }
        TRACE_EXIT;
        return last;
    }

protected:
    MmeStackCmd *m_next;
};

static void initDescPtrs(ExecuteParams *params, Mme::Desc * desc)
{
    ptrToInt aPtr = params->aPtr;
    ptrToInt bPtr = params->bPtr;
    ptrToInt cPtr = params->cPtr;

    //    unsigned aDataSize = (params->a->elementType == EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) :
    //    sizeof(Mme::f32_t); unsigned bDataSize = (params->b->elementType == EMmeDataType::e_type_bf16) ?
    //    sizeof(Mme::bf16_t) : sizeof(Mme::f32_t); unsigned cDataSize = (params->c->elementType ==
    //    EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);
    //
    //    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims; dim++)
    //    {
    //        aPtr.u64 += params->a->strides[dim] * params->a->bases[dim] * aDataSize;
    //        bPtr.u64 += params->b->strides[dim] * params->b->bases[dim] * bDataSize;
    //        cPtr.u64 += params->c->strides[dim] * params->c->bases[dim] * cDataSize;
    //    }

    desc->baseAddrHighS = aPtr.u32[1];
    desc->baseAddrLowS = aPtr.u32[0];
    desc->baseAddrHighL = bPtr.u32[1];
    desc->baseAddrLowL = bPtr.u32[0];
    desc->baseAddrHighO = cPtr.u32[1];
    desc->baseAddrLowO = cPtr.u32[0];
}

class MmeStackCmd_BuildDesc : public MmeStackCmd
{
public:
    MmeStackCmd_BuildDesc() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_BuildDesc); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        Mme::Desc localDesc;
        // Zeros reserved fields and makes the desc calculation deterministic
        memset(&localDesc, 0, sizeof(localDesc));

        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->a->strides[0] == 1, "FCD stride should be 1 in operand A");
        MME_ASSERT(params->b->strides[0] == 1, "FCD stride should be 1 in operand B");
        MME_ASSERT(params->c->strides[0] == 1, "FCD stride should be 1 in operand C");
        MME_ASSERT(params->a->elementType == params->b->elementType, "element type should match between operand A & B");
        MME_ASSERT(!params->controls->reluEn || (params->opType == e_mme_fwd), "expected reluEn=false if not in FWD.");

        GeoParams gp;
        geo2Params(params->strategy->geometry, params->a->elementType, &gp);

        MmeRoi paddedRoi;
        padRoi(gp.aguAnr, params->roi, &paddedRoi);

        uint32_t spPos[c_operand_max_agu][Mme::c_mme_max_tensor_dims - 1];
        for (unsigned i = 0; i < gp.aguAnr; i++)
        {
            // The original conv code should not be modified unless there is a bug.
            // However, in case we do not use descriptors produced by the original code, and
            // we want to compare the new descriptors to the original ones, we can make changes
            // required to match the descriptors.
            bool compareNewDescriptors = (getenv("ENABLE_COMPARE_FIRST_NEW_DESCRIPTORS") != nullptr) ||
                                         (getenv("ENABLE_COMPARE_ALL_NEW_DESCRIPTORS") != nullptr);
            bool useNewDescriptors = (getenv("ENABLE_USE_NEW_DESCRIPTORS") != nullptr);
            if (useNewDescriptors && compareNewDescriptors)
            {
                // use the padded Roi, similarly to the new code
                spPosToCoord(paddedRoi.spBase + i, &paddedRoi.size[1], spPos[i]);
            }
            else
            {
                // The original code
                spPosToCoord(params->roi->spBase + i, &params->roi->size[1], spPos[i]);
            }
        }

        localDesc.sw.dw = 0;
        localDesc.pcu.rlSaturation = 16*4096;
        localDesc.axiUserData.dw = getUserDataVal(
                params->controls->atomicAdd,
                params->c->elementType,
                params->controls->roundingMode);
        localDesc.perfEvtS.dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtS.startEndMask);
        localDesc.perfEvtL[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtL[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_remote].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_remote].startEndMask);
        localDesc.metaData.aguS = 0;
        localDesc.metaData.aguL[Mme::e_mme_local] = 0;
        localDesc.metaData.aguL[Mme::e_mme_remote] = 0;
        localDesc.metaData.aguO[Mme::e_mme_local] = 0;
        localDesc.metaData.aguO[Mme::e_mme_remote] = 0;

        localDesc.rateLimiter.aguL = 4;
        localDesc.rateLimiter.aguS = 4;
        localDesc.rateLimiter.aguO = 2;

        localDesc.sbRepeat.teEnS = 1;
        localDesc.sbRepeat.teEnL = 1;
        localDesc.sbRepeat.loadS = 1;
        localDesc.sbRepeat.loadL = 1;
        localDesc.sbRepeat.aguSLoopMask = 0;
        localDesc.sbRepeat.aguLLoopMask = 0;
        localDesc.sbRepeat.repeatSMinus1 = 0;
        localDesc.sbRepeat.repeatLMinus1 = 0;

        localDesc.header.fpEn = 1;
        localDesc.header.euBEn = 1;
        localDesc.header.transS = 1;
        localDesc.header.transL = (params->opType == e_mme_dedx);
        localDesc.header.lowerS = 0;
        localDesc.header.lowerL = 0;
        localDesc.header.transO = 0;
        localDesc.header.accumMask = (1 << c_padded_conv_dim) - 1;
        localDesc.header.storeEn = 1;
        localDesc.header.advanceS = 1;
        localDesc.header.advanceL = 0;
        localDesc.header.advanceO = 1;
        localDesc.header.accStoreIncDisable = 0;
        localDesc.header.dataTypeIn = ConvertToGaudiDataType(params->a->elementType);
        localDesc.header.dataTypeOut = ConvertToGaudiDataType(params->c->elementType);
        localDesc.header.reluEn = params->controls->reluEn;
        localDesc.header.accum = 0;
        localDesc.header.rollAccums = 0;
        localDesc.header.roundingMode = params->controls->roundingMode;
        localDesc.header.signalEn = 0;
        localDesc.header.signalMask = 0;
        localDesc.syncObject.addrHigh = 0;
        localDesc.syncObject.addrLow[0] = 0;
        localDesc.syncObject.addrLow[1] = 0;
        localDesc.syncObject.operation = 1;
        localDesc.syncObject.value = 1;

        localDesc.numIterationsMinus1 = ((paddedRoi.spSize + gp.matrixHeight - 1) / gp.matrixHeight) - 1;

        for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
        {
            localDesc.conv.associatedDims[i].dimS = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimL = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimO = Mme::c_mme_max_tensor_dims;
        }
        localDesc.outerLoop.associatedDims.dimS = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimL = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimO = Mme::c_mme_max_tensor_dims;

        initDescPtrs(params, &localDesc);

        int spElementsPerAguA = (paddedRoi.spSize + gp.aguAnr - 1) / gp.aguAnr;
        int spatialSizeAC = spElementsPerAguA % gp.subMatrixHeight ? spElementsPerAguA % gp.subMatrixHeight : gp.subMatrixHeight;
        int spatialSizeB = params->b->sizes[1];
        if (params->opType == e_mme_dedx)
        {
            spatialSizeB %= gp.subMatrixWidth;
            spatialSizeB = spatialSizeB ? spatialSizeB : gp.subMatrixWidth;
        }

        localDesc.tensorS.spatialSizeMinus1 = spatialSizeAC - 1;
        localDesc.tensorO.spatialSizeMinus1 = spatialSizeAC - 1;
        localDesc.tensorL.spatialSizeMinus1 = spatialSizeB - 1;

        localDesc.conv.kernelSizeMinus1.dw = 0;
        localDesc.outerLoop.sizeMinus1 = 0;

        if (params->opType == e_mme_fwd)
        {
            MmeDW pv;
            pv.f32 = params->conv->paddingValue;
            if (params->a->elementType == EMmeDataType::e_type_bf16)
            {
                pv.u16[0] = pv.u16[1];
            }
            localDesc.paddingValueS = pv.u32;
        }
        else
        {
            localDesc.paddingValueS = 0;
        }

        localDesc.paddingValueL = 0;

        // set default values
        Mme::MmeAguCoreDesc aguS[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguL[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguO[c_operand_max_agu] = { 0 };


        for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
        {

            localDesc.tensorS.loopStride[i] = 0;
            localDesc.tensorL.loopStride[i] = 0;
            localDesc.tensorO.loopStride[i] = 0;

            localDesc.tensorS.validElements[i] = 1;
            localDesc.tensorL.validElements[i] = 1;
            localDesc.tensorO.validElements[i] = 1;

            if (i < Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[i] = 1;
                localDesc.tensorL.roiSize[i] = 1;
                localDesc.tensorO.roiSize[i] = 1;
            }

            if (i > 0)
            {
                localDesc.tensorS.spatialStrides[i - 1] = 1;
                localDesc.tensorL.spatialStrides[i - 1] = 1;
                localDesc.tensorO.spatialStrides[i - 1] = 1;
            }
        }

        // tensor A: (dim 0)
        localDesc.tensorS.validElements[0] = params->a->sizes[0];
        localDesc.tensorS.roiSize[0] = params->a->sizes[0];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[0] = 0;
        }

        // tensor C: (dim 0)
        localDesc.tensorO.validElements[0] = params->c->sizes[0];
        localDesc.tensorO.roiSize[0] = params->roi->size[0] % gp.subMatrixWidth;
        localDesc.tensorO.roiSize[0] = localDesc.tensorO.roiSize[0] ? localDesc.tensorO.roiSize[0] : gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[0] = ((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi->denseBase - params->c->bases[0];
        }

        // tensor B: K (dim 0)
        localDesc.tensorL.validElements[0] = params->b->sizes[0];
        localDesc.tensorL.roiSize[0] = localDesc.header.transL ? params->b->sizes[0] : (params->roi->size[0] % gp.subMatrixWidth);
        localDesc.tensorL.roiSize[0] = localDesc.tensorL.roiSize[0] ? localDesc.tensorL.roiSize[0] : gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[0] = localDesc.header.transL ? 0 : (((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi->denseBase);
            // TODO: align partial ROI of A and B using the base
            //aguL[j].roiBaseOffset[0] -= params->b->bases[0];
        }

        // tensor B: C (dim 1)
        localDesc.tensorL.validElements[1] = params->b->sizes[1] * params->b->strides[1];
        localDesc.tensorL.roiSize[1] = params->b->strides[1] * (localDesc.header.transL ? std::min(gp.subMatrixWidth, params->roi->size[0]) : params->b->sizes[1]);
        localDesc.tensorL.spatialStrides[0] = params->b->strides[1];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[1] = localDesc.header.transL ? (((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi->denseBase) : 0;
            // TODO: align partial ROI of A and B using the base
            //aguL[j].roiBaseOffset[1] -= params->b->bases[1];
            aguL[j].roiBaseOffset[1] *= params->b->strides[1];
        }

        unsigned loopS[c_operand_max_agu];
        unsigned loopL[c_operand_max_agu];
        unsigned loopO[c_operand_max_agu];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            loopS[j] = getLoopMask(params->strategy->pattern, LOOP_SPATIAL);
            unsigned rem = params->roi->size[0] % gp.matrixWidth;
            rem = rem ? rem : gp.matrixWidth;
            loopL[j] = ((rem + params->roi->denseBase - aguO[j].roiBaseOffset[0]) < gp.subMatrixWidth) ? getLoopMask(params->strategy->pattern, LOOP_K) : 0;
            loopO[j] = ((rem + params->roi->denseBase - aguO[j].roiBaseOffset[0]) < gp.subMatrixWidth) ? getLoopMask(params->strategy->pattern, LOOP_K) : 0;
        }


        for (int dim = 0; dim < c_padded_conv_dim; dim++)
        {
            int ioDim = dim + 1;
            int wDim = dim + 2;

            // tensor A: spatial
            localDesc.tensorS.validElements[ioDim] = params->a->sizes[ioDim] * params->a->strides[ioDim];
            localDesc.tensorS.roiSize[ioDim] = params->conv->stride[dim] * paddedRoi.size[ioDim] * params->a->strides[ioDim];
            localDesc.tensorS.loopStride[ioDim] = params->conv->dilation[dim] * params->a->strides[ioDim];
            localDesc.tensorS.spatialStrides[ioDim - 1] = params->conv->stride[dim] * params->a->strides[ioDim] * (dim ? 1 : gp.aguAnr);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                if (params->opType == e_mme_dedx)
                {
                    aguS[j].roiBaseOffset[ioDim] = -(params->b->sizes[wDim] + params->b->bases[wDim] - 1);
                    aguS[j].roiBaseOffset[ioDim] *= params->conv->dilation[dim];
                    aguS[j].roiBaseOffset[ioDim] += params->conv->padding[dim];
                }
                else
                {
                    aguS[j].roiBaseOffset[ioDim] = params->b->bases[wDim];
                    aguS[j].roiBaseOffset[ioDim] *= params->conv->dilation[dim];
                    aguS[j].roiBaseOffset[ioDim] -= params->conv->padding[dim];
                }
                aguS[j].roiBaseOffset[ioDim] += (params->conv->stride[dim] * params->c->bases[ioDim]);
                aguS[j].roiBaseOffset[ioDim] -= params->a->bases[ioDim];
                aguS[j].roiBaseOffset[ioDim] *= params->a->strides[ioDim];

                aguS[j].startOffset[ioDim - 1] = spPos[j % gp.aguAnr][dim] * params->conv->stride[dim] * params->a->strides[ioDim];
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_S, ioDim);

            // tensor B: QRS
            localDesc.tensorL.validElements[wDim] = params->b->sizes[wDim] * params->b->strides[wDim];
            if (wDim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorL.roiSize[wDim] = params->b->strides[wDim];
            }
            localDesc.tensorL.loopStride[wDim] =
                (params->opType == e_mme_dedx) ? -params->b->strides[wDim] : params->b->strides[wDim];
            localDesc.tensorL.spatialStrides[wDim - 1] = params->b->strides[wDim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguL[j].roiBaseOffset[wDim] = (params->opType == e_mme_dedx) ? (params->b->sizes[wDim] - 1) : 0;
                aguL[j].roiBaseOffset[wDim] *= params->b->strides[wDim];
                aguL[j].startOffset[wDim - 1] = 0;
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_L, wDim);
            setLoopSize(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, params->b->sizes[wDim] - 1);

            // tensor C: spatial loops
            localDesc.tensorO.validElements[ioDim] = params->c->sizes[ioDim] * params->c->strides[ioDim];
            localDesc.tensorO.roiSize[ioDim] = paddedRoi.size[ioDim] * params->c->strides[ioDim];
            localDesc.tensorO.loopStride[ioDim] = 0;
            localDesc.tensorO.spatialStrides[ioDim - 1] = params->c->strides[ioDim] * (dim ? 1 : gp.aguAnr);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].startOffset[dim] = spPos[j / gp.aguBnr][dim] * params->c->strides[ioDim];
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_O, ioDim);
        }

        for (int dim = c_padded_conv_dim + 1; dim < Mme::c_mme_max_tensor_dims; dim++)
        {
            // tensor A: outer spatial loops
            localDesc.tensorS.validElements[dim] = (params->a->sizes[dim] - params->a->bases[dim]) * params->a->strides[dim];
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[dim] = params->a->sizes[dim]* params->a->strides[dim];
            }
            localDesc.tensorS.spatialStrides[dim - 1] = params->a->strides[dim];
            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguS[j].roiBaseOffset[dim] = 0;
                //aguS[j].roiBaseOffset[dim] = params->a->bases[dim];
                aguS[j].startOffset[dim - 1] = spPos[j % gp.aguAnr][dim - 1] * params->a->strides[dim];
            }

            // tensor C: spatial loops
            localDesc.tensorO.validElements[dim] = (params->c->sizes[dim] - params->c->bases[dim]) * params->c->strides[dim];
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorO.roiSize[dim] = paddedRoi.size[dim] * params->c->strides[dim];
            }
            localDesc.tensorO.spatialStrides[dim - 1] = params->c->strides[dim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].roiBaseOffset[dim] = 0;
                aguO[j].startOffset[dim - 1] = spPos[j / gp.aguBnr][dim - 1] * params->c->strides[dim];
            }
        }

        if (params->opType == e_mme_dedx)
        {
            localDesc.tensorL.loopStride[1] = gp.matrixWidth * params->b->strides[1];
            setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_L, 1);
        }
        else
        {
            localDesc.tensorL.loopStride[0] = gp.matrixWidth;
            setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_L, 0);
        }

        localDesc.tensorS.loopStride[0] = 0;
        localDesc.tensorO.loopStride[0] = gp.matrixWidth;
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_O, 0);
        unsigned denseLoopSize = div_round_up(params->roi->size[0], gp.matrixWidth) - 1;
        setLoopSize(&localDesc, params->strategy->pattern, LOOP_K, denseLoopSize);

        if (gp.transO)
        {
            transposeDesc(&localDesc);
            std::swap(aguS, aguL);
            std::swap(gp.aguAnr, gp.aguBnr);
            std::swap(loopS, loopL);
        }

        DescGroup descGroup;
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            descGroup.desc[descIdx] = localDesc;
            descGroup.desc[descIdx].aguS = aguS[descIdx];
            descGroup.desc[descIdx].header.partialHeightLoopS = loopS[descIdx];
            localDesc.header.partialHeightLoopS = getLoopMask(params->strategy->pattern, LOOP_SPATIAL);

            for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
            {
                descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                if (aguIdx == Mme::e_mme_local)
                {
                    descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                    descGroup.desc[descIdx].header.partialHeightLoopOLocal = loopO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                }
                else
                {
                    descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                    descGroup.desc[descIdx].header.partialHeightLoopORemote = loopO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                }
            }
        }

        descList->push_back(descGroup);
        TRACE_EXIT;
    }
};


class MmeStackCmd_BuildDEDWDesc : public MmeStackCmd
{
public:
    MmeStackCmd_BuildDEDWDesc() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_BuildDEDWDesc); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->a->strides[0] == 1, "FCD stride should be 1 in operand A");
        MME_ASSERT(params->b->strides[0] == 1, "FCD stride should be 1 in operand B");
        MME_ASSERT(params->c->strides[0] == 1, "FCD stride should be 1 in operand C");
        MME_ASSERT(params->a->elementType == params->b->elementType, "element type should match between operand A & B");

        GeoParams gp;
        geo2Params(params->strategy->geometry, params->a->elementType, &gp);

        Mme::Desc localDesc;
        // Zeros reserved fields and makes the desc calculation deterministic
        memset(&localDesc, 0, sizeof(localDesc));

        unsigned spPosB[Mme::c_mme_max_tensor_dims]; // index 0 is ignored
        spPosToCoord(params->spatialSlice->spBase, &params->b->sizes[1], &spPosB[1]);

        localDesc.sw.dw = 0;
        localDesc.pcu.rlSaturation = 16 * 4096;
        localDesc.axiUserData.dw = getUserDataVal(
                params->controls->atomicAdd,
                params->c->elementType,
                params->controls->roundingMode);
        localDesc.perfEvtS.dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtS.startEndMask);
        localDesc.perfEvtL[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtL[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_remote].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_remote].startEndMask);
        localDesc.metaData.aguS = 0;
        localDesc.metaData.aguL[Mme::e_mme_local] = 0;
        localDesc.metaData.aguL[Mme::e_mme_remote] = 0;
        localDesc.metaData.aguO[Mme::e_mme_local] = 0;
        localDesc.metaData.aguO[Mme::e_mme_remote] = 0;

        localDesc.rateLimiter.aguL = 4;
        localDesc.rateLimiter.aguS = 4;
        localDesc.rateLimiter.aguO = 2;

        localDesc.sbRepeat.teEnS = 1;
        localDesc.sbRepeat.teEnL = 1;
        localDesc.sbRepeat.loadS = 1;
        localDesc.sbRepeat.loadL = 1;
        localDesc.sbRepeat.aguSLoopMask = 0;
        localDesc.sbRepeat.aguLLoopMask = 0;
        localDesc.sbRepeat.repeatSMinus1 = 0;
        localDesc.sbRepeat.repeatLMinus1 = 0;

        localDesc.header.euBEn = 1;
        localDesc.header.fpEn = 1;
        localDesc.header.transO = 0;
        localDesc.header.transS = 0;
        localDesc.header.transL = 0;
        localDesc.header.lowerS = 0;
        localDesc.header.lowerL = 0;
        localDesc.header.accumMask = 0;
        localDesc.header.storeEn = 1;
        localDesc.header.advanceS = 0;
        localDesc.header.advanceL = 0;
        localDesc.header.advanceO = 0;
        localDesc.header.accStoreIncDisable = 0;
        localDesc.header.dataTypeIn = ConvertToGaudiDataType(params->a->elementType);
        localDesc.header.dataTypeOut = ConvertToGaudiDataType(params->c->elementType);
        localDesc.header.reluEn = 0;
        localDesc.header.accum = 0;
        localDesc.header.rollAccums = 0;
        localDesc.header.roundingMode =
            (int) params->controls->roundingMode | (int) params->controls->conversionRoundingMode;
        localDesc.header.signalEn = 0;
        localDesc.header.signalMask = 0;

        for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
        {
            localDesc.conv.associatedDims[i].dimS = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimL = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimO = Mme::c_mme_max_tensor_dims;
        }
        localDesc.outerLoop.associatedDims.dimS = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimL = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimO = Mme::c_mme_max_tensor_dims;

        initDescPtrs(params, &localDesc);

        localDesc.tensorS.spatialSizeMinus1 = params->spatialSlice->spSize - 1;
        localDesc.tensorL.spatialSizeMinus1 = params->spatialSlice->spSize - 1;
        localDesc.tensorO.spatialSizeMinus1 = std::min(params->c->sizes[1], gp.subMatrixHeight) - 1;

        localDesc.paddingValueL = 0;
        localDesc.paddingValueS = 0;

        localDesc.syncObject.addrHigh = 0;
        localDesc.syncObject.addrLow[0] = 0;
        localDesc.syncObject.addrLow[1] = 0;
        localDesc.syncObject.operation = 1;
        localDesc.syncObject.value = 1;

        // set default values
        Mme::MmeAguCoreDesc aguS[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguL[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguO[c_operand_max_agu] = { 0 };

        localDesc.outerLoop.sizeMinus1 = 0;
        localDesc.numIterationsMinus1 = 0;
        localDesc.conv.kernelSizeMinus1.dw = 0;

        for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
        {

            localDesc.tensorS.loopStride[i] = 0;
            localDesc.tensorL.loopStride[i] = 0;
            localDesc.tensorO.loopStride[i] = 0;

            localDesc.tensorS.validElements[i] = 1;
            localDesc.tensorL.validElements[i] = 1;
            localDesc.tensorO.validElements[i] = 1;

            if (i < Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[i] = 1;
                localDesc.tensorL.roiSize[i] = 1;
                localDesc.tensorO.roiSize[i] = 1;
            }

            if (i > 0)
            {
                localDesc.tensorS.spatialStrides[i - 1] = 1;
                localDesc.tensorL.spatialStrides[i - 1] = 1;
                localDesc.tensorO.spatialStrides[i - 1] = 1;
            }
        }

        // tensor A: (dim 0)
        localDesc.tensorS.validElements[0] = params->a->sizes[0];
        localDesc.tensorS.roiSize[0] = std::min(params->c->sizes[1], gp.subMatrixHeight);
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[0] = (j % gp.aguAnr) * gp.subMatrixHeight;
        }

        // tensor B: (dim 0)
        localDesc.tensorL.validElements[0] = params->b->sizes[0];
        localDesc.tensorL.roiSize[0] = gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[0] = (j % gp.aguBnr) * gp.subMatrixWidth;
        }

        // tensor C: (dim 0)
        localDesc.tensorO.validElements[0] = params->c->sizes[0];
        localDesc.tensorO.roiSize[0] = gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[0] = (j % gp.aguBnr) * gp.subMatrixWidth;
        }

        // tensor C: (dim 1)
        localDesc.tensorO.validElements[1] = params->c->sizes[1] * params->c->strides[1];
        localDesc.tensorO.roiSize[1] = std::min(params->c->sizes[1], gp.subMatrixHeight) * params->c->strides[1];
        localDesc.tensorO.spatialStrides[0] = params->c->strides[1];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[1] = (j / gp.aguBnr) * gp.subMatrixHeight * params->c->strides[1];
        }

        for (int dim = 0; dim < c_padded_conv_dim; dim++)
        {
            int ioDim = dim + 1;
            int wDim = dim + 2;

            // tensor A: spatial
            localDesc.tensorS.validElements[ioDim] = params->a->sizes[ioDim] * params->a->strides[ioDim];
            localDesc.tensorS.roiSize[ioDim] = params->conv->stride[dim] * params->b->sizes[ioDim] * params->a->strides[ioDim];
            localDesc.tensorS.loopStride[ioDim] = params->conv->dilation[dim] * params->a->strides[ioDim];
            localDesc.tensorS.spatialStrides[ioDim - 1] = params->conv->stride[dim] * params->a->strides[ioDim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguS[j].roiBaseOffset[ioDim] = -params->conv->padding[dim] * params->a->strides[ioDim];
                aguS[j].startOffset[ioDim - 1] = spPosB[ioDim] * params->conv->stride[dim] * params->a->strides[ioDim];
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_S, ioDim);

            // tensor B: spatial
            localDesc.tensorL.validElements[ioDim] = params->b->sizes[ioDim] * params->b->strides[ioDim];
            localDesc.tensorL.roiSize[ioDim] = params->b->sizes[ioDim] * params->b->strides[ioDim];
            localDesc.tensorL.loopStride[ioDim] = 0;
            localDesc.tensorL.spatialStrides[ioDim - 1] = params->b->strides[ioDim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguL[j].roiBaseOffset[ioDim] = 0;
                aguL[j].startOffset[ioDim - 1] = spPosB[ioDim] * params->b->strides[ioDim];
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_L, ioDim);

            // tensor C: kernel loops
            localDesc.tensorO.validElements[wDim] = params->c->sizes[wDim] * params->c->strides[wDim];
            if (wDim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorO.roiSize[wDim] = params->c->strides[wDim];
            }
            localDesc.tensorO.loopStride[wDim] = params->c->strides[wDim];
            localDesc.tensorO.spatialStrides[wDim - 1] = params->c->strides[wDim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].roiBaseOffset[wDim] = 0;
                aguO[j].startOffset[wDim - 1] = 0;
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_O, wDim);
            setLoopSize(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, params->c->sizes[wDim] - 1);
        }

        for (int dim = c_padded_conv_dim + 1; dim < Mme::c_mme_max_tensor_dims; dim++)
        {
            // tensor A: outer spatial loops
            localDesc.tensorS.validElements[dim] = params->a->sizes[dim]* params->a->strides[dim];
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[dim] = params->a->sizes[dim]* params->a->strides[dim];
            }
            localDesc.tensorS.spatialStrides[dim - 1] = params->a->strides[dim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguS[j].roiBaseOffset[dim] = 0;
                aguS[j].startOffset[dim - 1] = spPosB[dim] * params->a->strides[dim];
            }

            // tensor B: outer spatial loops
            localDesc.tensorL.validElements[dim] = params->b->sizes[dim]* params->b->strides[dim];
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorL.roiSize[dim] = params->b->sizes[dim]* params->b->strides[dim];
            }
            localDesc.tensorL.spatialStrides[dim - 1] = params->b->strides[dim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguL[j].roiBaseOffset[dim] = 0;
                aguL[j].startOffset[dim - 1] = spPosB[dim] * params->b->strides[dim];
            }
        }

        localDesc.tensorS.loopStride[0] = gp.matrixHeight;
        localDesc.tensorL.loopStride[0] = gp.matrixWidth;
        localDesc.tensorO.loopStride[0] = gp.matrixWidth;
        localDesc.tensorO.loopStride[1] = gp.matrixHeight * params->c->strides[1];

        unsigned loopSizeK = ((params->c->sizes[0] + gp.matrixWidth - 1) / gp.matrixWidth) - 1;
        setLoopSize(&localDesc, params->strategy->pattern, LOOP_K, loopSizeK);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_S, Mme::c_mme_max_tensor_dims);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_L, 0);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_O, 0);

        unsigned loopSizeC = ((params->c->sizes[1] + gp.matrixHeight - 1) / gp.matrixHeight) - 1;
        setLoopSize(&localDesc, params->strategy->pattern, LOOP_C, loopSizeC);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_C, OP_S, 0);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_C, OP_L, Mme::c_mme_max_tensor_dims);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_C, OP_O, 1);

        unsigned loopS = getLoopMask(params->strategy->pattern, LOOP_C);
        unsigned loopL = getLoopMask(params->strategy->pattern, LOOP_K);

        if (gp.transO)
        {
            transposeDesc(&localDesc);
            std::swap(aguS, aguL);
            std::swap(gp.aguAnr, gp.aguBnr);
            std::swap(loopS, loopL);
        }

        DescGroup descGroup;
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            descGroup.desc[descIdx] = localDesc;
            descGroup.desc[descIdx].aguS = aguS[descIdx];
            descGroup.desc[descIdx].header.partialHeightLoopS = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopL;
            descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopL;
            for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
            {
                descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
            }
        }

        descList->push_back(descGroup);
        TRACE_EXIT;
    }
};

class MmeStackCmd_BuildMemsetDesc : public MmeStackCmd
{
public:
    MmeStackCmd_BuildMemsetDesc() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_BuildMemsetDesc); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        Mme::Desc localDesc;
        // Zeros reserved fields and makes the desc calculation deterministic
        memset(&localDesc, 0, sizeof(localDesc));

        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->c->strides[0] == 1, "FCD stride should be 1 in operand C");
        MME_ASSERT(!params->controls->reluEn || (params->opType == e_mme_fwd), "expected reluEn = false if not in FWD");

        GeoParams gp;
        geo2Params(params->strategy->geometry, params->c->elementType, &gp);

        MmeRoi paddedRoi;
        padRoi(gp.aguAnr, params->roi, &paddedRoi);

        uint32_t spPos[c_operand_max_agu][Mme::c_mme_max_tensor_dims - 1];
        for (unsigned i = 0; i < gp.aguAnr; i++)
        {
            spPosToCoord(params->roi->spBase + i, &params->roi->size[1], spPos[i]);
        }

        localDesc.sw.reserved0 = 0;
        localDesc.sw.swMemsetFwd = (params->opType == e_mme_fwd);
        localDesc.sw.swMemsetDedx = (params->opType == e_mme_dedx);

        localDesc.pcu.rlSaturation = 16*4096;
        localDesc.axiUserData.dw = getUserDataVal(
                params->controls->atomicAdd,
                params->c->elementType,
                params->controls->roundingMode);
        localDesc.perfEvtS.dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtS.startEndMask);
        localDesc.perfEvtL[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtL[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_remote].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_remote].startEndMask);
        localDesc.metaData.aguS = 0;
        localDesc.metaData.aguL[Mme::e_mme_local] = 0;
        localDesc.metaData.aguL[Mme::e_mme_remote] = 0;
        localDesc.metaData.aguO[Mme::e_mme_local] = 0;
        localDesc.metaData.aguO[Mme::e_mme_remote] = 0;

        localDesc.rateLimiter.aguL = 4;
        localDesc.rateLimiter.aguS = 4;
        localDesc.rateLimiter.aguO = 2;

        localDesc.sbRepeat.teEnS = 1;
        localDesc.sbRepeat.teEnL = 1;
        localDesc.sbRepeat.loadS = 1;
        localDesc.sbRepeat.loadL = 1;
        localDesc.sbRepeat.aguSLoopMask = 0;
        localDesc.sbRepeat.aguLLoopMask = 0;
        localDesc.sbRepeat.repeatSMinus1 = 0;
        localDesc.sbRepeat.repeatLMinus1 = 0;

        localDesc.header.fpEn = 1;
        localDesc.header.euBEn = 1;
        localDesc.header.transS = 1;
        localDesc.header.transL = 0;
        localDesc.header.lowerS = 0;
        localDesc.header.lowerL = 0;
        localDesc.header.transO = 0;
        localDesc.header.accumMask = (1 << c_padded_conv_dim) - 1;
        localDesc.header.storeEn = 1;
        localDesc.header.advanceS = 0;
        localDesc.header.advanceL = 0;
        localDesc.header.advanceO = 1;
        localDesc.header.accStoreIncDisable = 0;
        localDesc.header.dataTypeIn = ConvertToGaudiDataType(params->c->elementType);
        localDesc.header.dataTypeOut = ConvertToGaudiDataType(params->c->elementType);
        localDesc.header.reluEn = params->controls->reluEn;
        localDesc.header.accum = 0;
        localDesc.header.rollAccums = 0;
        localDesc.header.roundingMode = Mme::e_mme_rm_rz;
        localDesc.header.signalEn = 0;
        localDesc.header.signalMask = 0;
        localDesc.syncObject.addrHigh = 0;
        localDesc.syncObject.addrLow[0] = 0;
        localDesc.syncObject.addrLow[1] = 0;
        localDesc.syncObject.operation = 1;
        localDesc.syncObject.value = 1;

        localDesc.numIterationsMinus1 = ((paddedRoi.spSize + gp.matrixHeight - 1) / gp.matrixHeight) - 1;

        for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
        {
            localDesc.conv.associatedDims[i].dimS = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimL = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimO = Mme::c_mme_max_tensor_dims;
        }
        localDesc.outerLoop.associatedDims.dimS = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimL = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimO = Mme::c_mme_max_tensor_dims;

        localDesc.baseAddrHighS = 0;
        localDesc.baseAddrLowS = 0;
        localDesc.baseAddrHighL = 0;
        localDesc.baseAddrLowL = 0;
        localDesc.baseAddrHighO = params->cPtr.u32[1];
        localDesc.baseAddrLowO = params->cPtr.u32[0];

        int spElementsPerAguA = (paddedRoi.spSize + gp.aguAnr - 1) / gp.aguAnr;
        int spatialSizeAC = spElementsPerAguA % gp.subMatrixHeight ? spElementsPerAguA % gp.subMatrixHeight : gp.subMatrixHeight;

        localDesc.tensorS.spatialSizeMinus1 = spatialSizeAC - 1;
        localDesc.tensorO.spatialSizeMinus1 = spatialSizeAC - 1;
        localDesc.tensorL.spatialSizeMinus1 = 0;

        localDesc.conv.kernelSizeMinus1.dw = 0;
        localDesc.outerLoop.sizeMinus1 = 0;

        localDesc.paddingValueS = 0;
        localDesc.paddingValueL = 0;

        // set default values
        Mme::MmeAguCoreDesc aguS[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguL[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguO[c_operand_max_agu] = { 0 };


        for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
        {

            localDesc.tensorS.loopStride[i] = 0;
            localDesc.tensorL.loopStride[i] = 0;
            localDesc.tensorO.loopStride[i] = 0;

            localDesc.tensorS.validElements[i] = 0;
            localDesc.tensorL.validElements[i] = 0;
            localDesc.tensorO.validElements[i] = 1;

            if (i < Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[i] = 1;
                localDesc.tensorL.roiSize[i] = 1;
                localDesc.tensorO.roiSize[i] = 1;
            }

            if (i > 0)
            {
                localDesc.tensorS.spatialStrides[i - 1] = 0;
                localDesc.tensorL.spatialStrides[i - 1] = 0;
                localDesc.tensorO.spatialStrides[i - 1] = 1;
            }
        }

        // tensor C: (dim 0)
        localDesc.tensorO.validElements[0] = params->c->sizes[0];
        localDesc.tensorO.roiSize[0] = params->roi->size[0] % gp.subMatrixWidth;
        localDesc.tensorO.roiSize[0] = localDesc.tensorO.roiSize[0] ? localDesc.tensorO.roiSize[0] : gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[0] = ((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi->denseBase - params->c->bases[0];
        }

        // tensor B: K (dim 0)
        localDesc.tensorL.roiSize[0] = params->roi->size[0] % gp.subMatrixWidth;
        localDesc.tensorL.roiSize[0] = localDesc.tensorL.roiSize[0] ? localDesc.tensorL.roiSize[0] : gp.subMatrixWidth;

        unsigned loopS[c_operand_max_agu];
        unsigned loopL[c_operand_max_agu];
        unsigned loopO[c_operand_max_agu];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            loopS[j] = getLoopMask(params->strategy->pattern, LOOP_SPATIAL);
            unsigned rem = params->roi->size[0] % gp.matrixWidth;
            rem = rem ? rem : gp.matrixWidth;
            loopL[j] = ((rem + params->roi->denseBase - aguO[j].roiBaseOffset[0]) < gp.subMatrixWidth) ? getLoopMask(params->strategy->pattern, LOOP_K) : 0;
            loopO[j] = ((rem + params->roi->denseBase - aguO[j].roiBaseOffset[0]) < gp.subMatrixWidth) ? getLoopMask(params->strategy->pattern, LOOP_K) : 0;
        }


        for (int dim = 0; dim < c_padded_conv_dim; dim++)
        {
            int ioDim = dim + 1;

            // tensor C: spatial loops
            localDesc.tensorO.validElements[ioDim] = params->c->sizes[ioDim] * params->c->strides[ioDim];
            localDesc.tensorO.roiSize[ioDim] = paddedRoi.size[ioDim] * params->c->strides[ioDim];
            localDesc.tensorO.loopStride[ioDim] = 0;
            localDesc.tensorO.spatialStrides[ioDim - 1] = params->c->strides[ioDim] * (dim ? 1 : gp.aguAnr);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].startOffset[dim] = spPos[j / gp.aguBnr][dim] * params->c->strides[ioDim];
            }

            setLoopDim(&localDesc, params->strategy->pattern, LOOP_FILTER + dim, OP_O, ioDim);
        }

        for (int dim = c_padded_conv_dim; dim < Mme::c_mme_max_tensor_dims; dim++)
        {
            // tensor C: spatial loops
            localDesc.tensorO.validElements[dim] = (params->c->sizes[dim] - params->c->bases[dim]) * params->c->strides[dim];
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorO.roiSize[dim] = paddedRoi.size[dim] * params->c->strides[dim];
            }
            localDesc.tensorO.spatialStrides[dim - 1] = params->c->strides[dim];

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].roiBaseOffset[dim] = 0;
                aguO[j].startOffset[dim - 1] = spPos[j / gp.aguBnr][dim - 1] * params->c->strides[dim];
            }
        }

        localDesc.tensorO.loopStride[0] = gp.matrixWidth;
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_O, 0);
        unsigned denseLoopSize = ((params->roi->size[0] + gp.matrixWidth - 1) / gp.matrixWidth) - 1;
        setLoopSize(&localDesc, params->strategy->pattern, LOOP_K, denseLoopSize);

        if (gp.transO)
        {
            transposeDesc(&localDesc);
            std::swap(aguS, aguL);
            std::swap(gp.aguAnr, gp.aguBnr);
            std::swap(loopS, loopL);
        }

        DescGroup descGroup;
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            descGroup.desc[descIdx] = localDesc;
            descGroup.desc[descIdx].aguS = aguS[descIdx];
            descGroup.desc[descIdx].header.partialHeightLoopS = loopS[descIdx];

            for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
            {
                descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                if (aguIdx == Mme::e_mme_local)
                {
                    descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                    descGroup.desc[descIdx].header.partialHeightLoopOLocal = loopO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                }
                else
                {
                    descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                    descGroup.desc[descIdx].header.partialHeightLoopORemote = loopO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                }
            }
        }

        descList->push_back(descGroup);
        TRACE_EXIT;
    }
};

// currently not used
class MmeStackCmd_BuildMemsetDescDEDW : public MmeStackCmd
{
public:
    MmeStackCmd_BuildMemsetDescDEDW() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_BuildMemsetDescDEDW); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->a->strides[0] == 1, "FCD stride should be 0 in operand A");
        MME_ASSERT(params->b->strides[0] == 1, "FCD stride should be 0 in operand B");
        MME_ASSERT(params->c->strides[0] == 1, "FCD stride should be 0 in operand C");
        MME_ASSERT(params->a->elementType == params->b->elementType, "element type should match between operand A & B");

        GeoParams gp;
        geo2Params(params->strategy->geometry, params->c->elementType, &gp);

        Mme::Desc localDesc;
        // Zeros reserved fields and makes the desc calculation deterministic
        memset(&localDesc, 0, sizeof(localDesc));

        localDesc.sw.reserved0 = 0;
        localDesc.sw.swMemsetDedw = 1;

        localDesc.pcu.rlSaturation = 16 * 4096;
        localDesc.axiUserData.dw = getUserDataVal(
                params->controls->atomicAdd,
                params->c->elementType,
                params->controls->roundingMode);
        localDesc.perfEvtS.dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtS.startEndMask);
        localDesc.perfEvtL[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtL[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtL[Mme::e_mme_remote].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_local].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_local].startEndMask);
        localDesc.perfEvtO[Mme::e_mme_remote].dw = 0;
        set_bf_to_all_ones(localDesc.perfEvtO[Mme::e_mme_remote].startEndMask);
        localDesc.metaData.aguS = 0;
        localDesc.metaData.aguL[Mme::e_mme_local] = 0;
        localDesc.metaData.aguL[Mme::e_mme_remote] = 0;
        localDesc.metaData.aguO[Mme::e_mme_local] = 0;
        localDesc.metaData.aguO[Mme::e_mme_remote] = 0;

        localDesc.rateLimiter.aguL = 4;
        localDesc.rateLimiter.aguS = 4;
        localDesc.rateLimiter.aguO = 2;

        localDesc.sbRepeat.teEnS = 1;
        localDesc.sbRepeat.teEnL = 1;
        localDesc.sbRepeat.loadS = 1;
        localDesc.sbRepeat.loadL = 1;
        localDesc.sbRepeat.aguSLoopMask = 0;
        localDesc.sbRepeat.aguLLoopMask = 0;
        localDesc.sbRepeat.repeatSMinus1 = 0;
        localDesc.sbRepeat.repeatLMinus1 = 0;

        localDesc.header.euBEn = 1;
        localDesc.header.fpEn = 1;
        localDesc.header.transO = 0;
        localDesc.header.transS = 0;
        localDesc.header.transL = 0;
        localDesc.header.lowerS = 0;
        localDesc.header.lowerL = 0;
        localDesc.header.accumMask = 0;
        localDesc.header.storeEn = 1;
        localDesc.header.advanceS = 0;
        localDesc.header.advanceL = 0;
        localDesc.header.advanceO = 0;
        localDesc.header.accStoreIncDisable = 0;
        localDesc.header.dataTypeIn = ConvertToGaudiDataType(params->c->elementType);
        localDesc.header.dataTypeOut = ConvertToGaudiDataType(params->c->elementType);
        localDesc.header.reluEn = 0;
        localDesc.header.accum = 0;
        localDesc.header.rollAccums = 0;
        localDesc.header.roundingMode = Mme::e_mme_rm_rz;
        localDesc.header.signalEn = 0;
        localDesc.header.signalMask = 0;

        for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
        {
            localDesc.conv.associatedDims[i].dimS = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimL = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimO = Mme::c_mme_max_tensor_dims;
        }
        localDesc.outerLoop.associatedDims.dimS = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimL = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimO = Mme::c_mme_max_tensor_dims;

        localDesc.baseAddrHighS = 0;
        localDesc.baseAddrLowS = 0;
        localDesc.baseAddrHighL = 0;
        localDesc.baseAddrLowL = 0;
        localDesc.baseAddrHighO = params->cPtr.u32[1];
        localDesc.baseAddrLowO = params->cPtr.u32[0];

        localDesc.tensorS.spatialSizeMinus1 = 0;
        localDesc.tensorL.spatialSizeMinus1 = 0;
        localDesc.tensorO.spatialSizeMinus1 = std::min(params->c->sizes[1], gp.subMatrixHeight) - 1;

        localDesc.paddingValueL = 0;
        localDesc.paddingValueS = 0;

        localDesc.syncObject.addrHigh = 0;
        localDesc.syncObject.addrLow[0] = 0;
        localDesc.syncObject.addrLow[1] = 0;
        localDesc.syncObject.operation = 1;
        localDesc.syncObject.value = 1;

        // set default values
        Mme::MmeAguCoreDesc aguS[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguL[c_operand_max_agu] = { 0 };
        Mme::MmeAguCoreDesc aguO[c_operand_max_agu] = { 0 };

        localDesc.outerLoop.sizeMinus1 = 0;
        localDesc.numIterationsMinus1 = 0;
        localDesc.conv.kernelSizeMinus1.dw = 0;

        for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
        {

            localDesc.tensorS.loopStride[i] = 0;
            localDesc.tensorL.loopStride[i] = 0;
            localDesc.tensorO.loopStride[i] = 0;

            localDesc.tensorS.validElements[i] = 0;
            localDesc.tensorL.validElements[i] = 0;
            localDesc.tensorO.validElements[i] = 1;

            if (i < Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[i] = 1;
                localDesc.tensorL.roiSize[i] = 1;
                localDesc.tensorO.roiSize[i] = 1;
            }

            if (i > 0)
            {
                localDesc.tensorS.spatialStrides[i - 1] = 1;
                localDesc.tensorL.spatialStrides[i - 1] = 1;
                localDesc.tensorO.spatialStrides[i - 1] = 1;
            }
        }

        memset(aguS, 0, sizeof(aguS));
        memset(aguL, 0, sizeof(aguL));
        memset(aguO, 0, sizeof(aguO));

        // tensor A & B: (dims 0)
        localDesc.tensorS.roiSize[0] = std::min(params->c->sizes[1], gp.subMatrixHeight);
        localDesc.tensorL.roiSize[0] = gp.subMatrixWidth;

        // tensor C: (dim 0)
        localDesc.tensorO.validElements[0] = params->c->sizes[0];
        localDesc.tensorO.roiSize[0] = gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[0] = (j % gp.aguBnr) * gp.subMatrixWidth;
        }

        // tensor C: (dim 1)
        localDesc.tensorO.validElements[1] = params->c->sizes[1] * params->c->strides[1];
        localDesc.tensorO.roiSize[1] = std::min(params->c->sizes[1], gp.subMatrixHeight) * params->c->strides[1];
        localDesc.tensorO.spatialStrides[0] = params->c->strides[1];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[1] = (j / gp.aguBnr) * gp.subMatrixHeight * params->c->strides[1];
        }

        localDesc.tensorO.loopStride[0] = gp.matrixWidth;
        localDesc.tensorO.loopStride[1] = gp.matrixHeight * params->c->strides[1];

        unsigned loopSizeK = ((params->c->sizes[0] + gp.matrixWidth - 1) / gp.matrixWidth) - 1;
        setLoopSize(&localDesc, params->strategy->pattern, LOOP_K, loopSizeK);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_K, OP_O, 0);

        unsigned loopSizeC = ((params->c->sizes[1] + gp.matrixHeight - 1) / gp.matrixHeight) - 1;
        setLoopSize(&localDesc, params->strategy->pattern, LOOP_C, loopSizeC);
        setLoopDim(&localDesc, params->strategy->pattern, LOOP_C, OP_O, 1);

        unsigned loopS = getLoopMask(params->strategy->pattern, LOOP_C);
        unsigned loopL = getLoopMask(params->strategy->pattern, LOOP_K);

        if (gp.transO)
        {
            transposeDesc(&localDesc);
            std::swap(aguS, aguL);
            std::swap(gp.aguAnr, gp.aguBnr);
            std::swap(loopS, loopL);
        }

        DescGroup descGroup;
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            descGroup.desc[descIdx] = localDesc;
            descGroup.desc[descIdx].aguS = aguS[descIdx];
            descGroup.desc[descIdx].header.partialHeightLoopS = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopL;
            descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopL;
            for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
            {
                descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx*Mme::e_mme_local_and_remote) + aguIdx];
            }
        }

        descList->push_back(descGroup);
        TRACE_EXIT;
    }
};


class MmeStackCmd_WeightUnroll : public MmeStackCmd
{
public:
    MmeStackCmd_WeightUnroll() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_WeightUnroll); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT_PTR(m_next);
        GeoParams gp;
        geo2Params(params->strategy->geometry, params->a->elementType, &gp);
        int maxUnrollFactor = std::min(gp.matrixWidth / params->c->sizes[0], gp.aguBnr);
        int currUnrollFactor = 1;
        unsigned convDim = 0;
        if (maxUnrollFactor > 1)
        {
            for (unsigned dim = 0; dim<c_padded_conv_dim; dim++)
            {
                int dimUnroll = mme_div_round_up(params->c->sizes[dim+2], params->conv->stride[dim]);
                dimUnroll = std::min(dimUnroll, maxUnrollFactor);
                if (dimUnroll > currUnrollFactor)
                {
                    currUnrollFactor = dimUnroll;
                    convDim = dim;
                }
            }
        }

        unsigned weightDim = convDim + 2;
        unsigned tensorDim = convDim + 1;

        if ((params->strategy->geometry == e_mme_geometry_4wx1h) && (currUnrollFactor > 1) &&
            (params->a->sizes[tensorDim] == params->b->sizes[tensorDim] * params->conv->stride[convDim]))
        {
            // partial filter in dedw is still not supported:
            MME_ASSERT(params->c->bases[tensorDim] == 0, "partial filter in dedw is still not supported");

            // roi in unrolled dedw is still not supported.
            MME_ASSERT(params->b->bases[weightDim] == 0, "roi in unrolled dedw is still not supported");

            unsigned wDataSize = (params->c->elementType == e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);

            for (unsigned wc = 0; wc < params->conv->stride[convDim]; wc++)
            {
                MmeControls new_controls = *params->controls;
                MmeConv new_conv = *params->conv;
                MmeTensorView new_w = *params->c;
                ExecuteParams new_params = *params;
                new_params.controls = &new_controls;
                new_params.conv = &new_conv;
                new_params.c = &new_w;

                new_params.cPtr.u64 += wc * params->c->strides[weightDim] * wDataSize;
                new_w.strides[weightDim] *= params->conv->stride[convDim];
                new_w.sizes[weightDim] /= params->conv->stride[convDim];
                if (wc < (params->c->sizes[weightDim] % params->conv->stride[convDim]))
                {
                    new_w.sizes[weightDim]++;
                }

                int RDMinusP = (params->conv->dilation[convDim] * wc) - params->conv->padding[convDim];
                new_conv.padding[convDim] = -mod_neg(RDMinusP, params->conv->stride[convDim]);

                unsigned groupIdx = (unsigned)descList->size();

                MME_ASSERT_PTR(m_next);
                m_next->execute(&new_params, descList);

                for (; groupIdx < descList->size(); groupIdx++)
                {
                    const Mme::Desc *desc0 = &(*descList)[groupIdx].desc[0];

                    MME_ASSERT(0 == (*descList)[groupIdx].desc[0].aguL[Mme::e_mme_local].roiBaseOffset[0],
                              "expected roiBaseOffset 0");

                    for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
                    {
                        Mme::Desc *currDesc = &(*descList)[groupIdx].desc[descIdx];

                        currDesc->tensorS.loopStride[tensorDim] = 0;
                        currDesc->tensorL.loopStride[tensorDim] = currDesc->tensorL.spatialStrides[tensorDim-1] * -currUnrollFactor;
                        currDesc->tensorO.loopStride[weightDim] = currDesc->tensorO.spatialStrides[weightDim-1] * currUnrollFactor;

                        setLoopDim(currDesc, new_params.strategy->pattern, LOOP_FILTER + convDim, OP_L, tensorDim);
                        setLoopDim(currDesc, new_params.strategy->pattern, LOOP_FILTER + convDim, OP_O, weightDim);
                        setLoopDim(currDesc, new_params.strategy->pattern, LOOP_FILTER + convDim, OP_S, Mme::c_mme_max_tensor_dims);

                        unsigned loopSize = getLoopSize(currDesc, new_params.strategy->pattern, LOOP_FILTER + convDim) / currUnrollFactor;
                        setLoopSize(currDesc, new_params.strategy->pattern, LOOP_FILTER + convDim, loopSize);

                        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_remote].startOffset[tensorDim - 1],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_local].startOffset[tensorDim - 1],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_remote].startOffset[weightDim - 1],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_local].startOffset[weightDim - 1],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_remote].roiBaseOffset[tensorDim],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguL[Mme::e_mme_local].roiBaseOffset[tensorDim],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_remote].roiBaseOffset[weightDim],
                                  "expected roiBaseOffset = 0");
                        MME_ASSERT(0 == currDesc->aguO[Mme::e_mme_local].roiBaseOffset[weightDim],
                                  "expected roiBaseOffset = 0");

                        currDesc->aguL[Mme::e_mme_local].roiBaseOffset[0] = desc0->aguL[Mme::e_mme_local].roiBaseOffset[0];
                        currDesc->aguO[Mme::e_mme_local].roiBaseOffset[0] = desc0->aguO[Mme::e_mme_local].roiBaseOffset[0];

                        currDesc->aguL[Mme::e_mme_local].roiBaseOffset[tensorDim] =
                            currDesc->tensorL.spatialStrides[tensorDim - 1] *
                            -(div_round_down(RDMinusP, (int) params->conv->stride[convDim]) + descIdx);

                        currDesc->aguO[Mme::e_mme_local].roiBaseOffset[weightDim] =
                            currDesc->tensorO.spatialStrides[weightDim-1] * descIdx;

                        currDesc->aguL[Mme::e_mme_remote].roiBaseOffset[0] = desc0->aguL[Mme::e_mme_local].roiBaseOffset[0];
                        currDesc->aguO[Mme::e_mme_remote].roiBaseOffset[0] = desc0->aguO[Mme::e_mme_local].roiBaseOffset[0];

                        currDesc->aguL[Mme::e_mme_remote].roiBaseOffset[tensorDim] =
                            currDesc->aguL[Mme::e_mme_local].roiBaseOffset[tensorDim] -
                            (Mme::MME_MASTERS_NR * currDesc->tensorL.spatialStrides[tensorDim-1]);

                        currDesc->aguO[Mme::e_mme_remote].roiBaseOffset[weightDim] =
                            currDesc->aguO[Mme::e_mme_local].roiBaseOffset[weightDim] +
                                (Mme::MME_MASTERS_NR * currDesc->tensorO.spatialStrides[weightDim-1]);
                    }
                }
            }
        }
        else
        {
            m_next->execute(params, descList);
        }

        TRACE_EXIT;
    }
};


class MmeStackCmd_LowerA : public MmeStackCmd
{
public:
    MmeStackCmd_LowerA() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_LowerA); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MmeTensorView new_a = *params->a;
        MmeTensorView new_w = *((params->opType == e_mme_dedw) ? params->c : params->b);

        bool lowerA =
                (params->strategy->loweringEn) &&
                (params->conv->dilation[0] == 1) &&
                (new_a.sizes[0] == new_a.strides[1]) &&
                (new_w.strides[2] == new_w.sizes[0] * new_w.sizes[1]);

        if (lowerA)
        {
            new_a.sizes[0] *= new_w.sizes[2];
            new_w.sizes[1] *= new_w.sizes[2];
            new_w.strides[2] *= new_w.sizes[2];
            new_w.sizes[2] = 1;
        }

        ExecuteParams new_params = *params;
        new_params.a = &new_a;
        if (params->opType == e_mme_dedw)
        {
            new_params.c = &new_w;
        }
        else
        {
            new_params.b = &new_w;
        }

        unsigned groupIdx = (unsigned)descList->size();

        MME_ASSERT_PTR(m_next);
        m_next->execute(&new_params, descList);

        for (; groupIdx < descList->size(); groupIdx++)
        {
            for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
            {
                if ((*descList)[groupIdx].desc[descIdx].header.transO)
                {
                    (*descList)[groupIdx].desc[descIdx].header.lowerL = lowerA;
                }
                else
                {
                    (*descList)[groupIdx].desc[descIdx].header.lowerS = lowerA;
                }
            }
        }
        TRACE_EXIT;
    }
};

class MmeStackCmd_LowerM : public MmeStackCmd
{
public:
    MmeStackCmd_LowerM() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_LowerM); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MmeTensorView new_a = *params->a;

        bool lowerA =
                (params->strategy->loweringEn) &&
                (getLoopMask(params->strategy->pattern, LOOP_FILTER0) == 0x1) &&
                (params->conv->dilation[0] == 1) &&
                (new_a.sizes[0] == new_a.strides[1]);

        if (lowerA)
        {
            const MmeTensorView* w = (params->opType == e_mme_dedw) ? params->c : params->b;
            new_a.sizes[0] *= w->sizes[2];
        }

        ExecuteParams new_params = *params;
        new_params.a = &new_a;


        unsigned groupIdx = (unsigned)descList->size();

        MME_ASSERT_PTR(m_next);
        m_next->execute(&new_params, descList);

        for (; groupIdx < descList->size(); groupIdx++)
        {
            for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
            {
                if ((*descList)[groupIdx].desc[descIdx].header.transO)
                {
                    (*descList)[groupIdx].desc[descIdx].header.lowerL = lowerA ? 1 : 0;
                    (*descList)[groupIdx].desc[descIdx].sbRepeat.aguLLoopMask |= lowerA ? getLoopMask(params->strategy->pattern, LOOP_FILTER0) : 0;
                }
                else
                {
                    (*descList)[groupIdx].desc[descIdx].header.lowerS = lowerA ? 1 : 0;
                    (*descList)[groupIdx].desc[descIdx].sbRepeat.aguSLoopMask |= lowerA ? getLoopMask(params->strategy->pattern, LOOP_FILTER0) : 0;
                }
            }
        }
        TRACE_EXIT;
    }
};


class MmeStackCmd_Partial : public MmeStackCmd
{
public:

    static const unsigned PARTIAL_ALL = INT32_MAX;

    MmeStackCmd_Partial(LOOP loop, unsigned chunkSize = PARTIAL_ALL) : MmeStackCmd(), m_loop(loop), m_chunk(chunkSize) {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_Partial); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        unsigned firstIdx = (unsigned)descList->size();

        switch (m_loop)
        {
            case LOOP_FILTER0:
            case LOOP_FILTER1:
            case LOOP_FILTER2:
                MME_ASSERT(params->opType != e_mme_dedw, "op type cannot be dedw");
                executePartialFilter(params, descList);
                break;
            case LOOP_C:
                MME_ASSERT(params->opType != e_mme_dedw, "op type cannot be dedw");
                executePartialChannel(params, descList);
                break;
            case LOOP_SPATIAL:
                MME_ASSERT(params->opType == e_mme_dedw, "op type must be dedw");
                executePartialSpatial(params, descList);
                break;
            default:
                MME_ASSERT(0, "invalid LOOP");
        }

        closePartialExecute(firstIdx, params, descList);
        TRACE_EXIT;
    }

private:

    void executePartialChannel(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;

        unsigned aDataSize =
            (params->a->elementType == EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);
        unsigned bDataSize =
            (params->b->elementType == EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);

        MmeTensorView new_a = *params->a;
        MmeTensorView new_b = *params->b;

        ExecuteParams new_params = *params;
        new_params.a = &new_a;
        new_params.b = &new_b;

        for (unsigned a_offsets0 = 0; a_offsets0 < params->a->sizes[0]; a_offsets0 += m_chunk)
        {
            new_a.sizes[0] = std::min(m_chunk, params->a->sizes[0] - a_offsets0);
            new_a.bases[0] = params->a->bases[0] + a_offsets0;
            new_params.aPtr.u64 = params->aPtr.u64 + (a_offsets0 * params->a->strides[0] * aDataSize);
            switch (params->opType)
            {
                case e_mme_fwd:
                    new_b.sizes[1] = std::min(new_a.sizes[0], new_b.sizes[1]);
                    new_b.bases[1] = params->b->bases[1] + a_offsets0;
                    new_params.bPtr.u64 = params->bPtr.u64 + (a_offsets0 * params->b->strides[1] * bDataSize);
                    break;
                case e_mme_dedx:
                    new_b.sizes[0] = std::min(new_a.sizes[0], new_b.sizes[0]);
                    new_b.bases[0] = params->b->bases[0] + a_offsets0;
                    new_params.bPtr.u64 = params->bPtr.u64 + (a_offsets0 * params->b->strides[0] * bDataSize);
                    break;
                case e_mme_dedw:
                default:
                    MME_ASSERT(0, "invalid operation");
            }

            MME_ASSERT_PTR(m_next);
            m_next->execute(&new_params, descList);
        }
        TRACE_EXIT;
    }

    void executePartialFilter(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        unsigned bDataSize =
            (params->b->elementType == EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);

        MME_ASSERT(params->opType != e_mme_dedw, "op type cannot be dedw");

        unsigned dim = m_loop - LOOP_FILTER0;
        unsigned wDim = dim + 2;

        MmeConv new_conv = *params->conv;
        MmeTensorView new_b = *params->b;
        ExecuteParams new_params = *params;
        new_params.b = &new_b;
        new_params.conv = &new_conv;

        for (unsigned b_offset = 0; b_offset < params->b->sizes[wDim]; b_offset += m_chunk)
        {
            new_b.sizes[wDim] = std::min(m_chunk, params->b->sizes[wDim] - b_offset);
            new_b.bases[wDim] = params->b->bases[wDim] + b_offset;
            new_params.bPtr.u64 = params->bPtr.u64 + (b_offset * params->b->strides[wDim] * bDataSize);

            MME_ASSERT_PTR(m_next);
            m_next->execute(&new_params, descList);
        }
        TRACE_EXIT;
    }

    void executePartialSpatial(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT(params->opType == e_mme_dedw, "op type must be dedw");

        MmeSpatialSlice new_spatialSlice;
        ExecuteParams new_params = *params;
        new_params.spatialSlice = &new_spatialSlice;

        for (unsigned base = 0; base < params->spatialSlice->spSize; base += m_chunk)
        {
            new_spatialSlice.spSize = std::min(params->spatialSlice->spSize - base, m_chunk);
            new_spatialSlice.spBase = params->spatialSlice->spBase + base;

            MME_ASSERT_PTR(m_next);
            m_next->execute(&new_params, descList);
        }
        TRACE_EXIT;
    }

    static void closePartialExecute(unsigned firstIdx, ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        unsigned accumsNr;
        Mme::Desc *desc0 = &(*descList)[firstIdx].desc[0];
        unsigned loopKSize = getLoopSize(desc0, params->strategy->pattern, LOOP_K) + 1;

        if (params->opType == e_mme_dedw)
        {
            unsigned loopCSize = getLoopSize(desc0, params->strategy->pattern, LOOP_C) + 1;
            accumsNr = loopKSize * loopCSize;
        }
        else
        {
            unsigned tetrisNr = desc0->numIterationsMinus1 + 1;
            accumsNr = loopKSize * tetrisNr;
        }

        unsigned descsNr = (unsigned)descList->size() - firstIdx;
        MME_ASSERT((descsNr == 1) || (accumsNr <= Mme::c_mme_accums_nr), "accumNr overflow");
        UNUSED(descsNr);

        bool transOut = descList->front().desc[0].header.transO;

        for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
        {
            Mme::Desc *desc;

            // steady state
            for (unsigned descIdx = firstIdx; descIdx < (unsigned)descList->size(); descIdx++)
            {
                desc = &(*descList)[descIdx].desc[masterIdx];
                desc->header.accum = (accumsNr == 1) ? 0 : 1;
                desc->header.storeEn = 0;
                desc->header.accStoreIncDisable = (accumsNr == 1) ? 1 : 0;
                desc->header.rollAccums = (accumsNr == 1) ? 0 : Mme::c_mme_accums_nr - accumsNr;
                desc->header.reluEn = false;
                MME_ASSERT(desc->header.transO == transOut, "invalid transO in desc");
                UNUSED(transOut);
            }

            // first desc
            desc = &(*descList)[firstIdx].desc[masterIdx];
            desc->header.accum = 0;

            // last desc
            desc = &descList->back().desc[masterIdx];
            desc->header.storeEn = 1;
            desc->header.accStoreIncDisable = 0;
            desc->header.rollAccums = 0;
            desc->header.reluEn = params->controls->reluEn && (params->opType == e_mme_fwd);
        }
        TRACE_EXIT;
    }

    LOOP m_loop;
    unsigned m_chunk;
};

class MmeStackCmd_ConvSpatialRoi : public MmeStackCmd
{
public:
    MmeStackCmd_ConvSpatialRoi(unsigned activations) : MmeStackCmd(), m_activations(activations) {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_ConvSpatialRoi); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        GeoParams geoParams;
        geo2Params(params->strategy->geometry, params->a->elementType, &geoParams);

        MmeRoi paddedRoi;
        padRoi(geoParams.aguAnr,  params->roi, &paddedRoi);

        MmeRoi new_roi = paddedRoi;
        ExecuteParams new_params = *params;
        new_params.roi = &new_roi;

        for (unsigned base = 0; base < paddedRoi.spSize; base += (geoParams.matrixHeight * m_activations))
        {
            new_roi.spBase = base + paddedRoi.spBase;
            new_roi.spSize = std::min(paddedRoi.spSize - base, geoParams.matrixHeight * m_activations);
            m_next->execute(&new_params, descList);
        }
        TRACE_EXIT;
    }

private:
    const unsigned m_activations;
};

class MmeStackCmd_DedwFilterRoi : public MmeStackCmd
{
public:
    MmeStackCmd_DedwFilterRoi(unsigned filter, unsigned activations) :
        MmeStackCmd(), m_activations(activations), m_filter(filter) {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_DedwFilterRoi); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        const unsigned elementSize =
            (params->c->elementType == EMmeDataType::e_type_fp32) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        ExecuteParams new_params = *params;
        MmeTensorView new_c = *params->c;
        MmeConv new_conv = *params->conv;
        new_params.c = &new_c;
        new_params.conv = &new_conv;


        for (unsigned base = 0; base < params->c->sizes[m_filter+2]; base += m_activations)
        {
            new_c.sizes[m_filter+2] = std::min(m_activations, params->c->sizes[m_filter+2]-base);
            new_conv.padding[m_filter] = params->conv->padding[m_filter] - (params->conv->dilation[m_filter] * base);
            new_params.cPtr.u64 = params->cPtr.u64 + ((uint64_t)params->c->strides[m_filter+2] * base * elementSize);

            m_next->execute(&new_params, descList);
        }

        TRACE_EXIT;
    }

private:
    const unsigned m_activations;
    const unsigned m_filter;
};

class MmeStackCmd_DedwKRoi : public MmeStackCmd
{
public:
    MmeStackCmd_DedwKRoi(unsigned activations) :
        MmeStackCmd(), m_activations(activations) {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_DedwKRoi); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        const unsigned elementSizeB =
            (params->b->elementType == EMmeDataType::e_type_fp32) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        const unsigned elementSizeC =
            (params->c->elementType == EMmeDataType::e_type_fp32) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        GeoParams geoParams;
        geo2Params(params->strategy->geometry, params->a->elementType, &geoParams);

        ExecuteParams new_params = *params;
        MmeTensorView new_b = *params->b;
        MmeTensorView new_c = *params->c;
        new_params.b = &new_b;
        new_params.c = &new_c;

        unsigned step = m_activations * geoParams.matrixWidth;
        for (unsigned base = 0; base < params->c->sizes[0]; base += step)
        {
            int64_t size_b = std::min(((int64_t)params->b->sizes[0]) - base, (int64_t)step);
            new_b.sizes[0] = std::max(size_b, 0L);
            new_params.bPtr.u64 = params->bPtr.u64 + (base * elementSizeB);

            new_c.sizes[0] = std::min(params->c->sizes[0] - base, step);
            new_params.cPtr.u64 = params->cPtr.u64 + (base * elementSizeC);

            m_next->execute(&new_params, descList);
        }

        TRACE_EXIT;
    }

private:
    const unsigned m_activations;
};

class MmeStackCmd_DedwCRoi : public MmeStackCmd
{
public:
    MmeStackCmd_DedwCRoi(unsigned activations) :
        MmeStackCmd(), m_activations(activations) {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_DedwCRoi); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        const unsigned elementSizeA =
            (params->a->elementType == EMmeDataType::e_type_fp32) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        const unsigned elementSizeC =
            (params->c->elementType == EMmeDataType::e_type_fp32) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        GeoParams geoParams;
        geo2Params(params->strategy->geometry, params->a->elementType, &geoParams);

        ExecuteParams new_params = *params;
        MmeTensorView new_a = *params->a;
        MmeTensorView new_c = *params->c;
        new_params.a = &new_a;
        new_params.c = &new_c;

        unsigned step = m_activations * geoParams.matrixHeight;
        for (unsigned base = 0; base < params->c->sizes[1]; base += step)
        {
            int64_t size_a = std::min(((int64_t)params->a->sizes[0]) - base, (int64_t)step);
            new_a.sizes[0] = std::max(size_a, 0L);
            new_params.aPtr.u64 = params->aPtr.u64 + (base * elementSizeA);

            new_c.sizes[1] = std::min(params->c->sizes[1] - base, step);
            new_params.cPtr.u64 = params->cPtr.u64 + (base * params->c->strides[1] * elementSizeC);

            m_next->execute(&new_params, descList);
        }

        TRACE_EXIT;
    }

private:
    const unsigned m_activations;
};


class MmeStackCmd_ConvDenseRoi : public MmeStackCmd
{
public:
    MmeStackCmd_ConvDenseRoi(unsigned activations) : MmeStackCmd(), m_activations(activations) {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_ConvDenseRoi); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        GeoParams geoParams;
        geo2Params(params->strategy->geometry, params->a->elementType, &geoParams);

        MmeRoi new_roi = *params->roi;
        ExecuteParams new_params = *params;
        new_params.roi = &new_roi;

        for (unsigned base = 0; base < params->roi->size[0]; base += (geoParams.matrixWidth * m_activations))
        {
            new_roi.size[0] = std::min(params->roi->size[0] - base, geoParams.matrixWidth * m_activations);
            new_roi.denseBase = params->roi->denseBase + base;

            m_next->execute(&new_params, descList);
        }
        TRACE_EXIT;
    }

private:
    const unsigned m_activations;
};


class MmeStackCmd_ReuseConv : public MmeStackCmd
{
public:

    MmeStackCmd_ReuseConv() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_ReuseConv); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;

        // call the lower layers in the command stack to generate a descriptor
        unsigned firstIdx = (unsigned)descList->size();
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);
        MME_ASSERT(descList->size() - firstIdx == 1, "only single descriptor should be added");

        // extract info from the descriptor to determine if reuse could take place
        unsigned mask;
        unsigned reuse;
        unsigned stripes;
        bool sharedOperandResue;
        if (params->strategy->pattern == EMmePattern::e_mme_z_reduction_ksf)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_SPATIAL);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_SPATIAL);
            stripes = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K);
            sharedOperandResue = false;
        }
        else if (params->strategy->pattern == EMmePattern::e_mme_z_reduction_skf)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_K);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K);
            stripes = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_SPATIAL);
            sharedOperandResue = true;
        }
        else
        {
            MME_ASSERT(0, "invalid pattern");
            mask = 0;
            reuse = 0;
            stripes = 0;
            sharedOperandResue = false;

        }

        // check if reuse could take place.
        if (reuse)
        {
            // count the SB entries in the reused operand
            unsigned readsNr = 0;

            if (reuse < Mme::c_mme_max_sb_reuse)
            {
                readsNr = aguStatsCountSBReads(
                    sharedOperandResue,
                    &(*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER],
                    &(*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER]);
            }

            // if partials are not needed - reuse the selected operand.
            if ((readsNr <= Mme::c_mme_sb_size) && (reuse < Mme::c_mme_max_sb_reuse))
            {
                static const unsigned c_max_activations = 4;

                MME_ASSERT(descList->size() - firstIdx <= c_max_activations, "too many activations generated");
                UNUSED(c_max_activations);

                for (unsigned idx = firstIdx; idx < descList->size(); idx++)
                {
                    unsigned load = (idx == firstIdx) ? 1 : 0;

                    if (sharedOperandResue != (bool)(*descList)[firstIdx].desc[0].header.transO)
                    {
                        (*descList)[idx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.loadS &= load;
                        (*descList)[idx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.loadS &= load;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.aguSLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatSMinus1 = reuse;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.aguSLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.repeatSMinus1 = reuse;
                    }
                    else
                    {
                        (*descList)[idx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.loadL &= load;
                        (*descList)[idx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.loadL &= load;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.aguLLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatLMinus1 = reuse;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.aguLLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.repeatLMinus1 = reuse;
                    }
                }
            }
            else
            {
                // partials are needed (or the reuse is too large) - pop the descriptor and replace it by descriptors that can be reused.
                DescGroup noReuseDescsGroup = descList->back();
                descList->pop_back();
                MmeStackCmd head;

                ExecuteParams new_params = *params;

                double divFactor = ((double)readsNr) / Mme::c_mme_sb_size;

                // make sure the reuse is not too large
                if (reuse >= Mme::c_mme_max_sb_reuse)
                {
                    if (sharedOperandResue)
                    {
                        head.setLast(new MmeStackCmd_ConvDenseRoi(Mme::c_mme_max_sb_reuse));
                    }
                    else
                    {
                        head.setLast(new MmeStackCmd_ConvSpatialRoi(Mme::c_mme_max_sb_reuse));
                    }
                }

                    // break the descriptor into stripes with one activation of the shared operand.
                else if (stripes)
                {
                    if (sharedOperandResue)
                    {
                        head.setLast(new MmeStackCmd_ConvSpatialRoi(1));
                    }
                    else
                    {
                        head.setLast(new MmeStackCmd_ConvDenseRoi(1));
                    }
                }
                    // make sure there are enough accumulators to hold the reuse factor of each stripe.
                else if (reuse >= Mme::c_mme_accums_nr)
                {
                    if (sharedOperandResue)
                    {
                        head.setLast(new MmeStackCmd_ConvDenseRoi(Mme::c_mme_accums_nr));
                    }
                    else
                    {
                        head.setLast(new MmeStackCmd_ConvSpatialRoi(Mme::c_mme_accums_nr));
                    }
                }
                else
                {
                    // partials
                    // if the common dim doesn't fit, try to break the filter,
                    if (new_params.b->sizes[4] > 1)
                    {
                        unsigned chunkSize = std::max((int)std::floor(new_params.b->sizes[4] / divFactor), 1);
                        head.setLast(new MmeStackCmd_Partial(LOOP_FILTER2, chunkSize));
                    }
                    else if (new_params.b->sizes[3] > 1)
                    {
                        unsigned chunkSize = std::max((int)std::floor(new_params.b->sizes[3] / divFactor), 1);
                        head.setLast(new MmeStackCmd_Partial(LOOP_FILTER1, chunkSize));
                    }
                    else if (new_params.b->sizes[2] > 1)
                    {
                        unsigned chunkSize = std::max((int)std::floor(new_params.b->sizes[2] / divFactor), 1);
                        head.setLast(new MmeStackCmd_Partial(LOOP_FILTER0, chunkSize));
                    }

                    // finally, break the input into chunks that fit in the SB. (with several sizes)
                    else
                    {
                        unsigned inputSize = new_params.a->sizes[0];
                        MME_ASSERT(inputSize > Mme::c_mme_sb_size / 2, "input size should be half CL or above");

                        unsigned chunkSize;
                        if (inputSize > Mme::c_mme_sb_size)
                        {
                            chunkSize = Mme::c_mme_sb_size;
                        }
                        else
                        {
                            chunkSize = Mme::c_mme_sb_size / 2;
                        }
                        head.setLast(new MmeStackCmd_Partial(LOOP_C, chunkSize));
                    }
                }

                head.setLast(new MmeStackCmd_ReuseConv);
                MME_ASSERT_PTR(m_next);
                head.setLast(m_next->clone());

                head.execute(&new_params, descList);

                if (!((*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatSMinus1) &&
                    !((*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatLMinus1))
                {
                    descList->resize(firstIdx);
                    descList->push_back(noReuseDescsGroup);
                }

            }
        }
        TRACE_EXIT;
    }
};

class MmeStackCmd_ReuseDedw : public MmeStackCmd
{
public:

    MmeStackCmd_ReuseDedw() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_ReuseDedw); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;

        // call the lower layers in the command stack to generate a descriptor
        unsigned firstIdx = (unsigned)descList->size();
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);
        MME_ASSERT(descList->size() - firstIdx == 1, "only single descriptor shoudl be generated");

        // extract info from the descriptor to determine if reuse could take place
        unsigned mask;
        unsigned reuse;
        bool stripes;
        unsigned filter = 1;
        bool sharedOperandResue;
        unsigned slowestNonTrivialFilter = 0;

        unsigned filterMask = 0;
        for (unsigned d=0; d<c_padded_conv_dim; d++)
        {
            unsigned f = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_FILTER + d);
            if (f)
            {
                filter *= f + 1;
                filterMask |= getLoopMask(params->strategy->pattern, LOOP_FILTER + d);
                slowestNonTrivialFilter = d;
            }
        }
        filter--;

        if (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_kfc)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_C) | filterMask;
            reuse = ((getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C) + 1) * (filter+1)) - 1;
            stripes = (getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K) > 0) || (filter > 0);
            sharedOperandResue = false;
        }
        else if (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_fkc)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_C);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C);
            stripes = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K) > 0;
            sharedOperandResue = false;
        }
        else if (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_fck)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_K);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K);
            stripes = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C) > 0;
            sharedOperandResue = true;
        }
        else if (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_cfk)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_K);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K);
            stripes = (getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C) > 0) || (filter > 0);
            sharedOperandResue = true;
        }
        else if (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_kcf)
        {
            mask = getLoopMask(params->strategy->pattern, LOOP_C) | filterMask;
            reuse = ((getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C) + 1) * (filter+1)) - 1;
            stripes = (getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K) > 0) ||
                      (getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C) > 0);
            sharedOperandResue = false;
        }
        else if (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_ckf)
        {
            mask = filterMask;
            reuse = filter;
            stripes = (getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_K) > 0) ||
                      (getLoopSize(&(*descList)[firstIdx].desc[0], params->strategy->pattern, LOOP_C) > 0);
            sharedOperandResue = false;
        }
        else
        {
            MME_ASSERT(0, "invalid pattern");
            mask = 0;
            reuse = 0;
            stripes = false;
            sharedOperandResue = false;
        }

        // check if reuse could take place.
        if (reuse)
        {
            // count the SB entries in the reused operand
            unsigned readsNr = 0;
            if (reuse < Mme::c_mme_max_sb_reuse)
            {
                readsNr = aguStatsCountSBReads(
                    sharedOperandResue,
                    &(*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER],
                    &(*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER]);
            }

            // if partials are not needed - reuse the selected operand.
            if ((readsNr <= Mme::c_mme_sb_size) && (reuse < Mme::c_mme_max_sb_reuse))
            {
                static const unsigned c_max_activations = 4;

                MME_ASSERT(descList->size() - firstIdx <= c_max_activations, "too many activations");
                UNUSED(c_max_activations);

                for (unsigned idx = firstIdx; idx < descList->size(); idx++)
                {
                    unsigned load = (idx == firstIdx) ? 1 : 0;

                    if (sharedOperandResue != (bool)(*descList)[firstIdx].desc[0].header.transO)
                    {
                        (*descList)[idx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.loadS &= load;
                        (*descList)[idx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.loadS &= load;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.aguSLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatSMinus1 = reuse;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.aguSLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.repeatSMinus1 = reuse;
                    }
                    else
                    {
                        (*descList)[idx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.loadL &= load;
                        (*descList)[idx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.loadL &= load;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.aguLLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatLMinus1 = reuse;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.aguLLoopMask |= mask;
                        (*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER].sbRepeat.repeatLMinus1 = reuse;
                    }
                }
            }
            else
            {
                // partials are needed (or the reuse is too large) - pop the descriptor and replace it by descriptors that can be reused.
                DescGroup noReuseDescsGroup = descList->back();
                descList->pop_back();
                MmeStackCmd head;

                ExecuteParams new_params = *params;

                // split the filter
                if (filter && ((params->strategy->pattern == EMmePattern::e_mme_sp_reduction_fck) ||
                               (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_fkc)))

                {
                    head.setLast(new MmeStackCmd_DedwFilterRoi(slowestNonTrivialFilter, 1));
                }

                // make sure the reuse is not too large
                else if (reuse >= Mme::c_mme_max_sb_reuse)
                {
                    if (filter && (params->strategy->pattern == EMmePattern::e_mme_sp_reduction_kfc))
                    {
                        head.setLast(new MmeStackCmd_DedwFilterRoi(slowestNonTrivialFilter, 1));
                    }

                    else if (sharedOperandResue && getLoopMask(params->strategy->pattern, LOOP_K))
                    {
                        head.setLast(new MmeStackCmd_DedwKRoi(Mme::c_mme_max_sb_reuse));
                    }

                    else if (!sharedOperandResue && getLoopMask(params->strategy->pattern, LOOP_C))
                    {
                        head.setLast(new MmeStackCmd_DedwCRoi(Mme::c_mme_max_sb_reuse));
                    }

                    else
                    {
                        MME_ASSERT(filter, "expected to have filter");
                        MME_ASSERT(params->strategy->pattern == EMmePattern::e_mme_sp_reduction_kcf, "invalid pattern");
                        head.setLast(new MmeStackCmd_DedwFilterRoi(slowestNonTrivialFilter, 1));
                    }
                }

                // partials are needed
                // break the descriptor into stripes with one activation of the shared operand.
                else if (stripes)
                {
                    if (filter && ((params->strategy->pattern == EMmePattern::e_mme_sp_reduction_kfc)))

                    {
                        head.setLast(new MmeStackCmd_DedwFilterRoi(slowestNonTrivialFilter, 1));
                    }
                    else if (sharedOperandResue)
                    {
                        head.setLast(new MmeStackCmd_DedwCRoi(1));
                    }
                    else
                    {
                        head.setLast(new MmeStackCmd_DedwKRoi(1));
                    }
                }

                // make sure there are enough accumulators to hold the reuse factor of each stripe.
                else if (reuse >= Mme::c_mme_accums_nr)
                {
                    if (filter)
                    {
                        head.setLast(new MmeStackCmd_DedwFilterRoi(slowestNonTrivialFilter, 1));
                    }
                    else if (sharedOperandResue)
                    {
                        head.setLast(new MmeStackCmd_DedwKRoi(Mme::c_mme_accums_nr));
                    }
                    else
                    {
                        head.setLast(new MmeStackCmd_DedwCRoi(Mme::c_mme_accums_nr));
                    }
                }
                else
                {
                    unsigned chunkSize = Mme::c_mme_sb_size;
                    unsigned spatialSize = (*descList)[firstIdx].desc[0].tensorS.spatialSizeMinus1;
                    MME_ASSERT(spatialSize > Mme::c_mme_sb_size / 2, "spatial size should be at least CL/2");
                    if (spatialSize <= Mme::c_mme_sb_size)
                    {
                        chunkSize /= 2;
                    }
                    head.setLast(new MmeStackCmd_Partial(LOOP_SPATIAL, chunkSize));
                }

                head.setLast(new MmeStackCmd_ReuseDedw);
                MME_ASSERT_PTR(m_next);
                head.setLast(m_next->clone());

                head.execute(&new_params, descList);

                if (!((*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatSMinus1) &&
                    !((*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER].sbRepeat.repeatLMinus1))
                {
                    descList->resize(firstIdx);
                    descList->push_back(noReuseDescsGroup);
                }

            }
        }
        TRACE_EXIT;
    }
};

class MmeStackCmd_StrategyConv : public MmeStackCmd
{
public:
    MmeStackCmd_StrategyConv() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_StrategyConv); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        ExecuteParams new_params = *params;
        MME_ASSERT(params->opType != e_mme_dedw, "invalid operation type");

        switch (params->strategy->pattern)
        {
            case EMmePattern::e_mme_z_reduction_ksf:
                break;
            case EMmePattern::e_mme_z_reduction_skf:
                break;
            default:
                MME_ASSERT(0, "invalid walking pattern");
        }

        switch (params->strategy->geometry)
        {
            case e_mme_geometry_4wx1h:
                break;
            case e_mme_geometry_2wx2h:
                break;
            case e_mme_geometry_1wx4h:
                break;
            default:
                MME_ASSERT(0, "invalid geometry");
        }

        MmeStackCmd head;

        if (m_next)
        {
            head.setNext(m_next->clone());
        }

        if (params->strategy->sbReuse)
        {
            head.setLast(new MmeStackCmd_ReuseConv());
        }

        if (params->strategy->loweringEn)
        {
            head.setLast(new MmeStackCmd_LowerM());
        }

        head.setLast(new MmeStackCmd_BuildDesc());

        head.execute(&new_params, descList);
        TRACE_EXIT;
    }
};

class MmeStackCmd_StrategyDedw : public MmeStackCmd
{
public:
    MmeStackCmd_StrategyDedw() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_StrategyDedw); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        ExecuteParams new_params = *params;
        MME_ASSERT(params->opType == e_mme_dedw, "invalid pattern");

        switch (params->strategy->pattern)
        {
            case EMmePattern::e_mme_sp_reduction_kfc:
                break;
            case EMmePattern::e_mme_sp_reduction_fkc:
                break;
            case EMmePattern::e_mme_sp_reduction_fck:
                break;
            case EMmePattern::e_mme_sp_reduction_cfk:
                break;
            case EMmePattern::e_mme_sp_reduction_kcf:
                break;
            case EMmePattern::e_mme_sp_reduction_ckf:
                break;
            default:
                MME_ASSERT(0, "invalid walking pattern");
        }

        switch (params->strategy->geometry)
        {
            case e_mme_geometry_4wx1h:
                break;
            case e_mme_geometry_2wx2h:
                break;
            case e_mme_geometry_1wx4h:
                break;
            default:
                MME_ASSERT(0, "invalid geometry");
        }

        GeoParams geoParams;
        geo2Params(params->strategy->geometry, params->a->elementType, &geoParams);

        unsigned filter = params->b->sizes[2];
        for (unsigned i=3; i<Mme::c_mme_max_tensor_dims; i++)
        {
            filter = std::max(filter, params->b->sizes[i]);
        }

        bool hasDilation = false;
        for (unsigned dim = 0; dim < Mme::c_mme_max_conv_dims - 1; dim++)
        {
            if (params->conv->dilation[dim] != 1)
            {
                hasDilation = true;
                break;
            }
        }
        bool unroll = ((params->strategy->unrollEn) && (params->strategy->geometry == e_mme_geometry_4wx1h) &&
                       (params->c->sizes[0] <= geoParams.subMatrixWidth) && (filter > 1)) &&
                      (!hasDilation);

        MmeStackCmd head;
        if (m_next)
        {
            head.setNext(m_next->clone());
        }

        if (!unroll && params->strategy->sbReuse &&
            // TODO: remove when the below patterns are supported
            (params->strategy->pattern != EMmePattern::e_mme_sp_reduction_kcf) &&
            (params->strategy->pattern != EMmePattern::e_mme_sp_reduction_ckf) &&
            (params->strategy->pattern != EMmePattern::e_mme_sp_reduction_cfk))
        {
            head.setLast(new MmeStackCmd_ReuseDedw());
        }
        else if (params->strategy->loweringEn)
        {
            head.setLast(new MmeStackCmd_LowerA());
        }

        if (unroll)
        {
            head.setLast(new MmeStackCmd_WeightUnroll());
        }

        head.setLast(new MmeStackCmd_BuildDEDWDesc());

        head.execute(&new_params, descList);

        TRACE_EXIT;
    }
};

class MmeStackCmd_TraceEvent : public MmeStackCmd
{
public:
    MmeStackCmd_TraceEvent() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_TraceEvent); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);

        switch(params->tracing->traceModeX)
        {
            case e_mme_trace_mode_none:
                break;
            case e_mme_trace_mode_layer_act:
                setOpStartAndEndEvents(params, descList, e_mme_op_x);
                break;
            default:
                MME_ASSERT(0, "invalid tracing mode");
        }

        switch(params->tracing->traceModeY)
        {
            case e_mme_trace_mode_none:
                break;
            case e_mme_trace_mode_layer_act:
                setOpStartAndEndEvents(params, descList, e_mme_op_y);
                break;
            default:
                MME_ASSERT(0, "invalid tracing mode");
        }

        switch(params->tracing->traceModeW)
        {
            case e_mme_trace_mode_none:
                break;
            case e_mme_trace_mode_layer_act:
                setOpStartAndEndEvents(params, descList, e_mme_op_w);
                break;
            default:
                MME_ASSERT(0, "invalid tracing mode");
        }

        TRACE_EXIT;
    }

    static void setOpStartAndEndEvents(const ExecuteParams* params, DescList* descList, const EMmeOperand userOperand)
    {
        TRACE_ENTER;
        Mme::MmePerfEvt evt = {0};
        evt.incMask = 1;
        set_bf_to_all_ones(evt.loopMask);
        evt.rst = 1;
        evt.value = params->tracing->ctxId;

        for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
        {
            evt.startEndMask = (descList->size() == 1) ? 0x0 : 0x2;
            Mme::Desc *desc;
            desc = &descList->front().desc[masterIdx];
            EPhysOperand operand = mmeOperand2PhysOperand(userOperand, params->opType, desc->header.transO);

            if (operand == OP_S)
            {
                desc->perfEvtS.dw = evt.dw;
            }
            else if (operand == OP_L)
            {
                desc->perfEvtL[Mme::e_mme_local].dw = evt.dw;
                desc->perfEvtL[Mme::e_mme_remote].dw = evt.dw;
            }
            else
            {
                desc->perfEvtO[Mme::e_mme_remote].dw = evt.dw;
                desc->perfEvtO[Mme::e_mme_local].dw = evt.dw;
            }

            if (descList->size() != 1)
            {
                desc = &descList->back().desc[masterIdx];
                evt.startEndMask = 1; // mask start

                if (operand == OP_S)
                {
                    desc->perfEvtS.dw = evt.dw;
                }
                else if (operand == OP_L)
                {
                    desc->perfEvtL[Mme::e_mme_local].dw = evt.dw;
                    desc->perfEvtL[Mme::e_mme_remote].dw = evt.dw;
                }
                else
                {
                    desc->perfEvtO[Mme::e_mme_remote].dw = evt.dw;
                    desc->perfEvtO[Mme::e_mme_local].dw = evt.dw;
                }
            }
        }
        TRACE_EXIT;
    }
};

class MmeStackCmd_SplitDedx : public MmeStackCmd
{
public:

    MmeStackCmd_SplitDedx() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_SplitDedx); }

    void verifyDedxParams(const ExecuteParams* params)
    {
#ifndef NDEBUG
        MME_ASSERT(params->opType == e_mme_dedx, "only valid for dedx");

        // stride and dilation with common gcd > 1 is not supported in old gaudi flow
        for (unsigned dim = 0; dim < Mme::c_mme_max_conv_dims - 1; dim++)
        {
            unsigned gcd = std::__gcd(params->conv->stride[dim], params->conv->dilation[dim]);
            if (gcd != 0 && gcd != 1)
            {
                MME_ASSERT(0, "not yet implemented");
            }
        }

        for (unsigned dim = 0; dim<Mme::c_mme_max_tensor_dims; dim++)
        {
            // roi in dedx is still not supported.
            MME_ASSERT(params->c->bases[dim] == 0, "roi in dedx is still not supported");

            // partial filter in dedx is still not supported
            MME_ASSERT(params->b->bases[dim] == 0, "partial filter in dedx is still not supported");
        }
#endif
    }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;

        verifyDedxParams(params);

        unsigned spSize = 1;
        for (unsigned dim = 1; dim < Mme::c_mme_max_tensor_dims; dim++)
        {
            spSize *= params->c->sizes[dim];
        }

        // roi is still not supported in dedx
        MME_ASSERT(spSize == params->roi->spSize, "roi is still not supported in dedx");
        MME_ASSERT(params->roi->spBase == 0, "roi is still not supported in dedx");

        unsigned wDataSize =
            (params->b->elementType == EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);
        unsigned xDataSize =
            (params->c->elementType == EMmeDataType::e_type_bf16) ? sizeof(Mme::bf16_t) : sizeof(Mme::f32_t);

        int numClasses = 1;
        for (unsigned i = 0; i < c_padded_conv_dim; i++)
        {
            numClasses *= params->conv->stride[i];
        }

        if (params->strategy->packingFactor > 1)
        {
            //  currently packing is supported only for convolutions without strides.
            //  packing increases the conv stride by packing factor on the first dimension.
            //  after packing we expect the conv strides to be [packingFactor, 1, 1]
            //  thus we expect numClasses to be equal to the packingFactor.
            MME_ASSERT(numClasses == params->strategy->packingFactor, "conv stride is not yet supported with dedx packing");

            MmeConv new_conv = *params->conv;
            new_conv.padding[0] += params->strategy->packingFactor - 1;

            ExecuteParams new_params = *params;
            new_params.conv = &new_conv;

            MME_ASSERT_PTR(m_next);
            m_next->execute(&new_params, descList);
            TRACE_EXIT;
            return;
        }

        for (unsigned wc = 0; wc < numClasses; wc++)
        {
            MmeControls new_controls = *params->controls;
            new_controls.reluEn = false;
            MmeConv new_conv;
            MmeTensorView new_x = *params->c;
            MmeTensorView new_w = *params->b;
            MmeRoi new_roi = *params->roi;
            ExecuteParams new_params = *params;
            new_params.controls = &new_controls;
            new_params.conv = &new_conv;
            new_params.c = &new_x;
            new_params.b = &new_w;
            new_params.roi = &new_roi;

            int rem = wc;
            for (int convDim = c_padded_conv_dim - 1; convDim >= 0; convDim--)
            {
                unsigned weightDim = convDim + 2;
                new_w.bases[weightDim] = rem % params->conv->stride[convDim];
                new_params.bPtr.u64 += new_w.bases[weightDim] * params->b->strides[weightDim] * wDataSize;
                rem /= params->conv->stride[convDim];
            }

            bool memsetClass = false;

            for (unsigned convDim = 0; convDim < c_padded_conv_dim; convDim++)
            {
                unsigned tensorDim = convDim + 1;
                unsigned weightDim = convDim + 2;
                new_conv.dilation[convDim] = params->conv->dilation[convDim];
                new_conv.stride[convDim] = 1;
                int RDMinusP = (params->conv->dilation[convDim] * new_w.bases[weightDim]) - params->conv->padding[convDim];
                new_conv.padding[convDim] = -div_round_down(RDMinusP, params->conv->stride[convDim]);
                new_x.bases[tensorDim] = mod_neg(RDMinusP, params->conv->stride[convDim]);
                new_params.cPtr.u64 += new_x.bases[tensorDim] * params->c->strides[tensorDim] * xDataSize;

                unsigned remX = new_x.bases[tensorDim];
                new_x.strides[tensorDim] *= params->conv->stride[convDim];
                new_x.sizes[tensorDim] /= params->conv->stride[convDim];
                new_x.bases[tensorDim] /= params->conv->stride[convDim];

                unsigned remW = new_w.bases[weightDim] % params->conv->stride[convDim];
                new_w.strides[weightDim] *= params->conv->stride[convDim];
                new_w.sizes[weightDim] /= params->conv->stride[convDim];
                new_w.bases[weightDim] /= params->conv->stride[convDim];

                if (remX < (params->c->sizes[tensorDim] % params->conv->stride[convDim]))
                {
                    new_x.sizes[tensorDim]++;
                }

                if (remW < (params->b->sizes[weightDim] % params->conv->stride[convDim]))
                {
                    new_w.sizes[weightDim]++;
                }

                if (!new_x.sizes[tensorDim])
                {
                    break;
                }

                // roi is still not supported in dedx. the ROI is set to the entire x view
                new_roi.denseBase = new_x.bases[0];
                new_roi.spBase = 0;
                new_roi.spSize = 1;
                for (unsigned i=1; i<Mme::c_mme_max_tensor_dims; i++)
                {
                    new_roi.spSize *= new_x.sizes[i];
                }
                memcpy(&new_roi.size[1], &new_x.sizes[1], sizeof(new_roi.size) - sizeof(new_roi.size[0]));
                new_roi.size[0] = new_x.sizes[0];

                if (!new_w.sizes[weightDim])
                {
                    if (params->strategy->memsetDedxVoidPixels)
                    {
                        memsetClass = true;
                    }
                    else
                    {
                        break;
                    }
                }

                if (convDim == c_padded_conv_dim - 1)
                {
                    if (memsetClass)
                    {
                        MmeStackCmd_BuildMemsetDesc memsetCmd;
                        memsetCmd.execute(&new_params, descList);
                    }
                    else
                    {
                        MME_ASSERT_PTR(m_next);
                        m_next->execute(&new_params, descList);
                    }
                }
            }
        }
        TRACE_EXIT;
    }
};

class MmeStackCmd_SetSignals : public MmeStackCmd
{
public:

    MmeStackCmd_SetSignals() : MmeStackCmd() {}

    virtual MmeStackCmd *clone() { MME_CMD_CLONE(MmeStackCmd_SetSignals); }

    void execute(ExecuteParams *params, DescList *descList)
    {
        TRACE_ENTER;
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);

        for (auto &descs : *descList)
        {
            if ((params->controls->signalingMode == e_mme_signaling_none) ||
                (params->controls->signalingMode == e_mme_signaling_once))
            {
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalEn = 0;
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalEn = 0;
            }
            else if (params->controls->signalingMode == e_mme_signaling_desc)
            {
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalEn = 1;
                set_bf_to_all_ones(descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalMask);
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalEn = 1;
                set_bf_to_all_ones(descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalMask);
            }
            else if (params->controls->signalingMode == e_mme_signaling_desc_with_store)
            {
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalEn = descs.desc[Mme::MME_CORE_NORTH_MASTER].header.storeEn;
                set_bf_to_all_ones(descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalMask);
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalEn = descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.storeEn;
                set_bf_to_all_ones(descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalMask);
            }
            else if (params->controls->signalingMode == e_mme_signaling_output)
            {
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalEn = descs.desc[Mme::MME_CORE_NORTH_MASTER].header.storeEn;
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalMask = descs.desc[Mme::MME_CORE_NORTH_MASTER].header.accumMask;
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalEn = descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.storeEn;
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalMask = descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.accumMask;
            }
            else if (params->controls->signalingMode == e_mme_signaling_partial)
            {
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalEn = 1;
                descs.desc[Mme::MME_CORE_NORTH_MASTER].header.signalMask = descs.desc[Mme::MME_CORE_NORTH_MASTER].header.accumMask;
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalEn = 1;
                descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.signalMask = descs.desc[Mme::MME_CORE_SOUTH_MASTER].header.accumMask;
            }
            else
            {
                MME_ASSERT(0, "invalid signaling mode");
            }
        }

        if ((params->controls->signalingMode == e_mme_signaling_once) && (!descList->empty()))
        {
            descList->back().desc[Mme::MME_CORE_NORTH_MASTER].header.signalEn = 1;
            set_bf_to_all_ones(descList->back().desc[Mme::MME_CORE_NORTH_MASTER].header.signalMask);
            descList->back().desc[Mme::MME_CORE_SOUTH_MASTER].header.signalEn = 1;
            set_bf_to_all_ones(descList->back().desc[Mme::MME_CORE_SOUTH_MASTER].header.signalMask);
        }
        TRACE_EXIT;
    }
};

void generateDescriptors(const MmeLayerParams& params, std::list<MmeActivation>& activations)
{
    TRACE_ENTER;
    MME_ASSERT(0, "dead code should not reach here");
    // run some basic checks
    MME_ASSERT(params.x.sizes[0] == params.w.sizes[1], "common dim should be the same");
    MME_ASSERT(params.y.sizes[0] == params.w.sizes[0], "output width should be the same");

    if (params.opType == e_mme_fwd)
    {
        MME_ASSERT(params.x.elementType == params.w.elementType, "input element type should be the same");
    }
    else if (params.opType == e_mme_dedx)
    {
        MME_ASSERT(params.w.elementType == params.y.elementType, "input element type should be the same");
    }
    else
    {
        MME_ASSERT(params.x.elementType == params.y.elementType, "input element type should be the same");
    }

    if (params.controls.conversionRoundingMode != RoundingMode::StochasticRounding)
    {
        MME_ASSERT(params.controls.roundingMode == params.controls.conversionRoundingMode,
                  "rounding mode should be the same");
    }

    MmeStackCmd head;
    head.setNext(new MmeStackCmd_SetSignals());
    head.setNext(new MmeStackCmd_TraceEvent());

    if (params.opType == e_mme_dedw)
    {
        head.setLast(new MmeStackCmd_StrategyDedw());
    }
    else
    {
        if (params.opType == e_mme_dedx)
        {
            head.setLast(new MmeStackCmd_SplitDedx());
        }
        head.setLast(new MmeStackCmd_StrategyConv());
    }

    ExecuteParams execParams(params.opType);
    MmeRoi roi;
    MmeSpatialSlice spatialSlice;
    if (params.opType == e_mme_dedw)
    {
        execParams.roi = nullptr;
        spatialSlice.spBase = params.spBase;
        spatialSlice.spSize = params.spSize;
        execParams.spatialSlice = &spatialSlice;
    }
    else
    {
        execParams.spatialSlice = nullptr;
        roi.spBase = params.spBase;
        roi.spSize = params.spSize;

        const MmeTensorView* view = (params.opType == e_mme_dedx) ? &params.x : &params.y;
        roi.denseBase = view->bases[0];
        memcpy(&roi.size[1], &view->sizes[1], sizeof(roi.size) - sizeof(roi.size[0]));
        roi.size[0] = view->sizes[0];

        execParams.roi = &roi;
    }

    execParams.aPtr.u64 = 0;
    execParams.bPtr.u64 = 0;
    execParams.cPtr.u64 = 0;
    execParams.opType = params.opType;
    execParams.conv = &params.conv;

    if (params.opType == e_mme_fwd)
    {
        execParams.a = &params.x;
        execParams.b = &params.w;
        execParams.c = &params.y;
    }
    else if (params.opType == e_mme_dedx)
    {
        execParams.a = &params.y;
        execParams.b = &params.w;
        execParams.c = &params.x;
    }
    else if (params.opType == e_mme_dedw)
    {
        execParams.a = &params.x;
        execParams.b = &params.y;
        execParams.c = &params.w;
    }
    else
    {
        MME_ASSERT(0, "invalid op type");
    }

    execParams.controls = &params.controls;
    execParams.strategy = &params.strategy;
    execParams.tracing = &params.tracing;

    DescList descList;
    head.execute(&execParams, &descList);
    TRACE_EXIT;
}

static void patchTensor(const uint64_t addr, uint32_t *high, uint32_t *low)
{
    TRACE_ENTER;
    ptrToInt ptrToAddr;
    ptrToAddr.u32[0] = *low;
    ptrToAddr.u32[1] = *high;
    ptrToAddr.u64 += addr;
    *low = ptrToAddr.u32[0];
    *high = ptrToAddr.u32[1];
    TRACE_EXIT;
}

void getTensorAddressFieldsAndMapping(const EMmeOperand operand,
                                      Mme::Desc& desc,
                                      uint32_t*& addrHigh,
                                      uint32_t*& addrLow,
                                      const Mme::MmeTensorDesc*& tensorDesc,
                                      bool isBgemm)
{
    TRACE_ENTER;
    bool isDedx = (!isBgemm && desc.header.transS && desc.header.transL);
    bool isDedw = (!isBgemm && !desc.header.transS && !desc.header.transL);
    bool isFwdOrBgemm = (isBgemm || (desc.header.transS != desc.header.transL));
    MME_ASSERT(isDedx || isDedw || isFwdOrBgemm, "One of these operations must be set");
    if ((isDedx && isDedw) || (isDedx && isFwdOrBgemm) || (isDedw && isFwdOrBgemm))
    {
        MME_ASSERT(0, "Only one of these operations can be set");
    }

    switch (operand)
    {
        case e_mme_op_x:
        {
            if (desc.sw.swMemsetFwd || desc.sw.swMemsetDedw)
            {
                addrHigh = nullptr;
                addrLow = nullptr;
                tensorDesc = nullptr;
            }
            else if (desc.sw.swMemsetDedx || isDedx)
            {
                addrHigh = &desc.baseAddrHighO;
                addrLow = &desc.baseAddrLowO;
                tensorDesc = &desc.tensorO;
            }
            else
            {
                addrHigh = desc.header.transO ? &desc.baseAddrHighL : &desc.baseAddrHighS;
                addrLow = desc.header.transO ? &desc.baseAddrLowL : &desc.baseAddrLowS;
                tensorDesc = desc.header.transO ? &desc.tensorL : &desc.tensorS;
            }
            break;
        }
        case e_mme_op_w:
        {
            if (desc.sw.swMemsetFwd || desc.sw.swMemsetDedx)
            {
                addrHigh = nullptr;
                addrLow = nullptr;
                tensorDesc = nullptr;
            }
            else if (desc.sw.swMemsetDedw || isDedw)
            {
                addrHigh = &desc.baseAddrHighO;
                addrLow = &desc.baseAddrLowO;
                tensorDesc = &desc.tensorO;
            }
            else
            {
                addrHigh = desc.header.transO ? &desc.baseAddrHighS : &desc.baseAddrHighL;
                addrLow = desc.header.transO ? &desc.baseAddrLowS : &desc.baseAddrLowL;
                tensorDesc = desc.header.transO ? &desc.tensorS : &desc.tensorL;
            }
            break;
        }
        case e_mme_op_y:
        {
            if (desc.sw.swMemsetDedw || desc.sw.swMemsetDedx)
            {
                addrHigh = nullptr;
                addrLow = nullptr;
                tensorDesc = nullptr;
            }
            else if (desc.sw.swMemsetFwd || isFwdOrBgemm)
            {
                addrHigh = &desc.baseAddrHighO;
                addrLow = &desc.baseAddrLowO;
                tensorDesc = &desc.tensorO;
            }
            else if (desc.header.transS == desc.header.transO)
            {
                addrHigh = &desc.baseAddrHighL;
                addrLow = &desc.baseAddrLowL;
                tensorDesc = &desc.tensorL;
            }
            else
            {
                addrHigh = &desc.baseAddrHighS;
                addrLow = &desc.baseAddrLowS;
                tensorDesc = &desc.tensorS;
            }
            break;
        }
        default:
        {
            MME_ASSERT(0, "invalid operand");
        }
    };
    TRACE_EXIT;
}

void getTensorAddressFields(const EMmeOperand operand,
                            Mme::Desc& desc,
                            uint32_t** addrHigh,
                            uint32_t** addrLow,
                            const bool isBgemm)
{
    TRACE_ENTER;
    MME_ASSERT_PTR(addrHigh);
    MME_ASSERT_PTR(addrLow);
    const Mme::MmeTensorDesc* dummyDesc;

    getTensorAddressFieldsAndMapping(operand, desc, *addrHigh, *addrLow, dummyDesc, isBgemm);

    TRACE_EXIT;
}

bool getOperandMapping(const EMmeOperand operand, Mme::Desc& desc, const Mme::MmeTensorDesc*& tensorDesc, bool isBgemm)
{
    TRACE_ENTER;
    uint32_t* dummyHigh;
    uint32_t* dummyLow;

    getTensorAddressFieldsAndMapping(operand, desc, dummyHigh, dummyLow, tensorDesc, isBgemm);

    TRACE_EXIT;
    return (tensorDesc != nullptr);
}

void patchTensorView(const EMmeOperand operand,
                     Mme::Desc& desc0,
                     Mme::Desc& desc1,
                     const uint64_t addr,
                     const bool isBgemm)
{
    TRACE_ENTER;
    uint32_t *addrHigh;
    uint32_t *addrLow;

    getTensorAddressFields(operand, desc0, &addrHigh, &addrLow, isBgemm);
    if (addrLow && addrHigh)
    {
        patchTensor(addr, addrHigh, addrLow);
    }

    getTensorAddressFields(operand, desc1, &addrHigh, &addrLow, isBgemm);
    if (addrLow && addrHigh)
    {
        patchTensor(addr, addrHigh, addrLow);
    }
    TRACE_EXIT;
}

void patchSyncObject(
    Mme::Desc &desc,
    const Mme::MmeHalf half,
    const uint64_t addr,
    const unsigned value,
    const bool inc,
    const bool perfEvent)
{
    TRACE_ENTER;

    ptrToInt ptrToAddr;
    ptrToAddr.u64 = addr;

    // address high
    desc.syncObject.addrHigh = ptrToAddr.u32[1];

    // address low
    MME_ASSERT((half == Mme::e_mme_local) || (half == Mme::e_mme_remote), "should be local or remote");
    desc.syncObject.addrLow[half] = ptrToAddr.u32[0];

    // data
    desc.syncObject.value = value;
    desc.syncObject.operation = inc ? 1 : 0;
    desc.syncObject.perfEn = perfEvent ? 1 : 0;
    TRACE_EXIT;
}

void getRoiBaseOffsetFields(const MmeCommon::EMmeOperand operand,
                            const MmeCommon::EMmeOpType opType,
                            Mme::Desc& desc,
                            std::vector<int32_t*>& roiBaseOffsetVec)
{
    switch (mmeOperandToGaudiAgu(operand, opType, desc.header.transO))
    {
        case MmeCommon::e_mme_agu_shared:
            roiBaseOffsetVec.push_back(&desc.aguS.roiBaseOffset[0]);
            break;
        case MmeCommon::e_mme_agu_local:
            roiBaseOffsetVec.push_back(&desc.aguL[Mme::e_mme_local].roiBaseOffset[0]);
            roiBaseOffsetVec.push_back(&desc.aguL[Mme::e_mme_remote].roiBaseOffset[0]);
            break;
        case MmeCommon::e_mme_agu_out:
            roiBaseOffsetVec.push_back(&desc.aguO[Mme::e_mme_local].roiBaseOffset[0]);
            roiBaseOffsetVec.push_back(&desc.aguO[Mme::e_mme_remote].roiBaseOffset[0]);
            break;
    }
}
void patchPadding(int32_t* roiBaseOffset,
                  uint32_t* tensorAStrides,
                  uint32_t* oldConvPadding,
                  uint32_t* oldConvStrides,
                  uint32_t* oldConvDilation,
                  uint32_t* newConvPadding,
                  MmeCommon::EMmeOpType opType)
{
    for (unsigned convIdx = 0; convIdx < Mme::c_mme_max_conv_dims - 1; convIdx++)
    {
        if (newConvPadding[convIdx] == oldConvPadding[convIdx]) continue;
        unsigned gcd = std::gcd(oldConvStrides[convIdx], oldConvDilation[convIdx]);
        MME_ASSERT(gcd == 0 || gcd == 1, "Stride and dilation has GCD - cannot apply dynamic newPadding");
        int spatialIdx = convIdx + 1;
        int32_t offsetDif = (oldConvPadding[convIdx] - newConvPadding[convIdx]) * tensorAStrides[spatialIdx];
        roiBaseOffset[spatialIdx] += offsetDif;
    }
}
}  // namespace gaudi
