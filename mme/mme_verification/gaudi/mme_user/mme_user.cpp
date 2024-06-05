
#undef NDEBUG
#include "gaudi/headers/mme_user.h"
#include "print_utils.h"
#include "convolution_params.h"
#include "gaudi/headers/tensor_utils.h"
#include "gaudi/mme_descriptor_generator.h"
#include "mme_params_factory.h"
#include "gaudi/gaudi_utils.h"
#include "mme_reference.h"
#include "common/mme_test_global_conf.h"
#include <bitset>
#include <climits>

// new mme stack
#include "gaudi/headers/mme_descriptor_comparator.h"

using namespace MmeCommon;
using namespace gaudi;

#undef min
#undef max

#define _IN_
#define _IO_
#define _OUT_

#define DESC_SIZE_IN_DW       (sizeof(Mme::Desc) / sizeof(uint32_t))
#define MME_REG_OFFSET(field) ((size_t) & (((Mme::RegBlock*) 0)->field))

static const uint32_t ONE = 1;
static const uint32_t ZERO = 0;
static const unsigned c_operand_max_agu = 4;
static const unsigned c_cl_size = 128;

static const MmeTestParams c_default_test_params = {false, false, 1, false, false, false};

static void pushCmd(std::list<MmeRegWriteCmd>* cmds, const size_t offset, const unsigned size, const void* const value)
{
    MmeRegWriteCmd cmd;
    MME_ASSERT(size * sizeof(uint32_t) <= sizeof(cmd.reg_values), "invalid cmd size");
    memcpy(cmd.reg_values, value, size * sizeof(uint32_t));
    cmd.num_regs = size;
    cmd.reg_offset = (unsigned) offset;
    cmds->push_back(cmd);
}

union RegValue
{
    struct
    {
        uint32_t offset;
        uint32_t value;
    };
    uint64_t ddw;
};

typedef union
{
    void* p;
    uint64_t ddw;
    uint32_t dw[2];
} ptr64;

int getTargetSOValue(Mme::Desc* desc)
{
    int ret;

    if (!desc->header.signalEn)
    {
        ret = 0;
    }
    else if (desc->header.accStoreIncDisable)
    {
        ret = 0;
    }
    else if (!desc->syncObject.operation)
    {
        ret = desc->syncObject.value;
    }
    else
    {
        ret = desc->syncObject.value;

        for (int d = 0; d < Mme::c_mme_max_conv_dims; d++)
        {
            if ((desc->header.signalMask & (1 << d)) == 0)
            {
                ret *= desc->conv.kernelSizeMinus1.dim[d] + 1;
            }
        }

        if ((desc->header.signalMask & (1 << Mme::c_mme_max_conv_dims)) == 0)
        {
            ret *= desc->numIterationsMinus1 + 1;
        }

        if ((desc->header.signalMask & (1 << (Mme::c_mme_max_conv_dims + 1))) == 0)
        {
            ret *= desc->outerLoop.sizeMinus1 + 1;
        }
    }

    return ret;
}

class AguSim
{
private:
    static const unsigned c_cb_cache_size = 8;

    struct FixedAguParams
    {
        const Mme::MmeTensorDesc* tensorDesc;
        uint64_t baseAddr;
        int accumMask;
        int signalMask;
        int partialHeightLoopMask;
        int fcdLoopMask;
        int elementSize;
        int height;
        int heightLast;
        unsigned fcd;
        unsigned fcdLast;
        bool lower;
        unsigned loopsNum;
        bool isInput;
        bool isShared;
    };

    struct AguDescStat
    {
        unsigned numVectors;
        unsigned numAccesses;
        unsigned numCycles;
        unsigned numSBEntries;
    };

    static void genPCLoops(const Mme::Desc* desc, bool isInput, bool isShared, std::list<unsigned>* loops)
    {
        unsigned loopSize[Mme::c_mme_max_conv_dims + 2];
        unsigned accLoopSize[Mme::c_mme_max_conv_dims + 2];
        unsigned mask;

        if (isShared)
        {
            mask = desc->sbRepeat.aguSLoopMask;
        }
        else if (!isInput)
        {
            mask = desc->header.accumMask;
        }
        else
        {
            mask = desc->sbRepeat.aguLLoopMask;
        }

        for (int i = 0; i < Mme::c_mme_max_conv_dims; i++)
        {
            loopSize[i] = desc->conv.kernelSizeMinus1.dim[i];
        }

        loopSize[Mme::c_mme_max_conv_dims] = desc->numIterationsMinus1;
        loopSize[Mme::c_mme_max_conv_dims + 1] = desc->outerLoop.sizeMinus1;

        unsigned loopsNr = 1;
        for (int i = 0; i < sizeof(loopSize) / sizeof(loopSize[0]); i++)
        {
            loopSize[i] = (mask & (1 << i) ? 0 : loopSize[i]) + 1;
            accLoopSize[i] = loopsNr;
            loopsNr *= loopSize[i];
        }

        for (unsigned j = 0; j < loopsNr; j++)
        {
            unsigned rem = j;
            unsigned end = 0;
            for (int i = 0; i < sizeof(loopSize) / sizeof(loopSize[0]); i++)
            {
                end |= ((rem % loopSize[i]) == (loopSize[i] - 1)) << i;
                rem /= loopSize[i];
            }
            loops->push_back(end);
        }
    }

    static void genGemmStat(const FixedAguParams* fp,
                            const bool isInput,
                            const bool isShared,
                            const bool advance,
                            unsigned matSize,
                            const int64_t* startOffsets,  // The current dim offsets.
                            int64_t* nextStartOffsets,  // The next dim offsets.
                            const int64_t* roiBase,
                            unsigned loopStartMask,
                            unsigned loopEndMask,
                            std::list<uint64_t>* cache,
                            std::list<AguDescStat>* stats)
    {
        int height = (loopEndMask & fp->partialHeightLoopMask) ? fp->heightLast : fp->height;
        unsigned fcd = (loopEndMask & fp->fcdLoopMask) ? fp->fcdLast : fp->fcd;

        int elementsInCL = c_cl_size / fp->elementSize;
        int elementsInTeWindow = matSize;
        int cl_nr = ((fp->tensorDesc->roiSize[0] * fp->elementSize) + c_cl_size - 1) / c_cl_size;
        int64_t currOffset[Mme::c_mme_max_tensor_dims];
        int64_t targetOffset[Mme::c_mme_max_tensor_dims - 1];

        bool cacheInvalidate = false;
        bool newConv = true;
        if (isInput)
        {
            newConv = ((fp->accumMask & loopStartMask) == fp->accumMask);
            bool newDesc = (loopStartMask & ((1 << (Mme::c_mme_max_conv_dims + 2)) - 1)) ==
                           ((1 << (Mme::c_mme_max_conv_dims + 2)) - 1);
            cacheInvalidate = newDesc || (newConv && fp->loopsNum);
        }

        if (newConv)
        {
            stats->push_back({0});
        }

        AguDescStat* currStats = &stats->back();

        targetOffset[0] = roiBase[0] + fcd;
        for (int i = 1; i < Mme::c_mme_max_tensor_dims - 1; i++)
        {
            targetOffset[i] = roiBase[i] + fp->tensorDesc->roiSize[i];
        }

        for (currOffset[0] = roiBase[0]; currOffset[0] < targetOffset[0]; currOffset[0] += elementsInCL)
        {
            for (int i = 1; i < Mme::c_mme_max_tensor_dims; i++)
            {
                currOffset[i] = roiBase[i] + startOffsets[i - 1];
            }

            bool lastGemmCol = (currOffset[0] + elementsInCL >= targetOffset[0]);
            bool trans1Pass = (currOffset[0] + elementsInTeWindow >= targetOffset[0]);

            MME_ASSERT(!trans1Pass || lastGemmCol, "");

            for (int spStep = 0; spStep < height; spStep++)
            {
                int64_t denseOffset;
                int64_t denseTarget;
                if (fp->lower)
                {
                    denseOffset = currOffset[0] + currOffset[1];
                    denseTarget = std::min((int64_t)(currOffset[1] + roiBase[0] + fcd),
                                           (int64_t)(fp->tensorDesc->validElements[1]));
                    denseTarget = std::min(denseTarget, currOffset[1] + fp->tensorDesc->validElements[0]);
                }
                else
                {
                    denseOffset = currOffset[0];
                    denseTarget = std::min((int64_t) targetOffset[0], (int64_t) fp->tensorDesc->validElements[0]);
                }

                bool pad = false;
                for (int spDim = fp->lower ? 2 : 1; spDim < Mme::c_mme_max_tensor_dims; spDim++)
                {
                    if ((currOffset[spDim] < 0) || (currOffset[spDim] >= fp->tensorDesc->validElements[spDim]))
                    {
                        pad = true;
                        break;
                    }
                }

                if (cacheInvalidate)
                {
                    cache->clear();
                    cacheInvalidate = false;
                }

                int64_t denseEndOffset = denseOffset + elementsInCL;
                int64_t padMsb = denseEndOffset > denseTarget ? denseEndOffset - denseTarget : 0;
                int64_t padLsb = denseOffset < 0 ? -denseOffset : 0;

                currStats->numVectors++;
                currStats->numSBEntries++;
                if (pad || (padLsb >= elementsInCL) || (padMsb >= elementsInCL))
                {
                    currStats->numCycles++;
                }
                else
                {
                    uint64_t p = fp->baseAddr;

                    for (int dim = 0; dim < Mme::c_mme_max_tensor_dims; dim++)
                    {
                        p += currOffset[dim] * fp->elementSize;
                    }

                    unsigned lpadInBytes = padLsb * fp->elementSize;
                    unsigned mpadInBytes = padMsb * fp->elementSize;

                    uint64_t p0 = (p + lpadInBytes) & ~(((uint64_t) c_cl_size) - 1);
                    bool p0Hit = false;
                    if (isInput)
                    {
                        for (auto pv : *cache)
                            p0Hit = p0Hit || (pv == p0);
                        if (!p0Hit)
                        {
                            if (cache->size() == c_cb_cache_size) cache->pop_front();
                            cache->push_back(p0);
                            currStats->numAccesses++;
                        }
                    }

                    uint64_t p1 = p0 + c_cl_size;
                    bool p1Hit;
                    if (p1 < (p + c_cl_size - mpadInBytes))
                    {
                        currStats->numSBEntries++;
                        p1Hit = false;
                        if (isInput)
                        {
                            for (auto pv : *cache)
                                p1Hit = p1Hit || (pv == p1);
                            if (!p1Hit)
                            {
                                if (cache->size() == c_cb_cache_size) cache->pop_front();
                                cache->push_back(p1);
                                currStats->numAccesses++;
                            }
                        }
                    }
                    else
                    {
                        p1Hit = true;
                    }

                    currStats->numCycles++;
                    if (!p0Hit && !p1Hit)
                    {
                        currStats->numCycles++;
                    }
                }

                int inc = 1;
                for (int spDim = 1; inc > 0; spDim++)
                {
                    currOffset[spDim] += (fp->tensorDesc->spatialStrides[spDim - 1] * inc);
                    inc = 0;
                    if (spDim < Mme::c_mme_max_tensor_dims - 1)
                    {
                        for (; currOffset[spDim] >= targetOffset[spDim];
                             currOffset[spDim] -= fp->tensorDesc->roiSize[spDim])
                        {
                            inc++;
                        }
                    }
                    MME_ASSERT(inc <= 1, "");
                }
            }
        }

        if (advance)
        {
            for (int i = 1; i < Mme::c_mme_max_tensor_dims; i++)
            {
                nextStartOffsets[i - 1] = currOffset[i] - roiBase[i];
            }
        }
    }

    static void
    getDescStat(unsigned aguId, bool isInput, bool isShared, const Mme::Desc* desc, std::list<AguDescStat>* stats)
    {
        bool advance;
        const Mme::MmeAguCoreDesc* aguDesc;
        bool enable;
        Mme::MmeAguCoreDesc dummyAguDesc;
        Mme::MmeTensorDesc dummyTensorDesc;

        unsigned matSize = Mme::c_mme_matrix_size >> ((desc->header.dataTypeIn == Mme::e_mme_dt_sp) ? 1 : 0);
        Mme::MmeHalf half = (aguId < Mme::MME_MASTERS_NR) ? Mme::e_mme_local : Mme::e_mme_remote;
        Mme::MmeAssociatedDims assocDimMask = {0};
        Mme::MmeAssociatedDims assocDimShift = {0};
        FixedAguParams fp;
        bool filterNonSignal = false;

        if (isInput && isShared)
        {
            assocDimMask.dimS = -1;
            assocDimShift.dimS = 1;
            fp.accumMask = desc->header.accumMask;
            fp.signalMask = -1;
            fp.elementSize = (desc->header.dataTypeIn == Mme::e_mme_dt_sp) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
            aguDesc = &desc->aguS;
            fp.baseAddr = (((uint64_t) desc->baseAddrHighS) << 32) + desc->baseAddrLowS;
            fp.tensorDesc = &desc->tensorS;
            advance = desc->header.advanceS;
            fp.lower = desc->header.lowerS;
            fp.loopsNum = desc->sbRepeat.repeatSMinus1;
            enable = desc->sbRepeat.loadS;
            if (desc->header.transS)
            {
                fp.partialHeightLoopMask = desc->header.partialHeightLoopS;
                fp.fcdLoopMask = 0;
                fp.height = matSize;
                fp.heightLast = (fp.tensorDesc->spatialSizeMinus1 + 1) % matSize;
                fp.heightLast = fp.heightLast ? fp.heightLast : matSize;
                fp.fcd = fp.tensorDesc->roiSize[0];
                fp.fcdLast = fp.fcd;
            }
            else
            {
                MME_ASSERT(desc->header.transO || (desc->numIterationsMinus1 == 0), "");
                fp.partialHeightLoopMask = 0;
                fp.fcdLoopMask = desc->header.partialHeightLoopS;
                fp.height = fp.tensorDesc->spatialSizeMinus1 + 1;
                fp.heightLast = fp.height;
                fp.fcd = matSize;
                fp.fcdLast = fp.tensorDesc->roiSize[0] % matSize;
                fp.fcdLast = fp.fcdLast ? fp.fcdLast : matSize;
            }
        }
        else if (isInput && !isShared)
        {
            assocDimMask.dimL = -1;
            assocDimShift.dimL = 1;
            fp.accumMask = desc->header.accumMask;
            fp.signalMask = -1;
            fp.elementSize = (desc->header.dataTypeIn == Mme::e_mme_dt_sp) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
            aguDesc = &desc->aguL[half];
            fp.baseAddr = (((uint64_t) desc->baseAddrHighL) << 32) + desc->baseAddrLowL;
            fp.tensorDesc = &desc->tensorL;
            advance = desc->header.advanceL;
            fp.lower = desc->header.lowerL;
            fp.loopsNum = desc->sbRepeat.repeatLMinus1;
            enable = desc->sbRepeat.loadL;
            if (desc->header.transL)
            {
                fp.partialHeightLoopMask = (half == Mme::e_mme_local) ? desc->header.partialHeightLoopLLocal
                                                                      : desc->header.partialHeightLoopLRemote;
                fp.fcdLoopMask = 0;
                fp.height = matSize;
                fp.heightLast = (fp.tensorDesc->spatialSizeMinus1 + 1) % matSize;
                fp.heightLast = fp.heightLast ? fp.heightLast : matSize;
                fp.fcd = fp.tensorDesc->roiSize[0];
                fp.fcdLast = fp.fcd;
            }
            else
            {
                MME_ASSERT(!desc->header.transO || (desc->numIterationsMinus1 == 0), "");
                fp.partialHeightLoopMask = 0;
                fp.fcdLoopMask = (half == Mme::e_mme_local) ? desc->header.partialHeightLoopLLocal
                                                            : desc->header.partialHeightLoopLRemote;
                fp.height = fp.tensorDesc->spatialSizeMinus1 + 1;
                fp.heightLast = fp.height;
                fp.fcd = matSize;
                fp.fcdLast = fp.tensorDesc->roiSize[0] % matSize;
                fp.fcdLast = fp.fcdLast ? fp.fcdLast : matSize;
            }
        }
        else
        {
            assocDimMask.dimO = -1;
            assocDimShift.dimO = 1;
            fp.accumMask = desc->header.accumMask;
            fp.signalMask = desc->header.signalEn ? desc->header.signalMask : -1;
            fp.elementSize = (desc->header.dataTypeOut == Mme::e_mme_dt_sp) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
            aguDesc = &desc->aguO[half];
            fp.baseAddr = (((uint64_t) desc->baseAddrHighO) << 32) + desc->baseAddrLowO;
            fp.tensorDesc = &desc->tensorO;
            advance = desc->header.advanceO;
            fp.lower = false;
            enable = !desc->header.accStoreIncDisable & (desc->header.storeEn || desc->header.signalEn);
            fp.loopsNum = 0;

            fp.partialHeightLoopMask = 1 << Mme::c_mme_max_conv_dims;
            fp.height = matSize;
            fp.heightLast = desc->tensorO.spatialSizeMinus1 + 1;
            fp.fcdLoopMask = (half == Mme::e_mme_local) ? desc->header.partialHeightLoopOLocal
                                                        : desc->header.partialHeightLoopORemote;
            fp.fcd = matSize;
            fp.fcdLast = desc->tensorO.roiSize[0] % matSize;
            fp.fcdLast = fp.fcdLast ? fp.fcdLast : matSize;

            if (!desc->header.storeEn)
            {
                fp.height = 1;
                fp.heightLast = 1;
                fp.fcd = 1;
                fp.fcdLast = 1;
                advance = false;
                dummyAguDesc.roiBaseOffset[1] = -1;
                aguDesc = &dummyAguDesc;
                memcpy(&dummyTensorDesc, fp.tensorDesc, sizeof(Mme::MmeTensorDesc));
                dummyTensorDesc.roiSize[0] = 1;
                fp.tensorDesc = &dummyTensorDesc;
            }
        }

        MME_ASSERT(fp.heightLast <= fp.height, "");

        if (enable)
        {
            std::list<unsigned> gemms;
            genPCLoops(desc, isInput, isShared, &gemms);

            unsigned associatedDim[Mme::c_mme_max_conv_dims + 2];
            for (int i = 0; i < Mme::c_mme_max_conv_dims; i++)
            {
                associatedDim[i] = (desc->conv.associatedDims[i].w & assocDimMask.w) / assocDimShift.w;
            }
            associatedDim[Mme::c_mme_max_conv_dims] = Mme::c_mme_max_tensor_dims;
            associatedDim[Mme::c_mme_max_conv_dims + 1] =
                (desc->outerLoop.associatedDims.w & assocDimMask.w) / assocDimShift.w;

            int64_t startOffsets[Mme::c_mme_max_tensor_dims - 1] = {0};
            int64_t nextStartOffsets[Mme::c_mme_max_tensor_dims - 1] = {0};
            int64_t roiBase[Mme::c_mme_max_tensor_dims];
            for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
            {
                roiBase[i] = aguDesc->roiBaseOffset[i];
            }

            std::list<uint64_t> cache;

            unsigned first = (1 << (Mme::c_mme_max_conv_dims + 2)) - 1;
            unsigned reset = (1 << (Mme::c_mme_max_conv_dims + 2)) - 1;

            for (auto& last : gemms)
            {
                if (!filterNonSignal || ((last & fp.signalMask) == fp.signalMask))
                {
                    for (int i = 0; i < Mme::c_mme_max_conv_dims + 2; i++)
                    {
                        unsigned dim = associatedDim[i];
                        if (reset & (1 << i))
                        {
                            if (dim < Mme::c_mme_max_tensor_dims)
                            {
                                roiBase[dim] = aguDesc->roiBaseOffset[dim];
                            }
                        }
                        else
                        {
                            if (dim < Mme::c_mme_max_tensor_dims)
                            {
                                roiBase[dim] += fp.tensorDesc->loopStride[dim];
                            }
                            first &= ~(1 << i);
                            break;
                        }
                    }

                    bool advanceStartOffset = advance && ((reset & ((1 << Mme::c_mme_max_conv_dims) - 1)) ==
                                                          ((1 << Mme::c_mme_max_conv_dims) - 1));

                    if (reset & (1 << Mme::c_mme_max_conv_dims))
                    {
                        for (int i = 0; i < Mme::c_mme_max_tensor_dims - 1; i++)
                        {
                            startOffsets[i] = aguDesc->startOffset[i];
                            nextStartOffsets[i] = aguDesc->startOffset[i];
                        }
                    }
                    else if (advanceStartOffset)
                    {
                        std::swap(startOffsets, nextStartOffsets);
                    }

                    genGemmStat(&fp,
                                isInput,
                                isShared,
                                matSize,
                                advanceStartOffset,
                                startOffsets,
                                nextStartOffsets,
                                roiBase,
                                first,
                                last,
                                &cache,
                                stats);
                }

                reset = last & ~(last + 1);
                first |= reset;
            }
        }
    }

public:
    static unsigned countSBReads(bool isShared, const Mme::Desc* northDesc, const Mme::Desc* southDesc)
    {
        unsigned reads = 0;
        if (northDesc->header.transO)
        {
            isShared = !isShared;
        }

        std::list<AguDescStat> stats;

        for (unsigned id = 0; id < (isShared ? Mme::MME_MASTERS_NR : Mme::MME_CORES_NR); id++)
        {
            getDescStat(id, true, isShared, (id & 0x1) ? southDesc : northDesc, &stats);
        }

        for (auto& stat : stats)
        {
            reads = std::max(reads, stat.numSBEntries);
        }

        return reads;
    }
};

typedef struct
{
    enum Pattern
    {
        PATTERN_NA = 0x00000000,
        PATTERN_S_REDUCTION_K_F_C = 0x0104ff00,
        PATTERN_S_REDUCTION_F_K_C = 0x0201ff00,
        PATTERN_S_REDUCTION_F_C_K = 0x0200ff01,
        PATTERN_S_REDUCTION_C_F_K = 0x0100ff04,
        PATTERN_S_REDUCTION_K_C_F = 0x0004ff03,
        PATTERN_S_REDUCTION_C_K_F = 0x0003ff04,
        PATTERN_C_REDUCTION_K_S_F = 0x000403ff,
        PATTERN_C_REDUCTION_S_K_F = 0x000304ff,
    };

    enum OP
    {
        OP_FWD,
        OP_DEDX,
        OP_DEDW,
    };

    struct MmeRoi
    {
        int spPos[Mme::c_mme_max_tensor_dims - 1];
        unsigned spSize;
        unsigned denseBase;
        unsigned denseSize;
    };

    struct InputRoi
    {
        unsigned base;
        unsigned size;
    };

    const ConvolutionParams* conv;
    uint32_t soAddrLow[Mme::MME_CORES_NR];
    uint32_t soAddrHigh;
    const MmeSimTensor* a;
    const MmeSimTensor* b;
    const MmeSimTensor* c;
    OP op;
    RoundingMode roundingMode;
    RoundingMode conversionRoundingMode;
    const MmeStrategy* strategy;
    EMmeGeometry selectedGeo;
    Pattern SelectedPattern;
    const MmeTestParams* testParams;
    MmeRoi roi;
} executeConvParams_t;

enum EOperand
{
    OP_S = 0x0,
    OP_L = 0x1,
    OP_O = 0x2,
};

#define swap_bf(a, b)                                                                                                  \
    {                                                                                                                  \
        auto ___t___ = (a);                                                                                            \
        (a) = (b);                                                                                                     \
        (b) = ___t___;                                                                                                 \
    }  // swap bit field

static void transposeDesc(Mme::Desc* desc)
{
    desc->header.transO = !desc->header.transO;

    std::swap(desc->baseAddrHighS, desc->baseAddrHighL);
    std::swap(desc->baseAddrLowS, desc->baseAddrLowL);
    swap_bf(desc->header.transS, desc->header.transL);
    swap_bf(desc->header.advanceS, desc->header.advanceL);
    swap_bf(desc->header.lowerS, desc->header.lowerL);
    for (int i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        swap_bf(desc->conv.associatedDims[i].dimS, desc->conv.associatedDims[i].dimL);
    }
    swap_bf(desc->outerLoop.associatedDims.dimS, desc->outerLoop.associatedDims.dimL);
    std::swap(desc->tensorS, desc->tensorL);
    std::swap(desc->paddingValueS, desc->paddingValueL);
}

static void setLoopSize(Mme::Desc* desc, executeConvParams_t::Pattern pattern, int loop, unsigned sizeMinus1)
{
    if (loop == LOOP_SPATIAL)
    {
        desc->numIterationsMinus1 = sizeMinus1;
    }
    else
    {
        int loopIdx = std::min(loop, (int) LOOP_FILTER);
        unsigned descLoopIdx = (((uint8_t*) &pattern)[loopIdx]) + (loop - loopIdx);
        if (descLoopIdx < Mme::c_mme_max_conv_dims)
        {
            desc->conv.kernelSizeMinus1.dim[descLoopIdx] = sizeMinus1;
        }
        else
        {
            desc->outerLoop.sizeMinus1 = sizeMinus1;
        }
    }
}

static unsigned getLoopSize(const Mme::Desc* desc, const executeConvParams_t::Pattern pattern, const int loop)
{
    if (loop == LOOP_SPATIAL)
    {
        return desc->numIterationsMinus1;
    }
    else
    {
        int loopIdx = std::min(loop, (int) LOOP_FILTER);
        unsigned descLoopIdx = (((uint8_t*) &pattern)[loopIdx]) + (loop - loopIdx);
        if (descLoopIdx < Mme::c_mme_max_conv_dims)
        {
            return desc->conv.kernelSizeMinus1.dim[descLoopIdx];
        }
        else
        {
            return desc->outerLoop.sizeMinus1;
        }
    }
}

static unsigned getLoopMask(const executeConvParams_t::Pattern pattern, const int loop)
{
    int loopIdx = (((uint8_t*) &pattern)[loop]);
    MME_ASSERT(loopIdx != 0xff, "invalid loop index");
    if (loop == LOOP_SPATIAL)
    {
        return (1 << Mme::c_mme_max_conv_dims);
    }
    else if (loopIdx < Mme::c_mme_max_conv_dims)
    {
        return (1 << loopIdx);
    }
    else
    {
        return (2 << Mme::c_mme_max_conv_dims);
    }
}

static void setLoopDim(Mme::Desc* desc, executeConvParams_t::Pattern pattern, int loop, EOperand operand, unsigned dim)
{
    int loopIdx = std::min(loop, (int) LOOP_FILTER);
    Mme::MmeAssociatedDims* assocDim;
    unsigned descLoopIdx = (((uint8_t*) &pattern)[loopIdx]) + (loop - loopIdx);
    if (descLoopIdx < Mme::c_mme_max_conv_dims)
    {
        assocDim = &desc->conv.associatedDims[descLoopIdx];
    }
    else
    {
        assocDim = &desc->outerLoop.associatedDims;
    }

    switch (operand)
    {
        case EOperand::OP_S:
            assocDim->dimS = dim;
            break;
        case EOperand::OP_L:
            assocDim->dimL = dim;
            break;
        case EOperand::OP_O:
            assocDim->dimO = dim;
            break;
        default:
            MME_ASSERT(0, "invalid operand");
    }
}

typedef struct
{
    Mme::Desc desc[Mme::MME_MASTERS_NR];
} descGroup_t;

typedef std::vector<descGroup_t> descList_t;

typedef struct
{
    int matrixWidth;
    int matrixHeight;
    int aguAnr;
    int aguBnr;
    bool transO;
    int subMatrixHeight;
    int subMatrixWidth;
} geoParams_t;

static inline void geo2Params(const EMmeGeometry geo, const EMmeDataType inDataType, geoParams_t* params)
{
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
    }

    unsigned shiftAmnount = getDataTypeShiftAmount(inDataType);
    params->subMatrixHeight = Mme::c_mme_matrix_size >> shiftAmnount;
    params->subMatrixWidth = Mme::c_mme_matrix_size >> shiftAmnount;

    params->matrixHeight = params->aguAnr * params->subMatrixHeight;
    params->matrixWidth = params->aguBnr * params->subMatrixWidth;
}

void getOutputSizes(const MmeSimTensor* outTensor, int* sizes, unsigned* spSize)
{
    unsigned spSizeBck;
    int sizesBck[Mme::c_mme_max_tensor_dims];

    if (!spSize)
    {
        spSize = &spSizeBck;
    }

    if (!sizes)
    {
        sizes = sizesBck;
    }

    outTensor->copySizes(sizes);

    sizes[1] = std::max(sizes[1], 2);
    *spSize = sizes[1];
    for (int i = 2; i < outTensor->getDim(); i++)
    {
        *spSize *= sizes[i];
    }
    if (*spSize < 4)
    {
        sizes[1] = 4;
        *spSize = 4;
    }
}

#define MME_CMD_CLONE(_class_name_)                                                                                    \
    _class_name_* ret = new _class_name_(*this);                                                                       \
    if (ret->m_next)                                                                                                   \
    {                                                                                                                  \
        ret->m_next = ret->m_next->clone();                                                                            \
    }                                                                                                                  \
    return ret;

class MmeStackCmd
{
public:
    MmeStackCmd() : m_next(0) {}

    virtual ~MmeStackCmd()
    {
        if (m_next) delete m_next;
    }

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd); }

    virtual void execute(executeConvParams_t* params, descList_t* descList)
    {
        MME_ASSERT_PTR(m_next);
        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        m_next->execute(params, descList);
    }

    MmeStackCmd* setNext(MmeStackCmd* next)
    {
        if (next)
        {
            next->getLast()->m_next = m_next;
        }
        m_next = next;
        return next;
    }

    MmeStackCmd* getNext() { return m_next; }

    MmeStackCmd* setLast(MmeStackCmd* cmd)
    {
        getLast()->setNext(cmd);
        return cmd;
    }

    MmeStackCmd* getLast()
    {
        MmeStackCmd* last;
        for (last = this; last->m_next; last = last->m_next)
            ;
        return last;
    }

protected:
    MmeStackCmd* m_next;
};

class MmeStackCmd_BuildDesc : public MmeStackCmd
{
public:
    MmeStackCmd_BuildDesc() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_BuildDesc); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        Mme::Desc localDesc;

        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->a->getDim() == params->c->getDim(), "A and C dims should match");
        MME_ASSERT(params->conv->dim + 2 == params->b->getDim(), "conv dims should match B dims");
        MME_ASSERT(params->conv->dim + 1 <= params->a->getDim(), "conv dims should match A dims");
        MME_ASSERT(params->a->getStride(0) == 1, "tensor stride[0] should be 1");
        MME_ASSERT(params->b->getStride(0) == 1, "tensor stride[0] should be 1");
        MME_ASSERT(params->c->getStride(0) == 1, "tensor stride[0] should be 1");
        MME_ASSERT(params->a->getElementType() == params->b->getElementType(), "A and B data type should match");

        geoParams_t gp;
        geo2Params(params->selectedGeo, params->a->getElementType(), &gp);

        int sizesC[Mme::c_mme_max_tensor_dims];
        unsigned spElements = params->roi.spSize;
        getOutputSizes(params->c, sizesC, 0);

        uint32_t spPos[c_operand_max_agu][Mme::c_mme_max_tensor_dims - 1];
        memcpy(spPos[0], params->roi.spPos, sizeof(spPos[0]));
        for (int i = 1; i < gp.aguAnr; i++)
        {
            memcpy(spPos[i], spPos[i - 1], sizeof(spPos[0]));
            for (int dim = 0;; dim++)
            {
                MME_ASSERT(dim < Mme::c_mme_max_tensor_dims - 1, "dim should not exceed max tensor dim nr");
                spPos[i][dim]++;
                if ((dim < Mme::c_mme_max_tensor_dims - 2) && (spPos[i][dim] == sizesC[dim + 1]))
                {
                    spPos[i][dim] = 0;
                }
                else
                {
                    break;
                }
            }
        }

        localDesc.sw.dw = 0;
        localDesc.pcu.rlSaturation = 16 * 4096;
        localDesc.axiUserData.dw = 0;
        localDesc.perfEvtS.dw = 0;
        localDesc.perfEvtS.startEndMask = -1;
        localDesc.perfEvtL[Mme::e_mme_local].dw = 0;
        localDesc.perfEvtL[Mme::e_mme_local].startEndMask = -1;
        localDesc.perfEvtL[Mme::e_mme_remote].dw = 0;
        localDesc.perfEvtL[Mme::e_mme_remote].startEndMask = -1;
        localDesc.perfEvtO[Mme::e_mme_local].dw = 0;
        localDesc.perfEvtO[Mme::e_mme_local].startEndMask = -1;
        localDesc.perfEvtO[Mme::e_mme_remote].dw = 0;
        localDesc.perfEvtO[Mme::e_mme_remote].startEndMask = -1;
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
        localDesc.header.transL = (params->op == executeConvParams_t::OP::OP_DEDX);
        localDesc.header.lowerS = 0;
        localDesc.header.lowerL = 0;
        localDesc.header.transO = 0;
        localDesc.header.accumMask = (1 << params->conv->dim) - 1;
        localDesc.header.storeEn = 1;
        localDesc.header.advanceS = 1;
        localDesc.header.advanceL = 0;
        localDesc.header.advanceO = 1;
        localDesc.header.accStoreIncDisable = 0;
        localDesc.header.dataTypeIn = params->a->getElementType();
        localDesc.header.dataTypeOut = params->c->getElementType();
        localDesc.header.reluEn = params->conv->relu && (params->op == executeConvParams_t::OP::OP_FWD);
        localDesc.header.accum = 0;
        localDesc.header.rollAccums = 0;
        localDesc.header.roundingMode = (int) params->roundingMode | params->conversionRoundingMode;
        localDesc.header.signalEn = 1;
        localDesc.header.signalMask = localDesc.header.accumMask;

        localDesc.numIterationsMinus1 = ((spElements + gp.matrixHeight - 1) / gp.matrixHeight) - 1;

        for (unsigned i = 0; i < Mme::c_mme_max_conv_dims; i++)
        {
            localDesc.conv.associatedDims[i].dimS = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimL = Mme::c_mme_max_tensor_dims;
            localDesc.conv.associatedDims[i].dimO = Mme::c_mme_max_tensor_dims;
        }
        localDesc.outerLoop.associatedDims.dimS = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimL = Mme::c_mme_max_tensor_dims;
        localDesc.outerLoop.associatedDims.dimO = Mme::c_mme_max_tensor_dims;

        ptr64 ptr;
        ptr.ddw = (uint64_t) params->a->data();
        localDesc.baseAddrHighS = ptr.dw[1];
        localDesc.baseAddrLowS = ptr.dw[0];

        ptr.ddw = (uint64_t) params->b->data();
        localDesc.baseAddrHighL = ptr.dw[1];
        localDesc.baseAddrLowL = ptr.dw[0];

        ptr.ddw = (uint64_t) params->c->data();
        localDesc.baseAddrHighO = ptr.dw[1];
        localDesc.baseAddrLowO = ptr.dw[0];

        int spElementsPerAguA = (spElements + gp.aguAnr - 1) / gp.aguAnr;
        int spatialSizeAC =
            spElementsPerAguA % gp.subMatrixHeight ? spElementsPerAguA % gp.subMatrixHeight : gp.subMatrixHeight;
        int spatialSizeB = params->b->getSize(1);
        if (params->op == executeConvParams_t::OP::OP_DEDX)
        {
            spatialSizeB %= gp.subMatrixWidth;
            spatialSizeB = spatialSizeB ? spatialSizeB : gp.subMatrixWidth;
        }

        localDesc.tensorS.spatialSizeMinus1 = spatialSizeAC - 1;
        localDesc.tensorO.spatialSizeMinus1 = spatialSizeAC - 1;
        localDesc.tensorL.spatialSizeMinus1 = spatialSizeB - 1;

        localDesc.conv.kernelSizeMinus1.dw = 0;
        localDesc.outerLoop.sizeMinus1 = 0;

        if (params->a->getElementSize() == 2)
        {
            ((uint16_t*) &localDesc.paddingValueS)[0] = params->conv->paddingValue.bf16;
            ((uint16_t*) &localDesc.paddingValueS)[1] = params->conv->paddingValue.bf16;
        }
        else
        {
            localDesc.paddingValueS = params->conv->paddingValue.f32;
        }

        localDesc.paddingValueL = 0;

        // set default values
        Mme::MmeAguCoreDesc aguS[c_operand_max_agu] = {0};
        Mme::MmeAguCoreDesc aguL[c_operand_max_agu] = {0};
        Mme::MmeAguCoreDesc aguO[c_operand_max_agu] = {0};

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
        localDesc.tensorS.validElements[0] = params->a->getSize(0);
        localDesc.tensorS.roiSize[0] = params->a->getSize(0);
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[0] = 0;
        }

        // tensor C: (dim 0)
        localDesc.tensorO.validElements[0] = params->c->getSize(0);
        localDesc.tensorO.roiSize[0] = params->roi.denseSize % gp.subMatrixWidth;
        localDesc.tensorO.roiSize[0] = localDesc.tensorO.roiSize[0] ? localDesc.tensorO.roiSize[0] : gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[0] = ((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi.denseBase;
        }

        // tensor B: K (dim 0)
        localDesc.tensorL.validElements[0] = params->b->getSize(0);
        localDesc.tensorL.roiSize[0] =
            localDesc.header.transL ? params->b->getSize(0) : (params->roi.denseSize % gp.subMatrixWidth);
        localDesc.tensorL.roiSize[0] = localDesc.tensorL.roiSize[0] ? localDesc.tensorL.roiSize[0] : gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[0] =
                localDesc.header.transL ? 0 : (((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi.denseBase);
        }

        // tensor B: C (dim 1)
        localDesc.tensorL.validElements[1] = params->b->getSize(1) * params->b->getStride(1);
        localDesc.tensorL.roiSize[1] =
            params->b->getStride(1) * (localDesc.header.transL
                                           ? std::min(gp.subMatrixWidth, (int) params->roi.denseSize)
                                           : params->b->getSize(1));
        localDesc.tensorL.spatialStrides[0] = params->b->getStride(1);
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[1] =
                localDesc.header.transL
                    ? (((j % gp.aguBnr) * gp.subMatrixWidth) + params->roi.denseBase) * params->b->getStride(1)
                    : 0;
        }

        unsigned loopS[c_operand_max_agu];
        unsigned loopL[c_operand_max_agu];
        unsigned loopO[c_operand_max_agu];
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            loopS[j] = getLoopMask(params->SelectedPattern, LOOP_SPATIAL);
            unsigned rem = params->roi.denseSize % gp.matrixWidth;
            rem = rem ? rem : gp.matrixWidth;
            loopL[j] = ((rem + params->roi.denseBase - aguO[j].roiBaseOffset[0]) < gp.subMatrixWidth)
                           ? getLoopMask(params->SelectedPattern, LOOP_K)
                           : 0;
            loopO[j] = ((rem + params->roi.denseBase - aguO[j].roiBaseOffset[0]) < gp.subMatrixWidth)
                           ? getLoopMask(params->SelectedPattern, LOOP_K)
                           : 0;
        }

        for (int dim = 0; dim < params->conv->dim; dim++)
        {
            int ioDim = dim + 1;
            int wDim = dim + 2;

            // tensor A: spatial
            localDesc.tensorS.validElements[ioDim] = params->a->getSize(ioDim) * params->a->getStride(ioDim);
            localDesc.tensorS.roiSize[ioDim] =
                params->conv->convStride[dim] * sizesC[ioDim] * params->a->getStride(ioDim);
            localDesc.tensorS.loopStride[ioDim] = params->conv->dilation[dim] * params->a->getStride(ioDim);
            localDesc.tensorS.spatialStrides[ioDim - 1] =
                params->conv->convStride[dim] * params->a->getStride(ioDim) * (dim ? 1 : gp.aguAnr);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                if (params->op == executeConvParams_t::OP::OP_DEDX)
                {
                    aguS[j].roiBaseOffset[ioDim] = -(params->b->getSize(wDim) - 1);
                    aguS[j].roiBaseOffset[ioDim] *= params->conv->dilation[dim];
                    aguS[j].roiBaseOffset[ioDim] += params->conv->padding[dim];
                }
                else
                {
                    aguS[j].roiBaseOffset[ioDim] = -params->conv->padding[dim];
                }
                aguS[j].roiBaseOffset[ioDim] *= params->a->getStride(ioDim);
                aguS[j].startOffset[ioDim - 1] =
                    spPos[j % gp.aguAnr][dim] * params->conv->convStride[dim] * params->a->getStride(ioDim);
            }

            setLoopDim(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, EOperand::OP_S, ioDim);

            // tensor B: QRS
            localDesc.tensorL.validElements[wDim] = params->b->getSize(wDim) * params->b->getStride(wDim);
            if (wDim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorL.roiSize[wDim] = params->b->getStride(wDim);
            }
            localDesc.tensorL.loopStride[wDim] = (params->op == executeConvParams_t::OP::OP_DEDX)
                                                     ? -params->b->getStride(wDim)
                                                     : params->b->getStride(wDim);
            localDesc.tensorL.spatialStrides[wDim - 1] = params->b->getStride(wDim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguL[j].roiBaseOffset[wDim] = (params->op == executeConvParams_t::OP::OP_DEDX)
                                                  ? (params->b->getSize(wDim) - 1) * params->b->getStride(wDim)
                                                  : 0;
                aguL[j].startOffset[wDim - 1] = 0;
            }

            setLoopDim(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, EOperand::OP_L, wDim);
            setLoopSize(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, params->b->getSize(wDim) - 1);

            // tensor C: spatial loops
            localDesc.tensorO.validElements[ioDim] = params->c->getSize(ioDim) * params->c->getStride(ioDim);
            localDesc.tensorO.roiSize[ioDim] = sizesC[ioDim] * params->c->getStride(ioDim);
            localDesc.tensorO.loopStride[ioDim] = 0;
            localDesc.tensorO.spatialStrides[ioDim - 1] = params->c->getStride(ioDim) * (dim ? 1 : gp.aguAnr);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].roiBaseOffset[ioDim] = 0;
                aguO[j].startOffset[dim] = spPos[j / gp.aguBnr][dim] * params->c->getStride(ioDim);
            }

            setLoopDim(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, EOperand::OP_O, ioDim);
        }

        for (int dim = params->conv->dim + 1; dim < params->a->getDim(); dim++)
        {
            // tensor A: outer spatial loops
            localDesc.tensorS.validElements[dim] = params->a->getSize(dim) * params->a->getStride(dim);
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[dim] = params->a->getSize(dim) * params->a->getStride(dim);
            }
            localDesc.tensorS.spatialStrides[dim - 1] = params->a->getStride(dim);
            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguS[j].roiBaseOffset[dim] = 0;
                aguS[j].startOffset[dim - 1] = spPos[j % gp.aguAnr][dim - 1] * params->a->getStride(dim);
            }

            // tensor C: spatial loops
            localDesc.tensorO.validElements[dim] = params->c->getSize(dim) * params->c->getStride(dim);
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorO.roiSize[dim] = sizesC[dim] * params->c->getStride(dim);
            }
            localDesc.tensorO.spatialStrides[dim - 1] = params->c->getStride(dim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].roiBaseOffset[dim] = 0;
                aguO[j].startOffset[dim - 1] = spPos[j / gp.aguBnr][dim - 1] * params->c->getStride(dim);
            }
        }

        if (params->op == executeConvParams_t::OP::OP_DEDX)
        {
            localDesc.tensorL.loopStride[1] = gp.matrixWidth * params->b->getStride(1);
            setLoopDim(&localDesc, params->SelectedPattern, LOOP_K, EOperand::OP_L, 1);
        }
        else
        {
            localDesc.tensorL.loopStride[0] = gp.matrixWidth;
            setLoopDim(&localDesc, params->SelectedPattern, LOOP_K, EOperand::OP_L, 0);
        }

        localDesc.tensorS.loopStride[0] = 0;
        localDesc.tensorO.loopStride[0] = gp.matrixWidth;
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_K, EOperand::OP_O, 0);
        unsigned denseLoopSize = ((params->roi.denseSize + gp.matrixWidth - 1) / gp.matrixWidth) - 1;
        setLoopSize(&localDesc, params->SelectedPattern, LOOP_K, denseLoopSize);

        localDesc.syncObject.operation = 1;
        localDesc.syncObject.value = 1;

        if (gp.transO)
        {
            transposeDesc(&localDesc);
            std::swap(aguS, aguL);
            std::swap(gp.aguAnr, gp.aguBnr);
            std::swap(loopS, loopL);
        }

        descGroup_t descGroup;
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            descGroup.desc[descIdx] = localDesc;
            descGroup.desc[descIdx].aguS = aguS[descIdx];
            descGroup.desc[descIdx].syncObject.addrHigh = params->soAddrHigh;
            descGroup.desc[descIdx].header.partialHeightLoopS = loopS[descIdx];
            localDesc.header.partialHeightLoopS = getLoopMask(params->SelectedPattern, LOOP_SPATIAL);

            for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
            {
                descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].syncObject.addrLow[aguIdx] =
                    params->soAddrLow[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                if (aguIdx == Mme::e_mme_local)
                {
                    descGroup.desc[descIdx].header.partialHeightLoopLLocal =
                        loopL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                    descGroup.desc[descIdx].header.partialHeightLoopOLocal =
                        loopO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                }
                else
                {
                    descGroup.desc[descIdx].header.partialHeightLoopLRemote =
                        loopL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                    descGroup.desc[descIdx].header.partialHeightLoopORemote =
                        loopO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                }
            }
        }

        descList->push_back(descGroup);
    }
};

class MmeStackCmd_BuildDEDWDesc : public MmeStackCmd
{
public:
    MmeStackCmd_BuildDEDWDesc() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_BuildDEDWDesc); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->a->getDim() == params->b->getDim(), "A and B dims should match");
        MME_ASSERT(params->conv->dim + 2 == params->c->getDim(), "conv dims should match C dims");
        MME_ASSERT(params->conv->dim + 1 <= params->a->getDim(), "conv dims should match A dims");
        MME_ASSERT(params->a->getStride(0) == 1, "A stride[0] should be 1");
        MME_ASSERT(params->b->getStride(0) == 1, "B stride[0] should be 1");
        MME_ASSERT(params->c->getStride(0) == 1, "C stride[0] should be 1");
        MME_ASSERT(params->a->getElementType() == params->b->getElementType(), "A and B data types should match");

        geoParams_t gp;
        geo2Params(params->selectedGeo, params->a->getElementType(), &gp);

        unsigned int spElementsAB = 1;
        for (int dim = 1; dim < params->b->getDim(); dim++)
        {
            spElementsAB *= params->b->getSize(dim);
        }

        Mme::Desc localDesc;

        localDesc.sw.dw = 0;
        localDesc.pcu.rlSaturation = 16 * 4096;
        localDesc.axiUserData.dw = 0;
        localDesc.perfEvtS.dw = 0;
        localDesc.perfEvtS.startEndMask = -1;
        localDesc.perfEvtL[Mme::e_mme_local].dw = 0;
        localDesc.perfEvtL[Mme::e_mme_local].startEndMask = -1;
        localDesc.perfEvtL[Mme::e_mme_remote].dw = 0;
        localDesc.perfEvtL[Mme::e_mme_remote].startEndMask = -1;
        localDesc.perfEvtO[Mme::e_mme_local].dw = 0;
        localDesc.perfEvtO[Mme::e_mme_local].startEndMask = -1;
        localDesc.perfEvtO[Mme::e_mme_remote].dw = 0;
        localDesc.perfEvtO[Mme::e_mme_remote].startEndMask = -1;
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
        localDesc.header.signalEn = 1;
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
        localDesc.header.dataTypeIn = params->a->getElementType();
        localDesc.header.dataTypeOut = params->c->getElementType();
        localDesc.header.reluEn = 0;
        localDesc.header.accum = 0;
        localDesc.header.rollAccums = 0;
        localDesc.header.roundingMode = (int) params->roundingMode | params->conversionRoundingMode;
        localDesc.header.signalEn = 1;
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

        ptr64 ptr;
        ptr.ddw = (uint64_t) params->a->data();
        localDesc.baseAddrHighS = ptr.dw[1];
        localDesc.baseAddrLowS = ptr.dw[0];

        ptr.ddw = (uint64_t) params->b->data();
        localDesc.baseAddrHighL = ptr.dw[1];
        localDesc.baseAddrLowL = ptr.dw[0];

        ptr.ddw = (uint64_t) params->c->data();
        localDesc.baseAddrHighO = ptr.dw[1];
        localDesc.baseAddrLowO = ptr.dw[0];

        localDesc.tensorS.spatialSizeMinus1 = spElementsAB - 1;
        localDesc.tensorL.spatialSizeMinus1 = spElementsAB - 1;
        localDesc.tensorO.spatialSizeMinus1 = std::min(params->c->getSize(1), (uint64_t) gp.subMatrixHeight) - 1;

        localDesc.paddingValueL = 0;
        localDesc.paddingValueS = 0;

        // set default values
        Mme::MmeAguCoreDesc aguS[c_operand_max_agu] = {0};
        Mme::MmeAguCoreDesc aguL[c_operand_max_agu] = {0};
        Mme::MmeAguCoreDesc aguO[c_operand_max_agu] = {0};

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
        localDesc.tensorS.validElements[0] = params->a->getSize(0);
        localDesc.tensorS.roiSize[0] = std::min(params->c->getSize(1), (uint64_t) gp.subMatrixHeight);
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguS[j].roiBaseOffset[0] = (j % gp.aguAnr) * gp.subMatrixHeight;
        }

        // tensor B: (dim 0)
        localDesc.tensorL.validElements[0] = params->b->getSize(0);
        localDesc.tensorL.roiSize[0] = gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguL[j].roiBaseOffset[0] = (j % gp.aguBnr) * gp.subMatrixWidth;
        }

        // tensor C: (dim 0)
        localDesc.tensorO.validElements[0] = params->c->getSize(0);
        localDesc.tensorO.roiSize[0] = gp.subMatrixWidth;
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[0] = (j % gp.aguBnr) * gp.subMatrixWidth;
        }

        // tensor C: (dim 1)
        localDesc.tensorO.validElements[1] = params->c->getSize(1) * params->c->getStride(1);
        localDesc.tensorO.roiSize[1] =
            std::min(params->c->getSize(1), (uint64_t) gp.subMatrixHeight) * params->c->getStride(1);
        localDesc.tensorO.spatialStrides[0] = params->c->getStride(1);
        for (int j = 0; j < c_operand_max_agu; j++)
        {
            aguO[j].roiBaseOffset[1] = (j / gp.aguBnr) * gp.subMatrixHeight * params->c->getStride(1);
        }

        for (int dim = 0; dim < params->conv->dim; dim++)
        {
            int ioDim = dim + 1;
            int wDim = dim + 2;

            // tensor A: spatial
            localDesc.tensorS.validElements[ioDim] = params->a->getSize(ioDim) * params->a->getStride(ioDim);
            localDesc.tensorS.roiSize[ioDim] =
                params->conv->convStride[dim] * params->b->getSize(ioDim) * params->a->getStride(ioDim);
            localDesc.tensorS.loopStride[ioDim] = params->conv->dilation[dim] * params->a->getStride(ioDim);
            localDesc.tensorS.spatialStrides[ioDim - 1] = params->conv->convStride[dim] * params->a->getStride(ioDim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguS[j].roiBaseOffset[ioDim] = -params->conv->padding[dim] * params->a->getStride(ioDim);
                aguS[j].startOffset[ioDim - 1] = 0;
            }

            setLoopDim(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, EOperand::OP_S, ioDim);

            // tensor B: spatial
            localDesc.tensorL.validElements[ioDim] = params->b->getSize(ioDim) * params->b->getStride(ioDim);
            localDesc.tensorL.roiSize[ioDim] = params->b->getSize(ioDim) * params->b->getStride(ioDim);
            localDesc.tensorL.loopStride[ioDim] = 0;
            localDesc.tensorL.spatialStrides[ioDim - 1] = params->b->getStride(ioDim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguL[j].roiBaseOffset[ioDim] = 0;
                aguL[j].startOffset[ioDim - 1] = 0;
            }

            setLoopDim(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, EOperand::OP_L, ioDim);

            // tensor C: kernel loops
            localDesc.tensorO.validElements[wDim] = params->c->getSize(wDim) * params->c->getStride(wDim);
            if (wDim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorO.roiSize[wDim] = params->c->getStride(wDim);
            }
            localDesc.tensorO.loopStride[wDim] = params->c->getStride(wDim);
            localDesc.tensorO.spatialStrides[wDim - 1] = params->c->getStride(wDim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguO[j].roiBaseOffset[wDim] = 0;
                aguO[j].startOffset[wDim - 1] = 0;
            }

            setLoopDim(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, EOperand::OP_O, wDim);
            setLoopSize(&localDesc, params->SelectedPattern, LOOP_FILTER + dim, params->c->getSize(wDim) - 1);
        }

        for (int dim = params->conv->dim + 1; dim < params->a->getDim(); dim++)
        {
            // tensor A: outer spatial loops
            localDesc.tensorS.validElements[dim] = params->a->getSize(dim) * params->a->getStride(dim);
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorS.roiSize[dim] = params->a->getSize(dim) * params->a->getStride(dim);
            }
            localDesc.tensorS.spatialStrides[dim - 1] = params->a->getStride(dim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguS[j].roiBaseOffset[dim] = 0;
                aguS[j].startOffset[dim - 1] = 0;
            }

            // tensor B: outer spatial loops
            localDesc.tensorL.validElements[dim] = params->b->getSize(dim) * params->b->getStride(dim);
            if (dim != Mme::c_mme_max_tensor_dims - 1)
            {
                localDesc.tensorL.roiSize[dim] = params->b->getSize(dim) * params->b->getStride(dim);
            }
            localDesc.tensorL.spatialStrides[dim - 1] = params->b->getStride(dim);

            for (int j = 0; j < c_operand_max_agu; j++)
            {
                aguL[j].roiBaseOffset[dim] = 0;
                aguL[j].startOffset[dim - 1] = 0;
            }
        }

        localDesc.tensorS.loopStride[0] = gp.matrixHeight;
        localDesc.tensorL.loopStride[0] = gp.matrixWidth;
        localDesc.tensorO.loopStride[0] = gp.matrixWidth;
        localDesc.tensorO.loopStride[1] = gp.matrixHeight * params->c->getStride(1);

        unsigned loopSizeK = ((params->c->getSize(0) + gp.matrixWidth - 1) / gp.matrixWidth) - 1;
        setLoopSize(&localDesc, params->SelectedPattern, LOOP_K, loopSizeK);
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_K, EOperand::OP_S, Mme::c_mme_max_tensor_dims);
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_K, EOperand::OP_L, 0);
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_K, EOperand::OP_O, 0);

        unsigned loopSizeC = ((params->c->getSize(1) + gp.matrixHeight - 1) / gp.matrixHeight) - 1;
        setLoopSize(&localDesc, params->SelectedPattern, LOOP_C, loopSizeC);
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_C, EOperand::OP_S, 0);
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_C, EOperand::OP_L, Mme::c_mme_max_tensor_dims);
        setLoopDim(&localDesc, params->SelectedPattern, LOOP_C, EOperand::OP_O, 1);

        localDesc.syncObject.operation = 1;
        localDesc.syncObject.value = 1;

        unsigned loopS = getLoopMask(params->SelectedPattern, LOOP_C);
        unsigned loopL = getLoopMask(params->SelectedPattern, LOOP_K);

        if (gp.transO)
        {
            transposeDesc(&localDesc);
            std::swap(aguS, aguL);
            std::swap(gp.aguAnr, gp.aguBnr);
            std::swap(loopS, loopL);
        }

        descGroup_t descGroup;
        for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
        {
            descGroup.desc[descIdx] = localDesc;
            descGroup.desc[descIdx].aguS = aguS[descIdx];
            descGroup.desc[descIdx].syncObject.addrHigh = params->soAddrHigh;
            descGroup.desc[descIdx].header.partialHeightLoopS = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopS;
            descGroup.desc[descIdx].header.partialHeightLoopLLocal = loopL;
            descGroup.desc[descIdx].header.partialHeightLoopLRemote = loopL;
            for (int aguIdx = 0; aguIdx < Mme::e_mme_local_and_remote; aguIdx++)
            {
                descGroup.desc[descIdx].aguL[aguIdx] = aguL[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].aguO[aguIdx] = aguO[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
                descGroup.desc[descIdx].syncObject.addrLow[aguIdx] =
                    params->soAddrLow[(descIdx * Mme::e_mme_local_and_remote) + aguIdx];
            }
        }

        descList->push_back(descGroup);
    }
};

class MmeStackCmd_PadConvAndTensors : public MmeStackCmd
{
public:
    MmeStackCmd_PadConvAndTensors() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_PadConvAndTensors); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        MME_ASSERT_PTR(params);
        MME_ASSERT_PTR(descList);
        MME_ASSERT(params->conv->dim + 1 <= params->a->getDim(), "conv dim should match A dim");
        if (params->op == executeConvParams_t::OP_DEDW)
        {
            MME_ASSERT(params->a->getDim() == params->b->getDim(), "A and B dims should match");
            MME_ASSERT(params->conv->dim + 2 == params->c->getDim(), "conv dim should match C dim");
        }
        else
        {
            MME_ASSERT(params->a->getDim() == params->c->getDim(), "A and C dims should match");
            MME_ASSERT(params->conv->dim + 2 == params->b->getDim(), "conv dim should match B dim");
        }

        int a_sizes[MmeSimTensor::c_tensorMaxDim];
        int a_strides[MmeSimTensor::c_tensorMaxDim];
        int b_sizes[MmeSimTensor::c_tensorMaxDim];
        int b_strides[MmeSimTensor::c_tensorMaxDim];
        int c_sizes[MmeSimTensor::c_tensorMaxDim];
        int c_strides[MmeSimTensor::c_tensorMaxDim];

        params->a->copySizes(a_sizes);
        params->a->copyStrides(a_strides);
        params->b->copySizes(b_sizes);
        params->b->copyStrides(b_strides);
        params->c->copySizes(c_sizes);
        params->c->copyStrides(c_strides);

        int maxStride;

        maxStride = 0;
        for (int i = 0; i < params->a->getDim(); i++)
        {
            maxStride = std::max(maxStride, a_sizes[i] * a_strides[i]);
        }
        for (int i = params->a->getDim(); i < Mme::c_mme_max_tensor_dims; i++)
        {
            a_sizes[i] = 1;
            a_strides[i] = maxStride;
        }

        maxStride = 0;
        for (int i = 0; i < params->b->getDim(); i++)
        {
            maxStride = std::max(maxStride, b_sizes[i] * b_strides[i]);
        }
        for (int i = params->b->getDim(); i < Mme::c_mme_max_tensor_dims; i++)
        {
            b_sizes[i] = 1;
            b_strides[i] = maxStride;
        }

        maxStride = 0;
        for (int i = 0; i < params->c->getDim(); i++)
        {
            maxStride = std::max(maxStride, c_sizes[i] * c_strides[i]);
        }
        for (int i = params->c->getDim(); i < Mme::c_mme_max_tensor_dims; i++)
        {
            c_sizes[i] = 1;
            c_strides[i] = maxStride;
        }

        ConvolutionParams newConv = *params->conv;
        newConv.dim = Mme::c_mme_max_conv_dims - 1;
        for (int i = params->conv->dim; i < newConv.dim; i++)
        {
            newConv.padding[i] = 0;
            newConv.dilation[i] = 1;
            newConv.convStride[i] = 1;
        }

        MmeSimTensor new_a(a_sizes,
                           Mme::c_mme_max_tensor_dims,
                           params->a->getElementType(),
                           params->a->data(),
                           a_strides);
        MmeSimTensor new_b(b_sizes,
                           Mme::c_mme_max_tensor_dims,
                           params->b->getElementType(),
                           params->b->data(),
                           b_strides);
        MmeSimTensor new_c(c_sizes,
                           Mme::c_mme_max_tensor_dims,
                           params->c->getElementType(),
                           params->c->data(),
                           c_strides);

        executeConvParams_t new_params = *params;
        new_params.a = &new_a;
        new_params.b = &new_b;
        new_params.c = &new_c;
        new_params.conv = &newConv;

        MME_ASSERT_PTR(m_next);
        m_next->execute(&new_params, descList);
    }
};

class MmeStackCmd_LowerA : public MmeStackCmd
{
public:
    MmeStackCmd_LowerA() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_LowerA); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        int a_sizes[MmeSimTensor::c_tensorMaxDim];
        int a_strides[MmeSimTensor::c_tensorMaxDim];
        int w_sizes[MmeSimTensor::c_tensorMaxDim];
        int w_strides[MmeSimTensor::c_tensorMaxDim];

        const MmeSimTensor* w = (params->op == executeConvParams_t::OP::OP_DEDW) ? params->c : params->b;

        params->a->copySizes(a_sizes);
        params->a->copyStrides(a_strides);
        w->copySizes(w_sizes);
        w->copyStrides(w_strides);

        bool lowerA = (params->conv->dilation[0] == 1) && (a_sizes[0] == a_strides[1]) &&
                      (w_strides[2] == w_sizes[0] * w_sizes[1]);

        if (lowerA)
        {
            a_sizes[0] *= w_sizes[2];
            w_sizes[1] *= w_sizes[2];
            w_strides[2] = w_sizes[1];
            w_sizes[2] = 1;
        }

        MmeSimTensor new_a(a_sizes, params->a->getDim(), params->a->getElementType(), params->a->data(), a_strides);
        MmeSimTensor new_w(w_sizes, w->getDim(), w->getElementType(), w->data(), w_strides);

        executeConvParams_t new_params = *params;
        new_params.a = &new_a;
        if (params->op == executeConvParams_t::OP::OP_DEDW)
        {
            new_params.c = &new_w;
        }
        else
        {
            new_params.b = &new_w;
        }

        // if (lowerA)
        //{
        //    printf("LowerA: Yes\n");
        //}
        // else
        //{
        //    printf("LowerA: No\n");
        //}

        unsigned groupIdx = (unsigned) descList->size();

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
    }
};

class MmeStackCmd_LowerM : public MmeStackCmd
{
public:
    MmeStackCmd_LowerM() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_LowerM); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        int a_sizes[MmeSimTensor::c_tensorMaxDim];
        int a_strides[MmeSimTensor::c_tensorMaxDim];

        params->a->copySizes(a_sizes);
        params->a->copyStrides(a_strides);

        bool lowerA = (getLoopMask(params->SelectedPattern, LOOP_FILTER0) == 0x1) && (params->conv->dilation[0] == 1) &&
                      (a_sizes[0] == a_strides[1]) && (params->conv->dim > 0);

        if (lowerA)
        {
            const MmeSimTensor* w = (params->op == executeConvParams_t::OP::OP_DEDW) ? params->c : params->b;
            a_sizes[0] *= w->getSize(2);
        }

        MmeSimTensor new_a(a_sizes, params->a->getDim(), params->a->getElementType(), params->a->data(), a_strides);

        executeConvParams_t new_params = *params;
        new_params.a = &new_a;

        // if (lowerA)
        //{
        //    printf("Lower M: Yes\n");
        //}
        // else
        //{
        //    printf("Lower M: No\n");
        //}

        unsigned groupIdx = (unsigned) descList->size();

        MME_ASSERT_PTR(m_next);
        m_next->execute(&new_params, descList);

        for (; groupIdx < descList->size(); groupIdx++)
        {
            for (int descIdx = 0; descIdx < Mme::MME_MASTERS_NR; descIdx++)
            {
                if ((*descList)[groupIdx].desc[descIdx].header.transO)
                {
                    (*descList)[groupIdx].desc[descIdx].header.lowerL = lowerA ? 1 : 0;
                    (*descList)[groupIdx].desc[descIdx].sbRepeat.aguLLoopMask |=
                        lowerA ? getLoopMask(params->SelectedPattern, LOOP_FILTER0) : 0;
                }
                else
                {
                    (*descList)[groupIdx].desc[descIdx].header.lowerS = lowerA ? 1 : 0;
                    (*descList)[groupIdx].desc[descIdx].sbRepeat.aguSLoopMask |=
                        lowerA ? getLoopMask(params->SelectedPattern, LOOP_FILTER0) : 0;
                }
            }
        }
    }
};

class MmeStackCmd_Partial : public MmeStackCmd
{
public:
    static const unsigned PARTIAL_ALL = INT_MAX;

    MmeStackCmd_Partial(LOOP loop, unsigned chunkSize = PARTIAL_ALL) : MmeStackCmd(), m_loop(loop), m_chunk(chunkSize)
    {
    }

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_Partial); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        unsigned firstIdx = (unsigned) descList->size();

        switch (m_loop)
        {
            case LOOP_FILTER0:
            case LOOP_FILTER1:
            case LOOP_FILTER2:
                MME_ASSERT(params->op != executeConvParams_t::OP_DEDW,
                          "operation cannot be dedw when partials are at filter loops");
                executePartialFilter(params, descList);
                break;
            case LOOP_C:
                MME_ASSERT(params->op != executeConvParams_t::OP_DEDW,
                          "operation cannot be dedw when partials are at loop c");
                executePartialChannel(params, descList);
                break;
            default:
                MME_ASSERT(0, "invalid loop");
        }

        closePartialExecute(firstIdx, params, descList);
    }

private:
    void executePartialChannel(executeConvParams_t* params, descList_t* descList)
    {
        executeConvParams_t new_params = *params;

        int a_sizes[Mme::c_mme_max_tensor_dims];
        int a_strides[Mme::c_mme_max_tensor_dims];
        int a_offsets[Mme::c_mme_max_tensor_dims] = {0};
        int b_sizes[Mme::c_mme_max_tensor_dims];
        int b_strides[Mme::c_mme_max_tensor_dims];
        int b_offsets[Mme::c_mme_max_tensor_dims] = {0};

        params->a->copySizes(a_sizes);
        params->a->copyStrides(a_strides);
        params->b->copySizes(b_sizes);
        params->b->copyStrides(b_strides);

        for (; a_offsets[0] < params->a->getSize(0); a_offsets[0] += m_chunk)
        {
            a_sizes[0] = std::min((uint64_t) m_chunk, params->a->getSize(0) - a_offsets[0]);
            switch (params->op)
            {
                case executeConvParams_t::OP_FWD:
                    b_sizes[1] = a_sizes[0];
                    b_offsets[1] = a_offsets[0];
                    break;
                case executeConvParams_t::OP_DEDX:
                    b_sizes[0] = a_sizes[0];
                    b_offsets[0] = a_offsets[0];
                    break;
                case executeConvParams_t::OP_DEDW:
                default:
                    MME_ASSERT(0, "invalid operation");
            }

            MmeSimTensor new_a(a_sizes,
                               params->a->getDim(),
                               params->a->getElementType(),
                               params->a->getElementAt(a_offsets),
                               a_strides);
            MmeSimTensor new_b(b_sizes,
                               params->b->getDim(),
                               params->b->getElementType(),
                               params->b->getElementAt(b_offsets),
                               b_strides);

            new_params.a = &new_a;
            new_params.b = &new_b;

            MME_ASSERT_PTR(m_next);
            m_next->execute(&new_params, descList);
        }
    }

    void executePartialFilter(executeConvParams_t* params, descList_t* descList)
    {
        MME_ASSERT(params->op != executeConvParams_t::OP_DEDW, "cannot do partials on filter with operation=dedw");

        int b_sizes[Mme::c_mme_max_tensor_dims];
        int b_strides[Mme::c_mme_max_tensor_dims];
        int b_offsets[Mme::c_mme_max_tensor_dims] = {0};

        params->b->copySizes(b_sizes);
        params->b->copyStrides(b_strides);

        unsigned dim = m_loop - LOOP_FILTER0;
        unsigned wDim = dim + 2;

        executeConvParams_t new_params = *params;
        ConvolutionParams new_conv = *params->conv;

        for (; b_offsets[wDim] < params->b->getSize(wDim); b_offsets[wDim] += m_chunk)
        {
            b_sizes[wDim] = std::min((uint64_t) m_chunk, params->b->getSize(wDim) - b_offsets[wDim]);
            MmeSimTensor new_b(b_sizes,
                               params->b->getDim(),
                               params->b->getElementType(),
                               params->b->getElementAt(b_offsets),
                               b_strides);

            new_params.b = &new_b;
            new_params.conv = &new_conv;

            MME_ASSERT_PTR(m_next);
            m_next->execute(&new_params, descList);

            new_conv.padding[dim] -= new_conv.dilation[dim] * m_chunk;
        }
    }

    static void closePartialExecute(unsigned firstIdx, executeConvParams_t* params, descList_t* descList)
    {
        unsigned accumsNr;
        Mme::Desc* desc0 = &(*descList)[firstIdx].desc[0];
        unsigned loopKSize = getLoopSize(desc0, params->SelectedPattern, LOOP_K) + 1;

        if (params->op == executeConvParams_t::OP_DEDW)
        {
            unsigned loopCSize = getLoopSize(desc0, params->SelectedPattern, LOOP_C) + 1;
            accumsNr = loopKSize * loopCSize;
        }
        else
        {
            unsigned tetrisNr = desc0->numIterationsMinus1 + 1;
            accumsNr = loopKSize * tetrisNr;
        }

        unsigned descsNr = (unsigned) descList->size() - firstIdx;
        MME_ASSERT((descsNr == 1) || (accumsNr <= Mme::c_mme_accums_nr), "");

        bool transOut = descList->front().desc[0].header.transO;

        for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
        {
            Mme::Desc* desc;

            // steady state
            for (unsigned descIdx = firstIdx; descIdx < (unsigned) descList->size(); descIdx++)
            {
                desc = &(*descList)[descIdx].desc[masterIdx];
                desc->header.accum = (accumsNr == 1) ? 0 : 1;
                desc->header.storeEn = 0;
                desc->header.signalEn = params->strategy->signalPartial ? 1 : 0;
                desc->header.accStoreIncDisable = (accumsNr == 1) ? 1 : 0;
                desc->header.rollAccums = (accumsNr == 1) ? 0 : Mme::c_mme_accums_nr - accumsNr;
                desc->header.reluEn = false;
                MME_ASSERT(desc->header.transO == transOut, "");
            }

            // first desc
            desc = &(*descList)[firstIdx].desc[masterIdx];
            desc->header.accum = 0;

            // last desc
            desc = &descList->back().desc[masterIdx];
            desc->header.signalEn = 1;
            desc->header.storeEn = 1;
            desc->header.accStoreIncDisable = 0;
            desc->header.rollAccums = 0;
            desc->header.reluEn = params->conv->relu && (params->op == executeConvParams_t::OP::OP_FWD);
        }
    }

    LOOP m_loop;
    unsigned m_chunk;
};

class MmeStackCmd_ConvSpatialRoi : public MmeStackCmd
{
public:
    MmeStackCmd_ConvSpatialRoi(unsigned activations) : MmeStackCmd(), m_activations(activations) {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_ConvSpatialRoi); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        geoParams_t geoParams;
        geo2Params(params->selectedGeo, params->a->getElementType(), &geoParams);

        executeConvParams_t new_params = *params;

        int outSizes[Mme::c_mme_max_tensor_dims];
        unsigned outSpSize;
        getOutputSizes(params->c, outSizes, &outSpSize);

        unsigned factor = 1;
        unsigned spStart = 0;
        for (unsigned i = 1; i < params->c->getDim(); i++)
        {
            spStart += factor * params->roi.spPos[i - 1];
            factor *= outSizes[i];
        }

        for (unsigned base = 0; base < params->roi.spSize; base += (geoParams.matrixHeight * m_activations))
        {
            unsigned offset = spStart + base;
            unsigned rem = offset;
            for (unsigned i = 1; i < params->c->getDim(); i++)
            {
                new_params.roi.spPos[i - 1] = rem % outSizes[i];
                rem /= outSizes[i];
            }

            new_params.roi.spSize = geoParams.matrixHeight * m_activations;
            new_params.roi.spSize = std::min(new_params.roi.spSize, params->roi.spSize - base);
            new_params.roi.spSize = std::min(new_params.roi.spSize, outSpSize - offset);

            m_next->execute(&new_params, descList);
        }
    }

private:
    unsigned m_activations;
};

class MmeStackCmd_ConvDenseRoi : public MmeStackCmd
{
public:
    MmeStackCmd_ConvDenseRoi(unsigned activations) : MmeStackCmd(), m_activations(activations) {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_ConvDenseRoi); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        geoParams_t geoParams;
        geo2Params(params->selectedGeo, params->a->getElementType(), &geoParams);

        executeConvParams_t new_params = *params;

        for (unsigned base = 0; base < params->roi.denseSize; base += (geoParams.matrixWidth * m_activations))
        {
            unsigned offset = base + params->roi.denseBase;
            new_params.roi.denseBase = offset;
            new_params.roi.denseSize = geoParams.matrixWidth * m_activations;
            new_params.roi.denseSize = std::min(new_params.roi.denseSize, params->roi.denseSize - base);
            new_params.roi.denseSize = std::min((uint64_t) new_params.roi.denseSize, params->c->getSize(0) - offset);

            m_next->execute(&new_params, descList);
        }
    }

private:
    unsigned m_activations;
};

class MmeStackCmd_ReuseConv : public MmeStackCmd
{
public:
    MmeStackCmd_ReuseConv() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_ReuseConv); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        static const unsigned c_max_activations = 4;

        // call the lower layers in the command stack to generate a descriptor
        unsigned firstIdx = (unsigned) descList->size();
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);
        MME_ASSERT(descList->size() - firstIdx == 1, "");

        // extract info from the descriptor to determaine if reuse could take place
        unsigned mask;
        unsigned reuse;
        unsigned stripes;
        bool sharedOperandResue;
        if (params->SelectedPattern == executeConvParams_t::PATTERN_C_REDUCTION_K_S_F)
        {
            mask = getLoopMask(params->SelectedPattern, LOOP_SPATIAL);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->SelectedPattern, LOOP_SPATIAL);
            stripes = getLoopSize(&(*descList)[firstIdx].desc[0], params->SelectedPattern, LOOP_K);
            sharedOperandResue = false;
        }
        else if (params->SelectedPattern == executeConvParams_t::PATTERN_C_REDUCTION_S_K_F)
        {
            mask = getLoopMask(params->SelectedPattern, LOOP_K);
            reuse = getLoopSize(&(*descList)[firstIdx].desc[0], params->SelectedPattern, LOOP_K);
            stripes = getLoopSize(&(*descList)[firstIdx].desc[0], params->SelectedPattern, LOOP_SPATIAL);
            sharedOperandResue = true;
        }
        else
        {
            MME_ASSERT(0, "invalid pattern");
        }

        // check if reuse could take place.
        if (reuse)
        {
            // count the SB entries in the reused operand
            unsigned readsNr = AguSim::countSBReads(sharedOperandResue,
                                                    &(*descList)[firstIdx].desc[Mme::MME_CORE_NORTH_MASTER],
                                                    &(*descList)[firstIdx].desc[Mme::MME_CORE_SOUTH_MASTER]);

            // if partials are not needed - reuse the selected operand.
            if ((readsNr <= Mme::c_mme_sb_size) && (reuse < Mme::c_mme_max_sb_reuse))
            {
                if ((params->testParams->sbResuseInStripes) && (stripes == 0) && (!params->strategy->signalPartial))
                {
                    // Reuse over multiple activations - (mainly for testing)
                    descList->pop_back();
                    MmeStackCmd head;
                    if (sharedOperandResue)
                    {
                        head.setLast(new MmeStackCmd_ConvDenseRoi((reuse + c_max_activations) / c_max_activations));
                    }
                    else
                    {
                        head.setLast(new MmeStackCmd_ConvSpatialRoi((reuse + c_max_activations) / c_max_activations));
                    }
                    head.setLast(m_next->clone());
                    head.execute(params, descList);
                }

                MME_ASSERT(descList->size() - firstIdx <= c_max_activations, "");

                for (unsigned idx = firstIdx; idx < descList->size(); idx++)
                {
                    unsigned load = (idx == firstIdx) ? 1 : 0;

                    if (sharedOperandResue != (bool) (*descList)[firstIdx].desc[0].header.transO)
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
                // partials are needed (or the reuse is too large) - pop the descriptor.
                // It'll be replaced by a descriptor that can be reused.
                descList->pop_back();
                MmeStackCmd head;

                executeConvParams_t new_params = *params;
                MmeTestParams new_testParams = *params->testParams;
                new_params.testParams = &new_testParams;

                double divFactor = ((double) readsNr) / Mme::c_mme_sb_size;

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

                // break the descriptor into stripes with one activation of the shared
                // operand.
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
                // make sure there are enough accumulators to hold the reuse factor of
                // each stripe.
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
                    new_testParams.sbResuseInStripes = 0;

                    // if the common dim doesn't fit, try to break the filter,
                    if ((new_params.conv->dim > 2) && (new_params.b->getSize(4) > 1))
                    {
                        unsigned chunkSize = std::max((int) floor(new_params.b->getSize(4) / divFactor), 1);
                        head.setLast(new MmeStackCmd_Partial(LOOP_FILTER2, chunkSize));
                    }
                    else if ((new_params.conv->dim > 1) && (new_params.b->getSize(3) > 1))
                    {
                        unsigned chunkSize = std::max((int) floor(new_params.b->getSize(3) / divFactor), 1);
                        head.setLast(new MmeStackCmd_Partial(LOOP_FILTER1, chunkSize));
                    }
                    else if ((new_params.conv->dim > 0) && (new_params.b->getSize(2) > 1))
                    {
                        unsigned chunkSize = std::max((int) floor(new_params.b->getSize(2) / divFactor), 1);
                        head.setLast(new MmeStackCmd_Partial(LOOP_FILTER0, chunkSize));
                    }
                    // finaly, break the input into chunks that fit in the SB. (with
                    // several sizes)
                    else
                    {
                        unsigned inputSize = new_params.a->getSize(0);
                        MME_ASSERT(inputSize > Mme::c_mme_sb_size / 2, "");

                        unsigned chunkSize;
                        if (inputSize > Mme::c_mme_sb_size)
                        {
                            chunkSize = Mme::c_mme_sb_size;
                        }
                        // else if (inputSize == c_mme_sb_size)
                        //{
                        //    unsigned alignment = c_cl_size /
                        //    new_params.a->getElementSize(); chunkSize =
                        //    std::max(c_mme_sb_size / 2, (((unsigned)(c_mme_sb_size /
                        //    divFactor) / alignment) * alignment)); assert(chunkSize <
                        //    c_mme_sb_size);
                        //}
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
            }
        }
    }
};

class MmeStackCmd_StrategyConv : public MmeStackCmd
{
public:
    MmeStackCmd_StrategyConv() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_StrategyConv); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        executeConvParams_t new_params = *params;
        MME_ASSERT(params->op != executeConvParams_t::OP_DEDW, "operation cannot be dedw");
        MME_ASSERT(std::bitset<sizeof(params->strategy->pattern) * 8>(params->strategy->pattern).count() == 1,
                  "invalid pattern");

        switch (params->strategy->pattern)
        {
            case e_mme_z_reduction_ksf:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_C_REDUCTION_K_S_F;
                break;
            case e_mme_z_reduction_skf:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_C_REDUCTION_S_K_F;
                break;
            default:
                MME_ASSERT(0, "invalid pattern");
        }

        MME_ASSERT(std::bitset<32>(sizeof(params->strategy->geometry) * 8).count() == 1, "invalid geometry");

        switch (params->strategy->geometry)
        {
            case e_mme_geometry_4wx1h:
            case e_mme_geometry_2wx2h:
            case e_mme_geometry_1wx4h:
                new_params.selectedGeo = (EMmeGeometry) params->strategy->geometry;
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

        if (params->strategy->partial)
        {
            MME_ASSERT(0, "not supported yet");  // not supported yet
        }

        if (params->strategy->loweringEn)
        {
            head.setLast(new MmeStackCmd_LowerM());
        }

        head.setLast(new MmeStackCmd_BuildDesc());

        head.execute(&new_params, descList);
    }
};

class MmeStackCmd_StrategyDedw : public MmeStackCmd
{
public:
    MmeStackCmd_StrategyDedw() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_StrategyDedw); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        executeConvParams_t new_params = *params;
        MME_ASSERT(params->op == executeConvParams_t::OP_DEDW, "operation should be dedw");
        MME_ASSERT(std::bitset<sizeof(params->strategy->pattern) * 8>(params->strategy->pattern).count() == 1, "");

        switch (params->strategy->pattern)
        {
            case e_mme_sp_reduction_kfc:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_S_REDUCTION_K_F_C;
                break;
            case e_mme_sp_reduction_fkc:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_S_REDUCTION_F_K_C;
                break;
            case e_mme_sp_reduction_fck:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_S_REDUCTION_F_C_K;
                break;
            case e_mme_sp_reduction_cfk:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_S_REDUCTION_C_F_K;
                break;
            case e_mme_sp_reduction_kcf:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_S_REDUCTION_K_C_F;
                break;
            case e_mme_sp_reduction_ckf:
                new_params.SelectedPattern = executeConvParams_t::PATTERN_S_REDUCTION_C_K_F;
                break;
            default:
                MME_ASSERT(0, "invalid pattern");
        }

        MME_ASSERT(std::bitset<32>(sizeof(params->strategy->geometry) * 8).count() == 1, "");

        switch (params->strategy->geometry)
        {
            case e_mme_geometry_4wx1h:
            case e_mme_geometry_2wx2h:
            case e_mme_geometry_1wx4h:
                new_params.selectedGeo = (EMmeGeometry) params->strategy->geometry;
                // new_params.selectedGeo = e_mme_geometry_4wx1h;
                break;
            default:
                MME_ASSERT(0, "invalid geometry");
        }

        MmeStackCmd head;

        if (m_next)
        {
            head.setNext(m_next->clone());
        }

        if (params->strategy->sbReuse || params->strategy->partial)
        {
            // assert(0); // not supported yet ignored for now
            // head.setLast(new MmeStackCmd_Reuse());
            // head.setLast(new MmeStackCmd_Partial(LOOP_FILTER1, 1));
            // head.setLast(new MmeStackCmd_Partial(LOOP_FILTER0, 2));
            // head.setLast(new MmeStackCmd_Partial(LOOP_C, 256));
        }

        if (params->strategy->loweringEn)
        {
            head.setLast(new MmeStackCmd_LowerA());
        }

        head.setLast(new MmeStackCmd_BuildDEDWDesc());

        head.execute(&new_params, descList);
    }
};

static void descList2ExecPlan(const Mme::Desc* prevDesc,
                              descList_t* descList,
                              std::list<MmeRegWriteCmd>* cmds,
                              const unsigned repeats,
                              bool incDec,
                              bool maskSignal,
                              int* targetSoValue,
                              Mme::Desc* lastDesc)
{
    MME_ASSERT(repeats, "repeats should have a positive value");

    *targetSoValue = 0;
    for (int gIdx = 0; gIdx < descList->size(); gIdx++)
    {
        *targetSoValue += getTargetSOValue(&(*descList)[gIdx].desc[0]);
        MME_ASSERT(*targetSoValue <= 0x7fff, "invalid SO value");

        for (int mIdx = 0; mIdx < Mme::MME_MASTERS_NR; mIdx++)
        {
            Mme::Desc* curr = &(*descList)[gIdx].desc[mIdx];
            const Mme::Desc* prev;

            if (gIdx)
            {
                prev = &(*descList)[gIdx - 1].desc[mIdx];
            }
            else if (prevDesc)
            {
                prev = &prevDesc[mIdx];
            }
            else
            {
                prev = 0;
            }

            if (prev)
            {
                if (!curr->header.storeEn)
                {
                    int32_t roiSize0 = curr->tensorO.roiSize[0];
                    uint32_t spatialSize = curr->tensorO.spatialSizeMinus1;
                    memcpy(&curr->tensorO, &prev->tensorO, sizeof(curr->tensorO));
                    curr->tensorO.roiSize[0] = roiSize0;
                    curr->tensorO.spatialSizeMinus1 = spatialSize;
                    memcpy(&curr->aguO, &prev->aguO, sizeof(curr->aguO));
                    curr->rateLimiter.aguO = prev->rateLimiter.aguO;
                    curr->axiUserData.dw = prev->axiUserData.dw;
                    curr->baseAddrHighO = prev->baseAddrHighO;
                    curr->baseAddrLowO = prev->baseAddrLowO;
                }

                if (!curr->header.signalEn)
                {
                    curr->syncObject.ddw = prev->syncObject.ddw;
                }

                if ((!curr->header.storeEn) && (!curr->header.signalEn))
                {
                    memcpy(&curr->metaData.aguO, &prev->metaData.aguO, sizeof(curr->metaData.aguO));
                    memcpy(&curr->perfEvtO, &prev->perfEvtO, sizeof(curr->perfEvtO));
                }

                int diffIdx = -1;
                for (unsigned dw = 0; dw < DESC_SIZE_IN_DW; dw++)
                {
                    if ((curr->dw[dw] != prev->dw[dw]) && (diffIdx < 0))
                    {
                        diffIdx = dw;
                    }
                    else if ((curr->dw[dw] == prev->dw[dw]) && (diffIdx >= 0))
                    {
                        pushCmd(&cmds[mIdx], MME_REG_OFFSET(desc.dw[diffIdx]), dw - diffIdx, &curr->dw[diffIdx]);
                        diffIdx = -1;
                    }
                }

                if (diffIdx >= 0)
                {
                    pushCmd(&cmds[mIdx],
                            MME_REG_OFFSET(desc.dw[diffIdx]),
                            DESC_SIZE_IN_DW - diffIdx,
                            &curr->dw[diffIdx]);
                }
            }
            else
            {
                pushCmd(&cmds[mIdx], MME_REG_OFFSET(desc), DESC_SIZE_IN_DW, curr);
            }

            if (repeats > 1)
            {
                Mme::MmeHeader hdr = curr->header;
                MME_ASSERT(hdr.storeEn, "stroe shoudl be enabled");
                hdr.signalEn = 0;
                pushCmd(&cmds[mIdx], MME_REG_OFFSET(desc.header), sizeof(Mme::MmeHeader) / sizeof(uint32_t), &hdr);

                for (unsigned rep = 0; rep < repeats - 1; rep++)
                {
                    if (incDec)
                    {
                        Mme::MmeUserData ud;
                        ud.first = hdr.dataTypeOut ? 0x0f : 0x0d;
                        ud.first |= (rep & 0x1) ? 0x20 : 0x0;
                        ud.steady = ud.first;
                        pushCmd(&cmds[mIdx], MME_REG_OFFSET(desc.axiUserData), 1, &ud);
                    }

                    pushCmd(&cmds[mIdx], MME_REG_OFFSET(cmd), 1, &ONE);
                }

                if (!maskSignal)
                {
                    pushCmd(&cmds[mIdx],
                            MME_REG_OFFSET(desc.header),
                            sizeof(Mme::MmeHeader) / sizeof(uint32_t),
                            &curr->header);
                }

                if (incDec)
                {
                    Mme::MmeUserData ud;
                    ud.first = hdr.dataTypeOut ? 0x2f : 0x2d;
                    ud.steady = ud.first;
                    pushCmd(&cmds[mIdx], MME_REG_OFFSET(desc.axiUserData), 1, &ud);
                }
            }
            else if (maskSignal)
            {
                Mme::MmeHeader hdr = curr->header;
                MME_ASSERT(hdr.storeEn, "store should be enabled");
                hdr.signalEn = 0;
                pushCmd(&cmds[mIdx], MME_REG_OFFSET(desc.header), sizeof(Mme::MmeHeader) / sizeof(uint32_t), &hdr);
            }

            pushCmd(&cmds[mIdx], MME_REG_OFFSET(cmd), 1, &ONE);
        }
    }

    if (lastDesc && (descList->size() > 0))
    {
        memcpy(lastDesc, &descList->back().desc, Mme::MME_MASTERS_NR * sizeof(Mme::Desc));
    }
}

class MmeStackCmd_TraceEvent : public MmeStackCmd
{
public:
    MmeStackCmd_TraceEvent(uint16_t wkldId) : MmeStackCmd(), m_wkldId(wkldId) {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_TraceEvent); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);

        Mme::MmePerfEvt evt;
        evt.incMask = 1;
        evt.loopMask = -1;
        evt.rst = 1;
        evt.value = m_wkldId;

        for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
        {
            Mme::Desc* desc;
            evt.startEndMask = 2;  // mask end
            desc = &descList->front().desc[masterIdx];
            if (desc->header.transO)
            {
                desc->perfEvtL[Mme::e_mme_local].dw = evt.dw;
                desc->perfEvtL[Mme::e_mme_remote].dw = evt.dw;
            }
            else
            {
                desc->perfEvtS.dw = evt.dw;
            }

            evt.startEndMask = 1;  // mask start
            desc = &descList->back().desc[masterIdx];
            desc->perfEvtO[Mme::e_mme_remote].dw = evt.dw;
            desc->perfEvtO[Mme::e_mme_local].dw = evt.dw;
        }
    }

private:
    uint16_t m_wkldId;
};

class MmeStackCmd_RandomDbg : public MmeStackCmd
{
public:
    MmeStackCmd_RandomDbg() : MmeStackCmd() {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_RandomDbg); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        unsigned firstIdx = (unsigned) descList->size();
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);
        bool transOut = (*descList).back().desc[0].header.transO;

        for (unsigned descIdx = firstIdx; descIdx < (unsigned) descList->size(); descIdx++)
        {
            unsigned soVal = (rand() & 0xf) + 1;

            for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
            {
                Mme::Desc* desc = &(*descList)[descIdx].desc[masterIdx];
                desc->syncObject.value = soVal;
                desc->axiUserData.dw = rand32();
                desc->perfEvtS.dw = rand32();
                desc->perfEvtL[Mme::e_mme_local].dw = rand32();
                desc->perfEvtL[Mme::e_mme_remote].dw = rand32();
                desc->perfEvtO[Mme::e_mme_local].dw = rand32();
                desc->perfEvtO[Mme::e_mme_remote].dw = rand32();
                desc->metaData.aguS = rand32();
                desc->metaData.aguL[Mme::e_mme_local] = rand32();
                desc->metaData.aguL[Mme::e_mme_remote] = rand32();
                desc->metaData.aguO[Mme::e_mme_local] = rand32();
                desc->metaData.aguO[Mme::e_mme_remote] = rand32();
                desc->pcu.rlSaturation = rand32();
                desc->rateLimiter.aguS = rand();
                desc->rateLimiter.aguL = rand();
                desc->rateLimiter.aguO = rand();
            }
        }
    }

private:
    static inline uint32_t rand32()
    {
        uint8_t ret[4];
        ret[0] = rand();
        ret[1] = rand();
        ret[2] = rand();
        ret[3] = rand();
        return *(uint32_t*) ret;
    }
};

class MmeStackCmd_WkldIdMD : public MmeStackCmd
{
public:
    MmeStackCmd_WkldIdMD(uint16_t wkldId) : MmeStackCmd(), m_wkldId(wkldId) {}

    virtual MmeStackCmd* clone() { MME_CMD_CLONE(MmeStackCmd_WkldIdMD); }

    void execute(executeConvParams_t* params, descList_t* descList)
    {
        unsigned firstIdx = (unsigned) descList->size();
        MME_ASSERT_PTR(m_next);
        m_next->execute(params, descList);
        uint16_t wkldId = m_wkldId | 0x8000;

        for (unsigned descIdx = firstIdx; descIdx < (unsigned) descList->size(); descIdx++)
        {
            for (unsigned masterIdx = 0; masterIdx < Mme::MME_MASTERS_NR; masterIdx++)
            {
                Mme::Desc* desc = &(*descList)[descIdx].desc[masterIdx];
                ((uint16_t*) &desc->metaData.aguS)[0] = wkldId;
                ((uint16_t*) &desc->metaData.aguL[Mme::e_mme_local])[0] = wkldId;
                ((uint16_t*) &desc->metaData.aguL[Mme::e_mme_remote])[0] = wkldId;
                ((uint16_t*) &desc->metaData.aguO[Mme::e_mme_local])[0] = wkldId;
                ((uint16_t*) &desc->metaData.aguO[Mme::e_mme_remote])[0] = wkldId;
            }

            wkldId &= ~0x8000;
        }
    }

private:
    uint16_t m_wkldId;
};

static void executeConvFWD_verif(unsigned wkldId,
                                 const ConvolutionParams* conv,
                                 const uint32_t* soAddrLow,
                                 const uint32_t soAddrHigh,
                                 const MmeSimTensor* x,
                                 const MmeSimTensor* w,
                                 const MmeSimTensor* y,
                                 const RoundingMode roundingMode,
                                 const RoundingMode conversionRoundingMode,
                                 const MmeStrategy* strategy,
                                 const MmeTestParams* testParams,
                                 int* targetSoValue,
                                 std::list<MmeRegWriteCmd>* cmds,
                                 const Mme::Desc* prevDesc,
                                 Mme::Desc* lastDesc)
{
    MME_ASSERT_PTR(targetSoValue);
    MME_ASSERT_PTR(cmds);
    MME_ASSERT_PTR(x);
    MME_ASSERT_PTR(w);
    MME_ASSERT(x->getElementType() == w->getElementType(), "X and W data type should match");
    MME_ASSERT(x->getSize(0) == w->getSize(1), "common dim size should match");
    MME_ASSERT(y->getSize(0) == w->getSize(0), "output width should match W width");

    const MmeTestParams* localTestParams = testParams ? testParams : &c_default_test_params;

    MmeStackCmd head;
    if (localTestParams->wkldIdMD)
    {
        head.setLast(new MmeStackCmd_WkldIdMD(wkldId));
    }
    if (localTestParams->randomMD)
    {
        head.setLast(new MmeStackCmd_RandomDbg());
    }
    else
    {
        head.setNext(new MmeStackCmd_TraceEvent(wkldId));
    }

    head.setLast(new MmeStackCmd_StrategyConv());
    head.setLast(new MmeStackCmd_PadConvAndTensors());

    *targetSoValue = 0;

    executeConvParams_t params = {0};
    memcpy(params.soAddrLow, soAddrLow, sizeof(params.soAddrLow));
    params.soAddrHigh = soAddrHigh;
    params.conv = conv;
    params.a = x;
    params.b = w;
    params.c = y;
    params.op = executeConvParams_t::OP::OP_FWD;
    params.testParams = localTestParams;
    params.strategy = strategy;
    params.roundingMode = roundingMode;
    params.conversionRoundingMode = conversionRoundingMode;
    params.roi.denseSize = y->getSize(0);
    getOutputSizes(y, 0, &params.roi.spSize);

    descList_t descList;
    head.execute(&params, &descList);
    descList2ExecPlan(prevDesc,
                      &descList,
                      cmds,
                      localTestParams->repeats,
                      localTestParams->incDec,
                      localTestParams->maskSignals,
                      targetSoValue,
                      lastDesc);
}

static void executeConvDEDX_verif(unsigned wkldId,
                                  const ConvolutionParams* conv,
                                  const uint32_t* soAddrLow,
                                  const uint32_t soAddrHigh,
                                  const MmeSimTensor* x,
                                  const MmeSimTensor* w,
                                  const MmeSimTensor* y,
                                  const RoundingMode roundingMode,
                                  const RoundingMode conversionRoundingMode,
                                  const MmeStrategy* strategy,
                                  const MmeTestParams* testParams,
                                  int* targetSoValue,
                                  std::list<MmeRegWriteCmd>* cmds,
                                  const Mme::Desc* prevDesc,
                                  Mme::Desc* lastDesc)
{
    MME_ASSERT_PTR(targetSoValue);
    MME_ASSERT_PTR(cmds);
    MME_ASSERT_PTR(y);
    MME_ASSERT_PTR(w);
    MME_ASSERT(y->getElementType() == w->getElementType(), "Y and W data type should match");
    MME_ASSERT(x->getSize(0) == w->getSize(1), "output width should match Wt width");
    MME_ASSERT(y->getSize(0) == w->getSize(0), "common dim should match");

    descList_t descList;

    const MmeTestParams* localTestParams = testParams ? testParams : &c_default_test_params;

    MmeStackCmd head;
    if (localTestParams->wkldIdMD)
    {
        head.setLast(new MmeStackCmd_WkldIdMD(wkldId));
    }

    if (localTestParams->randomMD)
    {
        head.setLast(new MmeStackCmd_RandomDbg());
    }
    else
    {
        head.setNext(new MmeStackCmd_TraceEvent(wkldId));
    }

    head.setLast(new MmeStackCmd_StrategyConv());
    head.setLast(new MmeStackCmd_PadConvAndTensors());

    *targetSoValue = 0;

    int numClasses = 1;
    for (int i = 0; i < conv->dim; i++)
    {
        numClasses *= conv->convStride[i];
    }

    for (int wc = 0; wc < numClasses; wc++)
    {
        ConvolutionParams new_conv;
        new_conv.relu = 0;
        new_conv.paddingValue.int32 = 0;
        new_conv.dim = conv->dim;

        int xOffsets[Mme::c_mme_max_tensor_dims] = {0};
        int wOffsets[Mme::c_mme_max_tensor_dims] = {0};
        int xStrides[Mme::c_mme_max_tensor_dims];
        int wStrides[Mme::c_mme_max_tensor_dims];
        int xSizes[Mme::c_mme_max_tensor_dims];
        int wSizes[Mme::c_mme_max_tensor_dims];

        x->copyStrides(xStrides);
        w->copyStrides(wStrides);
        x->copySizes(xSizes);
        w->copySizes(wSizes);

        int rem = wc;
        for (int convDim = conv->dim - 1; convDim >= 0; convDim--)
        {
            int weightDim = convDim + 2;
            wOffsets[weightDim] = rem % conv->convStride[convDim];
            rem /= conv->convStride[convDim];
        }

        bool validInput = true;

        for (int convDim = 0; convDim < conv->dim; convDim++)
        {
            int tensorDim = convDim + 1;
            int weightDim = convDim + 2;
            new_conv.dilation[convDim] = conv->dilation[convDim];
            new_conv.convStride[convDim] = 1;
            int RDMinusP = (conv->dilation[convDim] * wOffsets[weightDim]) - conv->padding[convDim];
            new_conv.padding[convDim] = -div_round_down(RDMinusP, conv->convStride[convDim]);
            xOffsets[tensorDim] = mod_neg(RDMinusP, conv->convStride[convDim]);
            xStrides[tensorDim] *= conv->convStride[convDim];
            xSizes[tensorDim] /= conv->convStride[convDim];
            wStrides[weightDim] *= conv->convStride[convDim];
            wSizes[weightDim] /= conv->convStride[convDim];

            if (xOffsets[tensorDim] < (x->getSize(tensorDim) % conv->convStride[convDim]))
            {
                xSizes[tensorDim]++;
            }

            if (wOffsets[weightDim] < (w->getSize(weightDim) % conv->convStride[convDim]))
            {
                wSizes[weightDim]++;
            }

            if (!xSizes[tensorDim] || !wSizes[weightDim])
            {
                validInput = false;
                break;
            }
        }

        if (validInput)
        {
            MmeSimTensor new_x =
                MmeSimTensor(xSizes, x->getDim(), x->getElementType(), x->getElementAt(xOffsets), xStrides);
            MmeSimTensor new_w =
                MmeSimTensor(wSizes, w->getDim(), w->getElementType(), w->getElementAt(wOffsets), wStrides);

            executeConvParams_t params = {0};
            memcpy(params.soAddrLow, soAddrLow, sizeof(params.soAddrLow));
            params.soAddrHigh = soAddrHigh;
            params.conv = &new_conv;
            params.a = y;
            params.b = &new_w;
            params.c = &new_x;
            params.op = executeConvParams_t::OP::OP_DEDX;
            params.testParams = localTestParams;
            params.strategy = strategy;
            params.roundingMode = roundingMode;
            params.conversionRoundingMode = conversionRoundingMode;
            params.roi.denseSize = new_x.getSize(0);
            getOutputSizes(&new_x, 0, &params.roi.spSize);

            head.execute(&params, &descList);
        }
    }

    descList2ExecPlan(prevDesc,
                      &descList,
                      cmds,
                      localTestParams->repeats,
                      localTestParams->incDec,
                      localTestParams->maskSignals,
                      targetSoValue,
                      lastDesc);
}

static void executeConvDEDW_verif(unsigned wkldId,
                                  const ConvolutionParams* conv,
                                  const uint32_t* soAddrLow,
                                  const uint32_t soAddrHigh,
                                  const MmeSimTensor* x,
                                  const MmeSimTensor* w,
                                  const MmeSimTensor* y,
                                  const RoundingMode roundingMode,
                                  const RoundingMode conversionRoundingMode,
                                  const MmeStrategy* strategy,
                                  const MmeTestParams* testParams,
                                  int* targetSoValue,
                                  std::list<MmeRegWriteCmd>* cmds,
                                  const Mme::Desc* prevDesc,
                                  Mme::Desc* lastDesc)
{
    MME_ASSERT_PTR(targetSoValue);
    MME_ASSERT_PTR(cmds);
    MME_ASSERT_PTR(x);
    MME_ASSERT_PTR(w);
    MME_ASSERT(x->getElementType() == y->getElementType(), "X and Y data type should match");
    MME_ASSERT(x->getSize(0) == w->getSize(1), "output height should match Xt height");
    MME_ASSERT(y->getSize(0) == w->getSize(0), "output width should match Y width");

    const MmeTestParams* localTestParams = testParams ? testParams : &c_default_test_params;

    MmeStackCmd head;
    if (localTestParams->wkldIdMD)
    {
        head.setLast(new MmeStackCmd_WkldIdMD(wkldId));
    }

    if (localTestParams->randomMD)
    {
        head.setLast(new MmeStackCmd_RandomDbg());
    }
    else
    {
        head.setNext(new MmeStackCmd_TraceEvent(wkldId));
    }

    head.setLast(new MmeStackCmd_StrategyDedw());
    head.setLast(new MmeStackCmd_PadConvAndTensors());

    *targetSoValue = 0;

    executeConvParams_t params = {0};
    memcpy(params.soAddrLow, soAddrLow, sizeof(params.soAddrLow));
    params.soAddrHigh = soAddrHigh;
    params.conv = conv;
    params.a = x;
    params.b = y;
    params.c = w;
    params.op = executeConvParams_t::OP::OP_DEDW;
    params.testParams = localTestParams;
    params.strategy = strategy;
    params.roundingMode = roundingMode;
    params.conversionRoundingMode = conversionRoundingMode;

    descList_t descList;
    head.execute(&params, &descList);

    descList2ExecPlan(prevDesc,
                      &descList,
                      cmds,
                      localTestParams->repeats,
                      localTestParams->incDec,
                      localTestParams->maskSignals,
                      targetSoValue,
                      lastDesc);
}

static void initMmeTensorView(const MmeSimTensor* tensor, MmeTensorView* view)
{
    view->elementType = tensor->getElementType();
    tensor->copySizes((int*) view->sizes.data());
    tensor->copyStrides((int*) view->strides.data());
    memset(view->bases.data(), 0, sizeof(view->bases));
    for (unsigned dim = tensor->getDim(); dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        view->strides[dim] = view->strides[dim - 1] * view->sizes[dim - 1];
        view->sizes[dim] = 1;
    }
}

static void executeConvAndBGemm(EMmeOpType opType,
                                unsigned wkldId,
                                const ConvolutionParams* conv,
                                const uint32_t* soAddrLow,
                                const uint32_t soAddrHigh,
                                const MmeSimTensor* x,
                                const MmeSimTensor* w,
                                const MmeSimTensor* y,
                                const RoundingMode roundingMode,
                                const RoundingMode conversionRoundingMode,
                                const MmeStrategy* strategy,
                                const MmeTestParams* testParams,
                                int* targetSoValue,
                                std::list<MmeRegWriteCmd>* cmds,
                                const Mme::Desc* prevDesc,
                                Mme::Desc* lastDesc)
{
    const MmeTestParams* localTestParams = testParams ? testParams : &c_default_test_params;

    MmeLayerParams layerParams = getMmeLayerParams(e_mme_Gaudi);
    layerParams.opType = opType;
    layerParams.controls.reluEn =
        conv->relu && (opType != EMmeOpType::e_mme_dedx) && (opType != EMmeOpType::e_mme_dedw);
    layerParams.controls.signalingMode = EMmeSignalingMode::e_mme_signaling_once;  // TODO: set dynamically
    layerParams.controls.roundingMode = roundingMode;
    layerParams.controls.conversionRoundingMode = conversionRoundingMode;
    layerParams.controls.atomicAdd = false;  // TODO: set dynamically
    layerParams.conv.paddingValue = *(float*) &conv->paddingValue.f32;
    if (x->getElementType() == e_type_bf16)
    {
        ((uint16_t*) &layerParams.conv.paddingValue)[1] = ((uint16_t*) &layerParams.conv.paddingValue)[0];
        ((uint16_t*) &layerParams.conv.paddingValue)[0] = 0;
    }

    for (int i = 0; i < Mme::c_mme_max_conv_dims - 1; i++)
    {
        layerParams.conv.stride[i] = conv->convStride[i];
        layerParams.conv.dilation[i] = conv->dilation[i];
        layerParams.conv.padding[i] = conv->padding[i];
    }
    for (unsigned dim = conv->dim; dim < Mme::c_mme_max_conv_dims - 1; dim++)
    {
        layerParams.conv.stride[dim] = 1;
        layerParams.conv.dilation[dim] = 1;
        layerParams.conv.padding[dim] = 0;
    }
    layerParams.tracing.ctxId = wkldId;
    layerParams.tracing.traceModeX = EMmeTraceMode::e_mme_trace_mode_layer_act;
    layerParams.tracing.traceModeY = EMmeTraceMode::e_mme_trace_mode_layer_act;
    layerParams.tracing.traceModeW = EMmeTraceMode::e_mme_trace_mode_layer_act;
    initMmeTensorView(x, &layerParams.x);
    initMmeTensorView(y, &layerParams.y);
    initMmeTensorView(w, &layerParams.w);
    layerParams.spBase = 0;
    layerParams.spSize = 1;
    for (unsigned dim = 1; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        layerParams.spSize *= (opType == EMmeOpType::e_mme_dedx) ? layerParams.x.sizes[dim] : layerParams.y.sizes[dim];
    }
    layerParams.strategy.loweringEn = strategy->loweringEn;
    layerParams.strategy.sbReuse = strategy->sbReuse || (getenv("FORCE_SB_REUSE") != nullptr);
    layerParams.strategy.unrollEn = strategy->unrollEn;
    layerParams.strategy.memsetDedxVoidPixels = strategy->memsetDedxVoidPixels;
    layerParams.strategy.dedwAsBgemmEn = strategy->dedwAsBgemmEn;
    layerParams.strategy.recurringMisalignmentOptEn = strategy->recurringMisalignmentOptEn;
    switch (strategy->pattern)
    {
        case e_mme_sp_reduction_kfc:
            layerParams.strategy.pattern = e_mme_sp_reduction_kfc;
            break;
        case e_mme_sp_reduction_fkc:
            layerParams.strategy.pattern = e_mme_sp_reduction_fkc;
            break;
        case e_mme_sp_reduction_fck:
            layerParams.strategy.pattern = e_mme_sp_reduction_fck;
            break;
        case e_mme_sp_reduction_cfk:
            layerParams.strategy.pattern = e_mme_sp_reduction_cfk;
            break;
        case e_mme_sp_reduction_kcf:
            layerParams.strategy.pattern = e_mme_sp_reduction_kcf;
            break;
        case e_mme_sp_reduction_ckf:
            layerParams.strategy.pattern = e_mme_sp_reduction_ckf;
            break;
        case e_mme_z_reduction_ksf:
            layerParams.strategy.pattern = e_mme_z_reduction_ksf;
            break;
        case e_mme_z_reduction_skf:
            layerParams.strategy.pattern = e_mme_z_reduction_skf;
            break;
        default:
            MME_ASSERT(0, "invalid pattern");
    }
    switch (strategy->geometry)
    {
        case e_mme_geometry_1wx4h:
            layerParams.strategy.geometry = e_mme_geometry_1wx4h;
            break;
        case e_mme_geometry_2wx2h:
            layerParams.strategy.geometry = e_mme_geometry_2wx2h;
            break;
        case e_mme_geometry_4wx1h:
            layerParams.strategy.geometry = e_mme_geometry_4wx1h;
            break;
        default:
            MME_ASSERT(0, "invalid geometry");
    }

    std::list<gaudi::MmeActivation> activations;

    atomicColoredPrint(COLOR_CYAN, "INFO: Generating code for mme descriptors.\n");
    auto descGenerator = gaudi::MmeDescriptorGenerator::createMmeDescGenerator(layerParams);
    descGenerator->mmeGenerateActivations();
    activations = descGenerator->getMmeActivations();
    const auto recipeDebugInfo = descGenerator->getRecipeDebugInfo();
    if (!recipeDebugInfo.empty())
    {
        for (const auto& dbgStr : recipeDebugInfo)
        {
            atomicColoredPrint(COLOR_BLUE, "[DEBUG] %s\n", dbgStr.c_str());
        }
    }
    dumpCompareDescriptors(activations);

    descList_t descList;
    unsigned localTargetSoValue = 0;
    for (auto& act : activations)
    {
        localTargetSoValue += act.numSignals;
        gaudi::patchTensorView(EMmeOperand::e_mme_op_x,
                               act.getDesc(0),
                               act.getDesc(1),
                               (uint64_t) x->data(),
                               act.isGemm);
        gaudi::patchTensorView(EMmeOperand::e_mme_op_y,
                               act.getDesc(0),
                               act.getDesc(1),
                               (uint64_t) y->data(),
                               act.isGemm);
        gaudi::patchTensorView(EMmeOperand::e_mme_op_w,
                               act.getDesc(0),
                               act.getDesc(1),
                               (uint64_t) w->data(),
                               act.isGemm);
        MME_ASSERT(soAddrLow[0] == soAddrLow[1] - sizeof(uint32_t), "");
        MME_ASSERT(soAddrLow[1] == soAddrLow[2] - sizeof(uint32_t), "");
        MME_ASSERT(soAddrLow[2] == soAddrLow[3] - sizeof(uint32_t), "");
        gaudi::patchSyncObject(act.getDesc(0), Mme::e_mme_local, (((uint64_t) soAddrHigh) << 32) + soAddrLow[0]);
        gaudi::patchSyncObject(act.getDesc(0), Mme::e_mme_remote, (((uint64_t) soAddrHigh) << 32) + soAddrLow[1]);
        gaudi::patchSyncObject(act.getDesc(1), Mme::e_mme_local, (((uint64_t) soAddrHigh) << 32) + soAddrLow[2]);
        gaudi::patchSyncObject(act.getDesc(1), Mme::e_mme_remote, (((uint64_t) soAddrHigh) << 32) + soAddrLow[3]);

        descGroup_t group;
        group.desc[0] = act.getDesc(0);
        group.desc[1] = act.getDesc(1);
        descList.push_back(group);
    }
    if (GCFG_PRINT_MME_DESCRIPTORS.value())
    {
        auto dump = descGenerator->dumpDescriptors(false);
        for (const auto& actDump : dump)
        {
            for (const auto& descDump : actDump)
            {
                atomicColoredPrint(COLOR_BLUE, "%s\n", descDump.c_str());
            }
        }
    }
    descList2ExecPlan(prevDesc,
                      &descList,
                      cmds,
                      localTestParams->repeats,
                      localTestParams->incDec,
                      localTestParams->maskSignals,
                      targetSoValue,
                      lastDesc);
    MME_ASSERT(localTargetSoValue == *targetSoValue, "");
}

void executeBGemm(EMmeOpType opType,
                  const bool verifMode,
                  unsigned wkldId,
                  const ConvolutionParams* conv,
                  const uint32_t* soAddrLow,
                  const uint32_t soAddrHigh,
                  const MmeSimTensor* x,
                  const MmeSimTensor* w,
                  const MmeSimTensor* y,
                  const RoundingMode roundingMode,
                  const RoundingMode conversionRoundingMode,
                  const MmeStrategy* strategy,
                  const MmeTestParams* testParams,
                  int* targetSoValue,
                  std::list<MmeRegWriteCmd>* cmds,
                  const Mme::Desc* prevDesc,
                  Mme::Desc* lastDesc)
{
    executeConvAndBGemm(opType,
                        wkldId,
                        conv,
                        soAddrLow,
                        soAddrHigh,
                        x,
                        w,
                        y,
                        roundingMode,
                        conversionRoundingMode,
                        strategy,
                        testParams,
                        targetSoValue,
                        cmds,
                        prevDesc,
                        lastDesc);
}

void executeConvFWD(const bool verifMode,
                    unsigned wkldId,
                    const ConvolutionParams* conv,
                    const uint32_t* soAddrLow,
                    const uint32_t soAddrHigh,
                    const MmeSimTensor* x,
                    const MmeSimTensor* w,
                    const MmeSimTensor* y,
                    const RoundingMode roundingMode,
                    const RoundingMode conversionRoundingMode,
                    const MmeStrategy* strategy,
                    const MmeTestParams* testParams,
                    int* targetSoValue,
                    std::list<MmeRegWriteCmd>* cmds,
                    const Mme::Desc* prevDesc,
                    Mme::Desc* lastDesc)
{
    if (verifMode)
    {
        executeConvFWD_verif(wkldId,
                             conv,
                             soAddrLow,
                             soAddrHigh,
                             x,
                             w,
                             y,
                             roundingMode,
                             conversionRoundingMode,
                             strategy,
                             testParams,
                             targetSoValue,
                             cmds,
                             prevDesc,
                             lastDesc);
    }
    else
    {
        executeConvAndBGemm(EMmeOpType::e_mme_fwd,
                            wkldId,
                            conv,
                            soAddrLow,
                            soAddrHigh,
                            x,
                            w,
                            y,
                            roundingMode,
                            conversionRoundingMode,
                            strategy,
                            testParams,
                            targetSoValue,
                            cmds,
                            prevDesc,
                            lastDesc);
    }
}

void executeConvDEDX(const bool verifMode,
                     unsigned wkldId,
                     const ConvolutionParams* conv,
                     const uint32_t* soAddrLow,
                     const uint32_t soAddrHigh,
                     const MmeSimTensor* x,
                     const MmeSimTensor* w,
                     const MmeSimTensor* y,
                     const RoundingMode roundingMode,
                     const RoundingMode conversionRoundingMode,
                     const MmeStrategy* strategy,
                     const MmeTestParams* testParams,
                     int* targetSoValue,
                     std::list<MmeRegWriteCmd>* cmds,
                     const Mme::Desc* prevDesc,
                     Mme::Desc* lastDesc)
{
    if (verifMode)
    {
        executeConvDEDX_verif(wkldId,
                              conv,
                              soAddrLow,
                              soAddrHigh,
                              x,
                              w,
                              y,
                              roundingMode,
                              conversionRoundingMode,
                              strategy,
                              testParams,
                              targetSoValue,
                              cmds,
                              prevDesc,
                              lastDesc);
    }
    else
    {
        executeConvAndBGemm(EMmeOpType::e_mme_dedx,
                            wkldId,
                            conv,
                            soAddrLow,
                            soAddrHigh,
                            x,
                            w,
                            y,
                            roundingMode,
                            conversionRoundingMode,
                            strategy,
                            testParams,
                            targetSoValue,
                            cmds,
                            prevDesc,
                            lastDesc);
    }
}

void executeConvDEDW(const bool verifMode,
                     unsigned wkldId,
                     const ConvolutionParams* conv,
                     const uint32_t* soAddrLow,
                     const uint32_t soAddrHigh,
                     const MmeSimTensor* x,
                     const MmeSimTensor* w,
                     const MmeSimTensor* y,
                     const RoundingMode roundingMode,
                     const RoundingMode conversionRoundingMode,
                     const MmeStrategy* strategy,
                     const MmeTestParams* testParams,
                     int* targetSoValue,
                     std::list<MmeRegWriteCmd>* cmds,
                     const Mme::Desc* prevDesc,
                     Mme::Desc* lastDesc)
{
    if (verifMode)
    {
        executeConvDEDW_verif(wkldId,
                              conv,
                              soAddrLow,
                              soAddrHigh,
                              x,
                              w,
                              y,
                              roundingMode,
                              conversionRoundingMode,
                              strategy,
                              testParams,
                              targetSoValue,
                              cmds,
                              prevDesc,
                              lastDesc);
    }
    else
    {
        executeConvAndBGemm(EMmeOpType::e_mme_dedw,
                            wkldId,
                            conv,
                            soAddrLow,
                            soAddrHigh,
                            x,
                            w,
                            y,
                            roundingMode,
                            conversionRoundingMode,
                            strategy,
                            testParams,
                            targetSoValue,
                            cmds,
                            prevDesc,
                            lastDesc);
    }
}
