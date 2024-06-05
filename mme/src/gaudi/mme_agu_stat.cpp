//#define NDEBUG
#include <climits>
#include <list>
#include "gaudi/gaudi.h"
#include "containers.h"
#include "include/gaudi/mme_agu_stat.h"
#include "include/mme_assert.h"

#define ROI_CALC_CACHE_SIZE 8

struct FixedAguParams
{
    const Mme::MmeTensorDesc *tensorDesc;
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

static void genPCLoops(
        const Mme::Desc* desc,
        bool isInput,
        bool isShared,
        std::list<unsigned> *loops)
{
    unsigned loopSize[Mme::c_mme_max_conv_dims + 2];
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
                        SmallCircularFIFOCache<uint64_t, ROI_CALC_CACHE_SIZE>& cache,
                        std::list<AguDescStat>* stats,
                        gaudi::AguRanges* ranges)
{
    int height = (loopEndMask & fp->partialHeightLoopMask) ? fp->heightLast : fp->height;
    unsigned fcd = (loopEndMask & fp->fcdLoopMask) ? fp->fcdLast : fp->fcd;

    int elementsInCL = DEVICE_CACHE_LINE_SIZE / fp->elementSize;
    int64_t currOffset[Mme::c_mme_max_tensor_dims];
    int64_t targetOffset[Mme::c_mme_max_tensor_dims - 1];

    bool cacheInvalidate = false;
    bool newConv = true;
    if (isInput)
    {
        newConv = ((fp->accumMask & loopStartMask) == fp->accumMask);
        bool newDesc = (loopStartMask & ((1 << (Mme::c_mme_max_conv_dims + 2)) - 1)) == ((1 << (Mme::c_mme_max_conv_dims + 2)) - 1);
        cacheInvalidate = newDesc || (newConv && fp->loopsNum);
    }

    AguDescStat *currStats = nullptr;
    if (stats)
    {
        if (newConv)
        {
            stats->push_back({0});
        }
        currStats = &stats->back();
    }


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

        for (int spStep = 0; spStep < height; spStep++)
        {
            int64_t denseOffset;
            int64_t denseTarget;
            if (fp->lower)
            {
                denseOffset = currOffset[0] + currOffset[1];
                denseTarget = std::min((int64_t)(currOffset[1] + roiBase[0] + fcd), (int64_t)(fp->tensorDesc->validElements[1]));
                denseTarget = std::min(denseTarget, currOffset[1] + fp->tensorDesc->validElements[0]);
            }
            else
            {
                denseOffset = currOffset[0];
                denseTarget = std::min((int64_t)targetOffset[0], (int64_t)fp->tensorDesc->validElements[0]);
            }

            bool pad = false;
            for (int spDim = fp->lower ? 2 : 1; spDim < Mme::c_mme_max_tensor_dims; spDim++)
            {
                if ((currOffset[spDim] < 0) ||
                    (currOffset[spDim] >= fp->tensorDesc->validElements[spDim]))
                {
                    pad = true;
                    break;
                }
            }

            if (cacheInvalidate)
            {
                cache.clear();
                cacheInvalidate = false;
            }

            int64_t denseEndOffset = denseOffset + elementsInCL;
            int64_t padMsb = denseEndOffset > denseTarget ? denseEndOffset - denseTarget : 0;
            int64_t padLsb = denseOffset < 0 ? -denseOffset : 0;

            if (stats)
            {
                currStats->numVectors++;
                currStats->numSBEntries++;
                currStats->numCycles++;
            }

            if (!pad && (padLsb < elementsInCL) && (padMsb < elementsInCL))
            {
                uint64_t p = fp->baseAddr;

                for (int dim = 0; dim < Mme::c_mme_max_tensor_dims; dim++)
                {
                    p += currOffset[dim] * fp->elementSize;
                }

                unsigned lpadInBytes = padLsb * fp->elementSize;
                unsigned mpadInBytes = padMsb * fp->elementSize;

                if (ranges && (DEVICE_CACHE_LINE_SIZE - lpadInBytes - mpadInBytes > 0))
                {
                    ranges->addSegment(p + lpadInBytes, DEVICE_CACHE_LINE_SIZE - lpadInBytes - mpadInBytes, 0);
                }

                if (stats)
                {
                    uint64_t p0 = (p + lpadInBytes) & ~(((uint64_t)DEVICE_CACHE_LINE_SIZE) - 1);
                    bool p0Hit = false;
                    if (isInput)
                    {
                        p0Hit = cache.insert(p0);
                        if (!p0Hit) currStats->numAccesses++;
                    }

                    uint64_t p1 = p0 + DEVICE_CACHE_LINE_SIZE;
                    bool p1Hit;
                    if (p1 < (p + DEVICE_CACHE_LINE_SIZE - mpadInBytes))
                    {
                        currStats->numSBEntries++;
                        p1Hit = false;
                        if (isInput)
                        {
                            p1Hit = cache.insert(p1);
                            if (!p1Hit) currStats->numAccesses++;
                        }
                    }
                    else
                    {
                        p1Hit = true;
                    }

                    if (!p0Hit && !p1Hit)
                    {
                        currStats->numCycles++;
                    }
                }
            }

            int inc = 1;
            for (int spDim = 1; inc > 0; spDim++)
            {
                currOffset[spDim] += ((uint64_t) fp->tensorDesc->spatialStrides[spDim - 1] * (int64_t) inc);
                inc = 0;
                if (spDim < Mme::c_mme_max_tensor_dims - 1)
                {
                    for (;
                            currOffset[spDim] >= targetOffset[spDim];
                            currOffset[spDim] -= fp->tensorDesc->roiSize[spDim])
                    {
                        inc++;
                    }
                }
                MME_ASSERT(inc <= 1, "did not expect inc > 1");
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

static void getDescStat(const Mme::MmeHalf half,
                        const bool isInput,
                        const bool isShared,
                        const Mme::Desc* desc,
                        const unsigned eventBase,
                        std::list<AguDescStat>* stats,
                        std::vector<gaudi::AguRanges>* ranges)
{
    unsigned signalCtr = eventBase;
    bool advance;
    const Mme::MmeAguCoreDesc * aguDesc;
    bool enable;
    Mme::MmeAguCoreDesc dummyAguDesc;
    Mme::MmeTensorDesc dummyTensorDesc;

    unsigned matSize = Mme::c_mme_matrix_size >> ((desc->header.dataTypeIn == Mme::e_mme_dt_sp) ? 1 : 0);
    Mme::MmeAssociatedDims assocDimMask = { 0 };
    Mme::MmeAssociatedDims assocDimShift = { 0 };
    FixedAguParams fp;

    if (isInput && isShared)
    {
        assocDimMask.dimS = 1;
        assocDimMask.dimS = -assocDimMask.dimS;
        assocDimShift.dimS = 1;
        fp.accumMask = desc->header.accumMask;
        fp.signalMask = -1;
        fp.elementSize = (desc->header.dataTypeIn == Mme::e_mme_dt_sp) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        aguDesc = &desc->aguS;
        fp.baseAddr = (((uint64_t)desc->baseAddrHighS) << 32) + desc->baseAddrLowS;
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
        assocDimMask.dimL = 1;
        assocDimMask.dimL = -assocDimMask.dimL;
        assocDimShift.dimL = 1;
        fp.accumMask = desc->header.accumMask;
        fp.signalMask = -1;
        fp.elementSize = (desc->header.dataTypeIn == Mme::e_mme_dt_sp) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        aguDesc = &desc->aguL[half];
        fp.baseAddr = (((uint64_t)desc->baseAddrHighL) << 32) + desc->baseAddrLowL;
        fp.tensorDesc = &desc->tensorL;
        advance = desc->header.advanceL;
        fp.lower = desc->header.lowerL;
        fp.loopsNum = desc->sbRepeat.repeatLMinus1;
        enable = desc->sbRepeat.loadL;
        if (desc->header.transL)
        {
            fp.partialHeightLoopMask = (half == Mme::e_mme_local) ?
                                       desc->header.partialHeightLoopLLocal : desc->header.partialHeightLoopLRemote;
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
            fp.fcdLoopMask = (half == Mme::e_mme_local) ?
                             desc->header.partialHeightLoopLLocal : desc->header.partialHeightLoopLRemote;
            fp.height = fp.tensorDesc->spatialSizeMinus1 + 1;
            fp.heightLast = fp.height;
            fp.fcd = matSize;
            fp.fcdLast = fp.tensorDesc->roiSize[0] % matSize;
            fp.fcdLast = fp.fcdLast ? fp.fcdLast : matSize;
        }
    }
    else
    {
        assocDimMask.dimO = 1;
        assocDimMask.dimO = -assocDimMask.dimO;
        assocDimShift.dimO = 1;
        fp.accumMask = desc->header.accumMask;
        fp.signalMask = desc->header.signalEn ? desc->header.signalMask : -1;
        fp.elementSize = (desc->header.dataTypeOut == Mme::e_mme_dt_sp) ? sizeof(Mme::f32_t) : sizeof(Mme::bf16_t);
        aguDesc = &desc->aguO[half];
        fp.baseAddr = (((uint64_t)desc->baseAddrHighO) << 32) + desc->baseAddrLowO;
        fp.tensorDesc = &desc->tensorO;
        advance = desc->header.advanceO;
        fp.lower = false;
        enable = !desc->header.accStoreIncDisable & (desc->header.storeEn || desc->header.signalEn);
        fp.loopsNum = 0;

        fp.partialHeightLoopMask = 1 << Mme::c_mme_max_conv_dims;
        fp.height = matSize;
        fp.heightLast = desc->tensorO.spatialSizeMinus1 + 1;
        fp.fcdLoopMask = (half == Mme::e_mme_local) ?
                         desc->header.partialHeightLoopOLocal : desc->header.partialHeightLoopORemote;
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

    // Last vertical step cannot exceed the output height, unless transO is set which means output dims are transposed
    MME_ASSERT(desc->header.transO || fp.heightLast <= fp.height, "");

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
        associatedDim[Mme::c_mme_max_conv_dims + 1] = (desc->outerLoop.associatedDims.w & assocDimMask.w) / assocDimShift.w;

        int64_t startOffsets[Mme::c_mme_max_tensor_dims - 1] = {0LL};
        int64_t nextStartOffsets[Mme::c_mme_max_tensor_dims - 1] = {0LL};
        int64_t roiBase[Mme::c_mme_max_tensor_dims];
        for (int i = 0; i < Mme::c_mme_max_tensor_dims; i++)
        {
            roiBase[i] = aguDesc->roiBaseOffset[i];
        }

        SmallCircularFIFOCache<uint64_t, ROI_CALC_CACHE_SIZE> cache;

        unsigned first = (1 << (Mme::c_mme_max_conv_dims + 2)) - 1;
        unsigned reset = (1 << (Mme::c_mme_max_conv_dims + 2)) - 1;

        for (auto & last : gemms)
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

            bool advanceStartOffset = advance && ((reset & ((1 << Mme::c_mme_max_conv_dims) - 1)) == ((1 << Mme::c_mme_max_conv_dims) - 1));

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
                        cache,
                        stats,
                        ranges ? &(*ranges)[signalCtr] : nullptr);

            if (desc->header.signalEn && ((last & desc->header.signalMask) == desc->header.signalMask))
            {
                signalCtr++;
            }

            reset = last & ~(last + 1);
            first |= reset;
        }
    }
}

unsigned gaudi::aguStatsCountSBReads(bool isShared, const Mme::Desc *northDesc, const Mme::Desc *southDesc)
{
    unsigned reads = 0;
    if (northDesc->header.transO)
    {
        isShared = !isShared;
    }

    std::list<AguDescStat> stats;

    if (isShared)
    {
        getDescStat(Mme::e_mme_remote, true, true, southDesc, 0, &stats, nullptr);
        getDescStat(Mme::e_mme_remote, true, true, northDesc, 0, &stats, nullptr);
    }
    else
    {
        getDescStat(Mme::e_mme_local, true, false, southDesc, 0, &stats, nullptr);
        getDescStat(Mme::e_mme_local, true, false, northDesc, 0, &stats, nullptr);
        getDescStat(Mme::e_mme_remote, true, false, southDesc, 0, &stats, nullptr);
        getDescStat(Mme::e_mme_remote, true, false, northDesc, 0, &stats, nullptr);
    }

    for (auto &stat : stats)
    {
        reads = std::max(reads, stat.numSBEntries);
    }

    return reads;
}

void gaudi::aguStatsGetRanges(const bool isInput,
                              const bool isShared,
                              const Mme::Desc* desc,
                              const unsigned eventBase,
                              std::vector<AguRanges>* ranges)
{
    bool skipLinearRanges = (getenv("GAUDI_DEBUG_SKIP_LINEAR_RANGES") != nullptr);
    if (skipLinearRanges)
    {
        return;
    }

    if (!isInput)
    {
        getDescStat(Mme::e_mme_local, false, false, desc, eventBase, nullptr, ranges);
        getDescStat(Mme::e_mme_remote, false, false, desc, eventBase, nullptr, ranges);
    }
    else if (isShared)
    {
        getDescStat(Mme::e_mme_remote, true, true, desc, eventBase, nullptr, ranges);
    }
    else
    {
        getDescStat(Mme::e_mme_local, true, false, desc, eventBase, nullptr, ranges);
        getDescStat(Mme::e_mme_remote, true, false, desc, eventBase, nullptr, ranges);
    }
}
