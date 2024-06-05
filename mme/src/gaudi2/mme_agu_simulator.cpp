#include <fstream>
#include <iterator>
#include <list>
#include <numeric>
#include <vector>

#include "include/gaudi2/gaudi2_utils.h"
#include "mme_agu_simulator.h"
#include "mme_assert.h"

#define FS_ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))
#define mme_min(a, b)        (((a) < (b)) ? (a) : (b))
#define mme_div_ceil(n, d)   (((n) + (d) -1) / (d))

static const unsigned c_matrix_height = Gaudi2::Mme::c_mme_dcore_matrix_height_in_bytes / sizeof(uint16_t);
static const unsigned c_matrix_width = Gaudi2::Mme::c_mme_dcore_matrix_width_in_bytes / sizeof(uint16_t);

using namespace MmeCommon;

namespace Gaudi2
{
struct FixedParams
{
    Gaudi2::Mme::MmeTensorDesc tensorDesc;
    unsigned spatialSizeMinus1;
    uint64_t baseAddr;
    int loopMask;
    int partialHeightLoopMask;
    int fcdLoopMask;
    int logElementSize;
    int height;
    int heightLast;
    unsigned fcd;
    unsigned fcdLast;
    bool lower;
    Gaudi2::Mme::MmeSyncObject syncObject;
    bool fp32nonTransWalk;
    bool noData = false;
    bool signalOnly = false;

    bool transEn;
    unsigned signalMask;
    bool signalEn;
};

inline bool isOutputAgu(const EMmeOperandIdx aguId)
{
    return (aguId == e_mme_agu_cout0_idx || aguId == e_mme_agu_cout1_idx);
}

inline unsigned getNumOfElementsInChannel(const EMmeDataType dataType)
{
    return (Mme::c_cl_size / getElementSize(dataType));
}

void getSlaveSyncObject(const Mme::Desc* desc, Mme::MmeSyncObject* syncObject)
{
    if (desc->syncObject.slave0UseSlaveSOAddr)
    {
        syncObject->so0Addr = desc->slaveSyncObject0Addr;
    }
    else if (desc->syncObject.slave0UseMasterSOAddrPlus4)
    {
        syncObject->so0Addr += 4;
    }

    if (desc->syncObject.slave1UseSlaveSOAddr)
    {
        syncObject->so1Addr = desc->slaveSyncObject1Addr;
    }
    else if (desc->syncObject.slave1UseMasterSOAddrPlus4)
    {
        syncObject->so1Addr += 4;
    }
}

void swapBaseAndOffset(Gaudi2::Mme::MmeAguCoreDesc* aguDesc, Gaudi2::Mme::MmeTensorDesc* tensorDesc)
{
    for (int dim = 1; dim < Mme::c_mme_max_conv_dims; dim++)
    {
        int32_t swap = aguDesc->roiBaseOffset[dim];
        aguDesc->roiBaseOffset[dim] = tensorDesc->startOffset[dim - 1];
        tensorDesc->startOffset[dim - 1] = swap;
    }
}

static bool isOutputLoop(const Mme::Desc& desc, unsigned loop)
{
    unsigned coutLoopMask = desc.brains.aguOut0.loopMask;
    MME_ASSERT(desc.brains.aguOut0.loopMask == desc.brains.aguOut1.loopMask,
              "loop mask should be the same in aguOut0 and aguOut1");
    return ((loop & coutLoopMask) == coutLoopMask);
}

static unsigned calcOutputsNr(const Mme::Desc& desc, const std::vector<unsigned>& gemms)
{
    unsigned outputs = 0;
    for (const auto& gemm : gemms)
    {
        if (isOutputLoop(desc, gemm))
        {
            outputs++;
        }
    }

    return outputs;
}

static unsigned calcSignalBase(const unsigned outputsNr,
                               const unsigned outputsToCalc,
                               const unsigned signalMask,
                               const Mme::EMmeLoopMask loopmask)
{
    unsigned signalBase = 0;
    // if we signal more than once, skip gemms that repeat the same patterns seen before
    if (signalMask < Mme::e_mme_outer_loop)
    {
        signalBase = outputsNr - outputsToCalc;
        // the skipped gemms dont signal.
        // this means that the whole loop is represented as a single element in the ranges array.
        // divide the signalBase to represnt that.
        if (signalMask == loopmask)
        {
            // this code takes into account only a single loop
            // this is good enough for fwd, but might not be enough for other operations.
            signalBase /= outputsToCalc;
        }
    }

    return signalBase;
}
static unsigned getGemmsSkips(const unsigned aguMask,
                              const EMmeOpType opType,
                              const Mme::Desc* desc,
                              const unsigned outputsNr,
                              unsigned* signalBase)
{
    unsigned outputsToCalc = outputsNr;
    *signalBase = 0;
    if (opType == e_mme_fwd)
    {
        const bool isA = desc->header.aguReadsA & aguMask;
        const bool isB = desc->header.aguReadsB & aguMask;
        const bool isSkf = (desc->conv.kernelSizeMinus1.dim[Mme::c_mme_max_conv_dims - 1] > 0);

        if (isB && isSkf)
        {
            outputsToCalc = 1 + desc->conv.kernelSizeMinus1.dim[Mme::c_mme_max_conv_dims - 1];
            MME_ASSERT(outputsNr >= outputsToCalc, "cannot have less outputs than outputsToCalc");
            *signalBase =
                calcSignalBase(outputsNr, outputsToCalc, desc->syncObject.signalMask0, Mme::e_mme_conv_loop_3);
        }
        else if (isA && !isSkf)
        {
            outputsToCalc = desc->numIterationsMinus1 + 1;
            MME_ASSERT(outputsNr >= outputsToCalc, "cannot have less outputs than outputsToCalc");
            *signalBase =
                calcSignalBase(outputsNr, outputsToCalc, desc->syncObject.signalMask0, Mme::e_mme_tetris_loop);
        }
    }

    return outputsToCalc;
}

static unsigned calcNumSignalsPerLoop(const Mme::Desc* desc,
                                      unsigned loopMask,
                                      unsigned signalMask,
                                      bool signalEn,
                                      unsigned* totalSignals)
{
    if (!signalEn)
    {
        return 0;
    }
    unsigned signalTimes = 0;
    *totalSignals = 0;
    unsigned loopSize[Mme::c_mme_max_conv_dims + 2];

    // set conv[0..3] loop sizes
    for (int i = 0; i < Mme::c_mme_max_conv_dims; i++)
    {
        loopSize[i] = desc->conv.kernelSizeMinus1.dim[i] + 1;
    }

    // set spatial loop size
    loopSize[Mme::c_mme_max_conv_dims] = desc->numIterationsMinus1 + 1;
    // set outer loop size
    loopSize[Mme::c_mme_max_conv_dims + 1] = desc->outerLoop.sizeMinus1 + 1;

    // this loop calculates:
    // 1. if loop[i] is masked then set loopSize[i]=1
    // 2. totalIter - total number of iterations
    unsigned totalIter = 1;
    for (int i = 0; i < sizeof(loopSize) / sizeof((loopSize)[0]); i++)
    {
        totalIter *= loopSize[i];
    }

    // go over all iterations and calc 'end' mask
    for (unsigned j = 0; j < totalIter; j++)
    {
        unsigned rem = j;
        unsigned end = 0;
        for (int i = 0; i < sizeof(loopSize) / sizeof((loopSize)[0]); i++)
        {
            end |= ((rem % loopSize[i]) == (loopSize[i] - 1)) << i;
            rem /= loopSize[i];
        }
        if ((end & signalMask) == signalMask)
        {
            ++(*totalSignals);
            if ((end & loopMask) == loopMask)
            {
                ++signalTimes;
            }
        }
    }

    if (signalTimes > 1)
    {
        MME_ASSERT(*totalSignals % signalTimes == 0, "totalSignals should be aligned to loop iterations");
        return *totalSignals / signalTimes;
    }
    else
    {
        return 0;
    }
}

static std::vector<unsigned> genPCLoops(const Gaudi2::Mme::Desc* desc, unsigned mask)
{
    unsigned loopSize[Gaudi2::Mme::c_mme_max_conv_dims + 2];

    // set conv[0..3] loop sizes
    for (int i = 0; i < Gaudi2::Mme::c_mme_max_conv_dims; i++)
    {
        loopSize[i] = desc->conv.kernelSizeMinus1.dim[i] + 1;
    }

    // set spatial loop size
    loopSize[Gaudi2::Mme::c_mme_max_conv_dims] = desc->numIterationsMinus1 + 1;
    // set outer loop size
    loopSize[Gaudi2::Mme::c_mme_max_conv_dims + 1] = desc->outerLoop.sizeMinus1 + 1;

    // this loop calculates:
    // 1. if loop[i] is masked then set loopSize[i]=1
    // 2. totalIter - total number of iterations
    unsigned totalIter = 1;
    for (int i = 0; i < FS_ARRAY_SIZE(loopSize); i++)
    {
        loopSize[i] = (mask & (1 << i)) ? 1 : loopSize[i];
        totalIter *= loopSize[i];
    }

    std::vector<unsigned> loops;
    loops.reserve(totalIter);

    // go over all iterations and calc 'end' mask
    for (unsigned j = 0; j < totalIter; j++)
    {
        unsigned rem = j;
        unsigned end = 0;
        for (int i = 0; i < FS_ARRAY_SIZE(loopSize); i++)
        {
            end |= ((rem % loopSize[i]) == (loopSize[i] - 1)) << i;
            rem /= loopSize[i];
        }
        loops.push_back(end);
    }
    return loops;
}

void getAssociatedDims(unsigned* associatedDim, const EMmeOperandIdx aguId, const Gaudi2::Mme::Desc* desc)
{
    Mme::MmeAssociatedDims assocDimMask = {};
    Mme::MmeAssociatedDims assocDimShift = {};

    if (!isOutputAgu(aguId))  // input AGU
    {
        unsigned aguMask = 1 << aguId;
        bool aguReadsA = desc->header.aguReadsA & aguMask;
        if (aguReadsA)
        {
            assocDimMask.dimA = -1;
            assocDimShift.dimA = 1;
        }
        else
        {
            assocDimMask.dimB = -1;
            assocDimShift.dimB = 1;
        }
    }
    else  // output AGU
    {
        assocDimMask.dimOut = -1;
        assocDimShift.dimOut = 1;
    }

    // Conv loops
    for (int i = 0; i < Gaudi2::Mme::c_mme_max_conv_dims; i++)
    {
        associatedDim[i] = (desc->conv.associatedDims[i].w & assocDimMask.w) / assocDimShift.w;
    }
    // Spatial loop
    associatedDim[Gaudi2::Mme::c_mme_max_conv_dims] = Gaudi2::Mme::c_mme_max_tensor_dims;
    // Outer loop
    associatedDim[Gaudi2::Mme::c_mme_max_conv_dims + 1] =
        (desc->outerLoop.associatedDims.w & assocDimMask.w) / assocDimShift.w;
}

void genGemmAddresses(const FixedParams* fp,
                      const EMmeOperandIdx aguId,
                      const bool advance,
                      const int64_t* startOffsets,  // The current dim offsets.
                      int64_t* nextStartOffsets,  // The next dim offsets.
                      const int64_t* roiBase,
                      unsigned loopStartMask,
                      unsigned loopEndMask,
                      AguRanges* ranges)
{
    int height = (loopEndMask & fp->partialHeightLoopMask) ? fp->heightLast : fp->height;
    unsigned fcd = (loopEndMask & fp->fcdLoopMask) ? fp->fcdLast : fp->fcd;

    int elementsInFullCL = Mme::c_cl_size >> fp->logElementSize;

    int elementsInCL = elementsInFullCL;

    int64_t currOffset[Gaudi2::Mme::c_mme_max_tensor_dims];
    int64_t targetOffset[Gaudi2::Mme::c_mme_max_tensor_dims - 1];

    targetOffset[0] = roiBase[0] + fcd;
    for (int i = 1; i < Gaudi2::Mme::c_mme_max_tensor_dims - 1; i++)
    {
        targetOffset[i] = roiBase[i] + fp->tensorDesc.roiSize[i];
    }

    // Dense AGU performs 2 loops:
    // - for aguIn transposed (1/2B elements): first move down (inner loop) and
    // then move right (outer loop)
    // - for aguIn non-transposed (4B elements): first move right (inner loop) and
    // then move down (outer loop)
    // - for agu cout: first move right (inner loop) and then move down (outer
    // loop) we do it by coding 3 loops (height->fcd->height) and masking 1
    // according to agu type
    bool firstRightThenDown = fp->fp32nonTransWalk || isOutputAgu(aguId);
    int outerHeight = firstRightThenDown ? height : 1;
    int innerHeight = firstRightThenDown ? 1 : height;
    for (int h = 0; h < outerHeight; h++)
    {
        for (currOffset[0] = roiBase[0]; currOffset[0] < targetOffset[0]; currOffset[0] += elementsInCL)
        {
            if (h == 0)
            {
                for (int i = 1; i < Gaudi2::Mme::c_mme_max_tensor_dims; i++)
                {
                    currOffset[i] = roiBase[i] + startOffsets[i - 1];
                }
            }

            bool lastGemmCol = (currOffset[0] + elementsInCL >= targetOffset[0]);

            for (int spStep = 0; spStep < innerHeight; spStep++)
            {
                int64_t denseOffset;
                int64_t denseTarget;
                if (fp->lower)
                {
                    denseOffset = currOffset[0] + currOffset[1];
                    denseTarget = mme_min(currOffset[1] + roiBase[0] + fcd, fp->tensorDesc.validElements[1]);
                    denseTarget = mme_min(denseTarget, currOffset[1] + fp->tensorDesc.validElements[0]);
                }
                else
                {
                    denseOffset = currOffset[0];
                    denseTarget = mme_min(targetOffset[0], fp->tensorDesc.validElements[0]);
                }

                bool pad = false;
                for (int spDim = fp->lower ? 2 : 1; spDim < Gaudi2::Mme::c_mme_max_tensor_dims; spDim++)
                {
                    if ((currOffset[spDim] < 0) || (currOffset[spDim] >= fp->tensorDesc.validElements[spDim]))
                    {
                        pad = true;
                        break;
                    }
                }

                int64_t denseEndOffset = denseOffset + elementsInFullCL;
                int64_t padMsb = denseEndOffset > denseTarget ? denseEndOffset - denseTarget : 0;
                int64_t padLsb = denseOffset < 0 ? -denseOffset : 0;

                if (pad || fp->signalOnly || (padLsb >= elementsInFullCL) || (padMsb >= elementsInFullCL))
                {
                    padMsb = elementsInFullCL;
                    padLsb = 0;
                }

                const bool noData = (loopEndMask & fp->fcdLoopMask) ? fp->noData : 0;
                if (!fp->signalOnly && !noData && !pad &&
                    (padLsb < elementsInFullCL) && (padMsb < elementsInFullCL))
                {
                    uint64_t accumulatedOffset =
                        std::accumulate(std::begin(currOffset), std::end(currOffset), uint64_t {0})
                        << fp->logElementSize;

                    uint64_t addr = fp->baseAddr + accumulatedOffset;
                    uint64_t lpadInBytes = padLsb << fp->logElementSize;
                    uint64_t mpadInBytes = padMsb << fp->logElementSize;
                    if (Mme::c_cl_size > lpadInBytes + mpadInBytes)
                    {
                        ranges->addSegment(addr + lpadInBytes, Mme::c_cl_size - lpadInBytes - mpadInBytes, 0);
                    }
                }

                int inc = firstRightThenDown ? lastGemmCol : 1;
                for (int spDim = 1; inc > 0; spDim++)
                {
                    currOffset[spDim] += (fp->tensorDesc.spatialStrides[spDim - 1] * inc);
                    inc = 0;
                    if (spDim < Gaudi2::Mme::c_mme_max_tensor_dims - 1)
                    {
                        if (currOffset[spDim] >= targetOffset[spDim])
                        {
                            currOffset[spDim] -= fp->tensorDesc.roiSize[spDim];
                            inc++;
                        }
                    }
                }
            }  // innerHeight
        }  // fcd
    }  // outerHeight

    if (advance)
    {
        for (int i = 1; i < Gaudi2::Mme::c_mme_max_tensor_dims; i++)
        {
            nextStartOffsets[i - 1] = currOffset[i] - roiBase[i];
        }
    }
}

void genAddresses(const Mme::Desc* desc,
                  const EMmeOperandIdx aguId,
                  const EMmeOpType opType,
                  const bool master,
                  std::vector<AguRanges>* ranges)
{
    unsigned signalCtr = 0;
    bool advance;
    Mme::MmeAguCoreDesc aguDesc;
    bool enable;
    unsigned chSize = Mme::c_cl_size / 2;
    bool swapBaseAndOffsetTensor;
    unsigned aguMask = 1 << aguId;
    bool aguReadsA, aguReadsB;

    FixedParams fp = {};

    fp.signalEn = desc->syncObject.signalEn0 || desc->syncObject.signalEn1;
    fp.signalMask = desc->syncObject.signalMask0;
    if (desc->syncObject.signalEn1)
        MME_ASSERT(desc->syncObject.signalMask0 == desc->syncObject.signalMask1,
                  "signal mask should be the same for both outputs");

    if (!isOutputAgu(aguId))  // aguIn
    {
        aguReadsA = desc->header.aguReadsA & aguMask;
        aguReadsB = desc->header.aguReadsB & aguMask;
        fp.logElementSize = getLogElementSize(ConvertDataTypeFromGaudi2((Mme::EMmeDataType) desc->header.dataTypeIn));
        if (aguReadsA)
        {
            enable = master ? desc->brains.aguA.masterEn : desc->brains.aguA.slaveEn;
            fp.loopMask = desc->brains.aguA.loopMask;
            fp.spatialSizeMinus1 = desc->spatialSizeMinus1A;

            swapBaseAndOffsetTensor = desc->header.swapBaseAndOffsetA;
            advance = desc->header.advanceA;
            fp.baseAddr = 0;  // Intentionally set to 0
            fp.tensorDesc = desc->tensorA;
            fp.lower = desc->header.lowerA;

            fp.transEn = desc->header.transA;
            fp.partialHeightLoopMask = fp.transEn ? desc->header.partialHeightLoopA : 0;
            fp.fcdLoopMask = fp.transEn ? 0 : desc->header.partialHeightLoopA;
        }
        else if (aguReadsB)  // aguReadsB
        {
            enable = master ? desc->brains.aguB.masterEn : desc->brains.aguB.slaveEn;
            fp.loopMask = desc->brains.aguB.loopMask;
            fp.spatialSizeMinus1 = desc->spatialSizeMinus1B;

            swapBaseAndOffsetTensor = desc->header.swapBaseAndOffsetB;
            advance = desc->header.advanceB;
            fp.baseAddr = 0;  // Intentionally set to 0
            fp.tensorDesc = desc->tensorB;
            fp.lower = desc->header.lowerB;

            fp.transEn = desc->header.transB;
            fp.partialHeightLoopMask = fp.transEn ? desc->header.partialHeightLoopB : 0;
            fp.fcdLoopMask = fp.transEn ? 0 : desc->header.partialHeightLoopB;
        }
        else
        {
            return;  // Early release
        }

        if (fp.transEn)
        {
            fp.fcd = fp.tensorDesc.roiSize[0];
            fp.fcdLast = fp.fcd;
            fp.height = chSize;
            fp.heightLast = (fp.spatialSizeMinus1 + 1) % fp.height;
            fp.heightLast = fp.heightLast ? fp.heightLast : fp.height;
        }
        else
        {
            fp.height = fp.spatialSizeMinus1 + 1;
            fp.fp32nonTransWalk = fp.logElementSize == getLogElementSize(e_type_fp32);
            fp.fcd =
                fp.fp32nonTransWalk
                    ? chSize
                    : getNumOfElementsInChannel(ConvertDataTypeFromGaudi2((Mme::EMmeDataType) desc->header.dataTypeIn));
            fp.fcdLast = fp.tensorDesc.roiSize[0] % fp.fcd;
            fp.fcdLast = fp.fcdLast ? fp.fcdLast : fp.fcd;
            fp.heightLast = (fp.spatialSizeMinus1 + 1) % chSize;
            fp.heightLast = fp.heightLast ? fp.heightLast : chSize;
        }

        aguDesc = master ? desc->aguIn[aguId][Mme::MME_CORE_MASTER] : desc->aguIn[aguId][Mme::MME_CORE_SLAVE];
    }
    else  // aguOut
    {
        bool brainEn;
        unsigned augOutId = aguId - e_mme_agu_cout0_idx;
        if (aguId == e_mme_agu_cout0_idx)  // aguCout0
        {
            brainEn = (master ? desc->brains.aguOut0.masterEn : desc->brains.aguOut0.slaveEn);
            fp.loopMask = desc->brains.aguOut0.loopMask;
        }
        else  // aguCout1
        {
            brainEn = (master ? desc->brains.aguOut1.masterEn : desc->brains.aguOut1.slaveEn);
            fp.loopMask = desc->brains.aguOut1.loopMask;
        }
        enable = brainEn && !desc->brains.noRollup &&
                 ((desc->header.storeEn0 || desc->header.storeEn1) ||
                  (desc->syncObject.signalEn0 || desc->syncObject.signalEn1));
        advance = desc->header.advanceC;
        aguDesc = master ? desc->aguOut[augOutId][Mme::MME_CORE_MASTER] : desc->aguOut[augOutId][Mme::MME_CORE_SLAVE];
        swapBaseAndOffsetTensor = desc->header.swapBaseAndOffsetOut;
        fp.logElementSize = getLogElementSize(ConvertDataTypeFromGaudi2((Mme::EMmeDataType) desc->header.dataTypeOut));
        if (desc->header.storeEn0)
        {
            fp.baseAddr = 0;  // Intentionally set to 0
        }
        else if (desc->header.storeEn1)
        {
            fp.baseAddr = 0;  // Intentionally set to 0
        }
        fp.tensorDesc = desc->tensorCOut;
        fp.lower = false;

        fp.spatialSizeMinus1 = desc->spatialSizeMinus1Cout;

        fp.partialHeightLoopMask = desc->header.partialHeightLoopA;
        fp.fcdLoopMask = desc->header.partialHeightLoopB;
        fp.height = c_matrix_height;
        fp.heightLast = fp.spatialSizeMinus1 + 1;
        fp.fcd = c_matrix_width / 2;
        if (desc->header.hx2)
        {
            fp.fcdLast = fp.tensorDesc.roiSize[0];
        }
        else
        {
            if (aguId == e_mme_agu_cout0_idx)
            {
                fp.fcdLast = mme_min(fp.tensorDesc.roiSize[0], c_matrix_width / 2);
            }
            else
            {
                if (fp.tensorDesc.roiSize[0] > c_matrix_width / 2)
                {
                    fp.fcdLast = fp.tensorDesc.roiSize[0] - c_matrix_width / 2;
                }
                else
                {
                    fp.fcdLast = fp.tensorDesc.roiSize[0];  // align to RTL
                    fp.noData = true;
                }
            }
        }

        fp.syncObject = desc->syncObject;
        if (!master)
        {
            getSlaveSyncObject(desc, &fp.syncObject);
        }

        // Signal Only - AGU should send only 1 dummy-address to WBC
        if ((!(desc->header.storeEn0 || desc->header.storeEn1)))
        {
            fp.signalOnly = true;
            fp.noData = false;
            fp.height = 1;
            fp.heightLast = 1;
            fp.fcd = 1;
            fp.fcdLast = 1;
            advance = false;
        }
    }

    if (enable)
    {
        if (swapBaseAndOffsetTensor)
        {
            swapBaseAndOffset(&aguDesc, &fp.tensorDesc);
        }
        std::vector<unsigned> gemms = genPCLoops(desc, fp.loopMask);
        unsigned associatedDim[Gaudi2::Mme::c_mme_max_conv_dims + 2];
        getAssociatedDims(associatedDim, aguId, desc);

        int64_t startOffsets[Gaudi2::Mme::c_mme_max_tensor_dims - 1];
        int64_t nextStartOffsets[Gaudi2::Mme::c_mme_max_tensor_dims - 1];
        int64_t roiBase[Gaudi2::Mme::c_mme_max_tensor_dims];
        for (int i = 0; i < Gaudi2::Mme::c_mme_max_tensor_dims; i++)
        {
            roiBase[i] = aguDesc.roiBaseOffset[i];
        }

        // first = which loop is on its first iteration (init value - all loops)
        unsigned first = (1 << (Gaudi2::Mme::c_mme_max_conv_dims + 2)) - 1;
        // reset = which loop wraps around (init value - all loops)
        unsigned reset = (1 << (Gaudi2::Mme::c_mme_max_conv_dims + 2)) - 1;

        unsigned totalNumSignals = 0;
        unsigned signalsPerLoop =
            calcNumSignalsPerLoop(desc, fp.loopMask, fp.signalMask, fp.signalEn, &totalNumSignals);

        unsigned descSignalBase = 0;
        unsigned outputsNr = calcOutputsNr(*desc, gemms);
        unsigned gemmsOutputsToCalc = getGemmsSkips(aguMask, opType, desc, outputsNr, &descSignalBase);
        MME_ASSERT(gemmsOutputsToCalc <= gemms.size(), "cannot have more outputs than gemm size");

        unsigned outputIdx = 0;

        for (auto& last : gemms)
        {
            if (outputIdx == gemmsOutputsToCalc)
            {
                break;
            }

            // Set roiBase:
            // for each loop that reset - reset roiBase[dim]
            // on loop that doesn't reset add loopStride[dim]
            for (int i = 0; i < Gaudi2::Mme::c_mme_max_conv_dims + 2; i++)
            {
                unsigned dim = associatedDim[i];
                if (reset & (1 << i))
                {
                    if (dim < Gaudi2::Mme::c_mme_max_tensor_dims)
                    {
                        roiBase[dim] = aguDesc.roiBaseOffset[dim];
                    }
                }
                else
                {
                    if (dim < Gaudi2::Mme::c_mme_max_tensor_dims)
                    {
                        roiBase[dim] += fp.tensorDesc.loopStride[dim];
                    }
                    first &= ~(1 << i);
                    break;
                }
            }

            // Advance if Spatial loop resets
            bool advanceStartOffset = advance && ((reset & ((1 << Gaudi2::Mme::c_mme_max_conv_dims) - 1)) ==
                                                  ((1 << Gaudi2::Mme::c_mme_max_conv_dims) - 1));

            if (reset & (1 << Gaudi2::Mme::c_mme_max_conv_dims))
            {
                for (int i = 0; i < Gaudi2::Mme::c_mme_max_tensor_dims - 1; i++)
                {
                    startOffsets[i] = fp.tensorDesc.startOffset[i];
                    nextStartOffsets[i] = fp.tensorDesc.startOffset[i];
                }
            }
            else if (advanceStartOffset)
            {
                std::swap(startOffsets, nextStartOffsets);
            }

            MME_ASSERT(ranges->size() > signalCtr + descSignalBase, "num of ranges is smaller than num of signals ");

            genGemmAddresses(&fp,
                             aguId,
                             advanceStartOffset,
                             startOffsets,
                             nextStartOffsets,
                             roiBase,
                             first,
                             last,
                             &(*ranges)[signalCtr + descSignalBase]);

            if (fp.signalEn && ((last & fp.signalMask) == fp.signalMask))
            {
                signalCtr += signalsPerLoop;
            }

            reset = last & ~(last + 1);
            first |= reset;

            if (isOutputLoop(*desc, last))
            {
                ++outputIdx;
            }
        }
    }
}
}  // namespace Gaudi2
