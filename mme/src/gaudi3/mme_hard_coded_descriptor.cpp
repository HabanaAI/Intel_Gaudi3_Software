#include <stdio.h>
#include <cstring>
#include "general_utils.h"

#include "include/mme_common/mme_common_enum.h"
#include "include/gaudi3/mme_descriptor_generator.h"

using namespace MmeCommon;

namespace gaudi3
{
void MmeConvDescriptorGenerator::createHardcodedABDesc(Mme::Desc* desc, unsigned mmeIdx)
{
    memset(desc, 0, sizeof(Mme::Desc));

    // Set header
    desc->header.transA = 1;
    desc->header.sbTransA = 1;
    desc->header.transB = 0;

    desc->header.advanceA = 1;
    desc->header.advanceC = 1;
    desc->header.shuffleA = Mme::e_mme_shuffle_2ports;
    desc->header.dataTypeIn = Mme::EMmeDataType::e_mme_dt_bf16;
    desc->header.dataTypeOut = Mme::EMmeDataType::e_mme_dt_bf16;

    //  we need to turn this on if ports are interleaved
    //  since A is non transposed C ports are not interleaved.
    //  though non of this really matters because we have only 1 dimension and no movement..
    desc->header.swapBaseAndOffsetA = 1;
    desc->header.swapBaseAndOffsetB = 1;
    desc->header.swapBaseAndOffsetOut = 0;

    desc->header.storeEn0 = 1;
    desc->header.doubleAccums = 1;

    desc->header.partialHeightLoopA = getLoopFromLoopMask(e_mme_conv_loop_0);  // walk pattern - fkc
    desc->header.partialHeightLoopB = getLoopFromLoopMask(e_mme_conv_loop_1);  // walk pattern - fkc

    // Brains
    // Brain A
    desc->brains.aguA.masterEn = 1;
    desc->brains.aguA.slaveEn = 1;
    desc->brains.aguA.loopMask = 0;
    // Brain B
    desc->brains.aguB.masterEn = 1;
    desc->brains.aguB.slaveEn = 1;
    desc->brains.aguB.loopMask = 0;
    // Brain EU
    desc->brains.eu.masterEn = 1;
    desc->brains.eu.slaveEn = 1;
    desc->brains.eu.loopMask = 0;
    // Brain ap
    desc->brains.ap.masterEn = 1;
    desc->brains.ap.slaveEn = 1;
    desc->brains.ap.loopMask = 0;
    // Brain aguOut
    desc->brains.aguOut.masterEn = 1;
    desc->brains.aguOut.slaveEn = 1;
    desc->brains.aguOut.loopMask = 0;

    unsigned geoHeight = 2;
    unsigned geoWidth = (Mme::MME_CORE_MASTERS_NR / geoHeight);

    const unsigned mmeWidth = 256;
    const unsigned mmeHeight = 256;
    const unsigned euHeight = mmeHeight / 2;
    const unsigned dimK = mmeWidth * geoWidth;
    const unsigned dimC = 256;
    const unsigned dimW = mmeHeight * geoHeight;  //  2 high
    const unsigned aPortsHeight = 4 * geoHeight;  //  4 ports per MME over all MMEs, interleaved
    const unsigned bPortsHeight = 4;  //  4 ports per MME, interleaved
    const unsigned cPortsNr = 2;  //  2 spatial ports per MME
    const unsigned cTotalSpatialPortsNr = cPortsNr * geoHeight;

    // Tensor Desc A
    desc->tensorA.validElements[0] = dimC;  // stride = 1
    desc->tensorA.validElements[1] = dimC * dimW;

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        desc->tensorA.validElements[dim] = desc->tensorA.validElements[dim - 1];
    }

    desc->tensorA.loopStride[0] = 0;
    desc->tensorA.loopStride[1] = dimC * dimW;
    desc->tensorA.loopStride[2] = desc->tensorA.loopStride[1];
    desc->tensorA.loopStride[3] = desc->tensorA.loopStride[1];
    desc->tensorA.loopStride[4] = desc->tensorA.loopStride[1];

    desc->tensorA.spatialStrides[0] = dimC * aPortsHeight;
    desc->tensorA.spatialStrides[1] = 1;
    desc->tensorA.spatialStrides[2] = 1;
    desc->tensorA.spatialStrides[3] = 1;

    desc->tensorA.roiSize[0] = dimC;
    desc->tensorA.roiSize[1] = desc->tensorA.validElements[1];
    desc->tensorA.roiSize[2] = 1;
    desc->tensorA.roiSize[3] = 1;

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorA.startOffset[dim] = 0;
    }

    // Tensor Desc B
    desc->tensorB.validElements[0] = dimK * 1;  // stride = 1
    desc->tensorB.validElements[1] = dimC * desc->tensorB.validElements[0];

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        desc->tensorB.validElements[dim] = desc->tensorB.validElements[dim - 1];
    }

    desc->tensorB.loopStride[0] = dimK;
    desc->tensorB.loopStride[1] = 0;
    desc->tensorB.loopStride[2] = dimC * dimK;
    desc->tensorB.loopStride[3] = desc->tensorB.loopStride[2];
    desc->tensorB.loopStride[4] = desc->tensorB.loopStride[2];

    desc->tensorB.spatialStrides[0] = dimK * bPortsHeight;
    desc->tensorB.spatialStrides[1] = 1;
    desc->tensorB.spatialStrides[2] = 1;
    desc->tensorB.spatialStrides[3] = 1;

    desc->tensorB.roiSize[0] = mmeWidth;
    desc->tensorB.roiSize[1] = desc->tensorB.validElements[1];
    desc->tensorB.roiSize[2] = 1;
    desc->tensorB.roiSize[3] = 1;

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorB.startOffset[dim] = 0;
    }

    // Tensor Desc Cout
    desc->tensorCOut.validElements[0] = dimK;
    desc->tensorCOut.validElements[1] = dimK * dimW;
    desc->tensorCOut.validElements[2] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.validElements[3] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.validElements[4] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.loopStride[0] = dimK;
    desc->tensorCOut.loopStride[1] = dimK * dimW;
    desc->tensorCOut.loopStride[2] = 0;
    desc->tensorCOut.loopStride[3] = 0;
    desc->tensorCOut.loopStride[3] = 0;

    desc->tensorCOut.roiSize[0] = mmeWidth;
    for (unsigned dim = 1; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorCOut.roiSize[dim] = desc->tensorCOut.validElements[dim];
    }

    // TODO fix
    desc->tensorCOut.spatialStrides[0] = dimK * cTotalSpatialPortsNr;
    desc->tensorCOut.spatialStrides[1] = 0;  //  shouldnt matter
    desc->tensorCOut.spatialStrides[2] = 0;  //  shouldnt matter
    desc->tensorCOut.spatialStrides[3] = 0;  //  shouldnt matter

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorCOut.startOffset[dim] = 0;
    }

    // Sync obj
    desc->syncObject.signalMask0 = EMmeLoopMask::e_mme_outer_loop;
    desc->syncObject.signalEn0 = 1;
    desc->syncObject.signalMask1 = EMmeLoopMask::e_mme_outer_loop;
    desc->syncObject.masterWaitForSlaveFence = 1;
    desc->syncObject.slaveSendFence2Master = 1;
    desc->syncObject.so0Val.soValue = 1;
    desc->syncObject.so0Val.soOp = 1;
    desc->syncObject.so1Val.soValue = 1;

    // AGU
    unsigned widthIdx = mmeIdx / 2;
    unsigned hightIdx = mmeIdx & 1;
    unsigned aCoreOffset = hightIdx;  //  interleaved each MME starts at the its index
    unsigned bCoreOffset = mmeWidth * widthIdx;

    // aguA
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[1] = (0 * geoHeight + aCoreOffset) * dimC;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[1] = (2 * geoHeight + aCoreOffset) * dimC;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[1] = (1 * geoHeight + aCoreOffset) * dimC;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[0] = 0;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[1] = (3 * geoHeight + aCoreOffset) * dimC;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1A = (dimW / aPortsHeight) - 1;

    // aguB
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[1] = 0 * dimK;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[1] = 1 * dimK;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[1] = 2 * dimK;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[1] = 3 * dimK;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1B = (dimC / bPortsHeight) - 1;

    // aguOut - since A is non transposed C ports are not interleaved.
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[0] = bCoreOffset;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[1] = (0 * geoHeight + aCoreOffset) * dimK;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[2] = 0;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[3] = 0;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[4] = 0;

    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[0] = bCoreOffset;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[1] = (1 * geoHeight + aCoreOffset) * dimK;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[2] = 0;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[3] = 0;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1Cout = (dimW / cTotalSpatialPortsNr) - 1;

    desc->conv.associatedDims[0].dimA = 0;  // DIM_C
    desc->conv.associatedDims[0].dimB = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[0].dimOut = 1;  // DIM_C
    desc->conv.associatedDims[1].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[1].dimB = 0;  // DIM_K
    desc->conv.associatedDims[1].dimOut = 0;  // DIM_K
    desc->conv.associatedDims[2].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimB = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimOut = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimA = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimB = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimOut = Mme::c_mme_max_tensor_dims;  // dont care

    // outer loop
    desc->outerLoop.associatedDims.dimA = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimB = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimOut = Mme::c_mme_max_tensor_dims;  // skf pattern

    // Rate Limiter
    desc->rateLimiter.aguA = 4;
    desc->rateLimiter.aguB = 4;
    desc->rateLimiter.aguOut = 4;

    // TODO: pcu - not needed
    // TODO: slaveSyncObject0Addr - not needed
    // TODO: powerLoop - not needed.
}

void MmeConvDescriptorGenerator::createHardcodedAtBDesc(Mme::Desc* desc, unsigned mmeIdx)
{
    memset(desc, 0, sizeof(Mme::Desc));

    // Set header
    desc->header.transA = 0;
    desc->header.transB = 0;

    desc->header.advanceA = 0;
    desc->header.advanceC = 1;

    desc->header.dataTypeIn  = Mme::EMmeDataType::e_mme_dt_bf16;
    desc->header.dataTypeOut = Mme::EMmeDataType::e_mme_dt_bf16;

    //  we need to turn this on if ports are interleaved
    //  since A is non transposed C ports are not interleaved.
    //  though non of this really matters because we have only 1 dimension and no movement..
    desc->header.swapBaseAndOffsetA   = 1;
    desc->header.swapBaseAndOffsetB   = 1;
    desc->header.swapBaseAndOffsetOut = 0;

    desc->header.storeEn0     = 1;
    desc->header.doubleAccums = 1;

    desc->header.partialHeightLoopA = getLoopFromLoopMask(e_mme_conv_loop_0);  // walk pattern - fkc
    desc->header.partialHeightLoopB = getLoopFromLoopMask(e_mme_conv_loop_1);  // walk pattern - fkc

    // Brains
    // Brain A
    desc->brains.aguA.masterEn   = 1;
    desc->brains.aguA.slaveEn    = 1;
    desc->brains.aguA.loopMask   = 0;
    // Brain B
    desc->brains.aguB.masterEn   = 1;
    desc->brains.aguB.slaveEn    = 1;
    desc->brains.aguB.loopMask   = 0;
    // Brain EU
    desc->brains.eu.masterEn     = 1;
    desc->brains.eu.slaveEn      = 1;
    desc->brains.eu.loopMask     = 0;
    // Brain ap
    desc->brains.ap.masterEn     = 1;
    desc->brains.ap.slaveEn      = 1;
    desc->brains.ap.loopMask     = 0;
    // Brain aguOut
    desc->brains.aguOut.masterEn = 1;
    desc->brains.aguOut.slaveEn  = 1;
    desc->brains.aguOut.loopMask = 0;

    unsigned geoHeight = 2;
    unsigned geoWidth = (Mme::MME_CORE_MASTERS_NR / geoHeight);

    const unsigned mmeWidth        = 256;
    const unsigned mmeHeight       = 256;
    const unsigned euHeight = mmeHeight / 2;
    const unsigned dimK            = mmeWidth * geoWidth;
    const unsigned dimC            = 8;
    const unsigned dimW            = mmeHeight * geoHeight;  //  2 high
    const unsigned aPortsHeight        = 4; //  4 ports per MME, interleaved
    const unsigned bPortsHeight        = 4; //  4 ports per MME, interleaved
    const unsigned cPortsNr = 2; //  2 spatial ports per MME
    const unsigned cTotalSpatialPortsNr = cPortsNr * geoHeight;

    // Tensor Desc A
    desc->tensorA.validElements[0] = dimW;  // stride = 1
    desc->tensorA.validElements[1] = dimC * dimW;

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        desc->tensorA.validElements[dim] = desc->tensorA.validElements[dim - 1];
    }

    desc->tensorA.loopStride[0] = dimW;
    desc->tensorA.loopStride[1] = 0;
    desc->tensorA.loopStride[2] = dimC * dimW;
    desc->tensorA.loopStride[3] = desc->tensorA.loopStride[2];
    desc->tensorA.loopStride[4] = desc->tensorA.loopStride[2];

    desc->tensorA.spatialStrides[0] = dimW * aPortsHeight;
    desc->tensorA.spatialStrides[1] = 1;
    desc->tensorA.spatialStrides[2] = 1;
    desc->tensorA.spatialStrides[3] = 1;

    desc->tensorA.roiSize[0] = mmeHeight;
    desc->tensorA.roiSize[1] = desc->tensorA.validElements[1];
    desc->tensorA.roiSize[2] = 1;
    desc->tensorA.roiSize[3] = 1;

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorA.startOffset[dim] = 0;
    }

    // Tensor Desc B
    desc->tensorB.validElements[0] = dimK * 1;  // stride = 1
    desc->tensorB.validElements[1] = dimC * desc->tensorB.validElements[0];

    for (unsigned dim = 2; dim < Mme::c_mme_max_tensor_dims; dim++)
    {
        desc->tensorB.validElements[dim] = desc->tensorB.validElements[dim - 1];
    }

    desc->tensorB.loopStride[0] = dimK;
    desc->tensorB.loopStride[1] = 0;
    desc->tensorB.loopStride[2] = dimC * dimK;
    desc->tensorB.loopStride[3] = desc->tensorB.loopStride[2];
    desc->tensorB.loopStride[4] = desc->tensorB.loopStride[2];

    desc->tensorB.spatialStrides[0] = dimK * bPortsHeight;
    desc->tensorB.spatialStrides[1] = 1;
    desc->tensorB.spatialStrides[2] = 1;
    desc->tensorB.spatialStrides[3] = 1;

    desc->tensorB.roiSize[0] = mmeWidth;
    desc->tensorB.roiSize[1] = desc->tensorB.validElements[1];
    desc->tensorB.roiSize[2] = 1;
    desc->tensorB.roiSize[3] = 1;

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorB.startOffset[dim] = 0;
    }

    // Tensor Desc Cout
    desc->tensorCOut.validElements[0] = dimK;
    desc->tensorCOut.validElements[1] = dimW * dimK;
    desc->tensorCOut.validElements[2] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.validElements[3] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.validElements[4] = desc->tensorCOut.validElements[1];
    desc->tensorCOut.loopStride[0]    = dimK;
    desc->tensorCOut.loopStride[1]    = 0;
    desc->tensorCOut.loopStride[2]    = 0;
    desc->tensorCOut.loopStride[3]    = 0;
    desc->tensorCOut.loopStride[3]    = 0;

    desc->tensorCOut.roiSize[0] = mmeWidth;
    for (unsigned dim = 1; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorCOut.roiSize[dim] = desc->tensorCOut.validElements[dim];
    }

    // TODO fix
    desc->tensorCOut.spatialStrides[0] = dimK;
    desc->tensorCOut.spatialStrides[1] = dimK * dimW;  //  shouldnt matter
    desc->tensorCOut.spatialStrides[2] = dimK * dimW;  //  shouldnt matter
    desc->tensorCOut.spatialStrides[3] = dimK * dimW;  //  shouldnt matter

    for (unsigned dim = 0; dim < Mme::c_mme_max_tensor_dims - 1; dim++)
    {
        desc->tensorCOut.startOffset[dim] = 0;
    }

    // Sync obj
    desc->syncObject.signalMask0             = EMmeLoopMask::e_mme_outer_loop;
    desc->syncObject.signalEn0               = 1;
    desc->syncObject.signalMask1             = EMmeLoopMask::e_mme_outer_loop;
    desc->syncObject.masterWaitForSlaveFence = 1;
    desc->syncObject.slaveSendFence2Master   = 1;
    desc->syncObject.so0Val.soValue          = 1;
    desc->syncObject.so0Val.soOp             = 1;
    desc->syncObject.so1Val.soValue          = 1;

    // AGU
    unsigned widthIdx    = mmeIdx / 2;
    unsigned hightIdx    = mmeIdx & 1;
    unsigned aCoreOffset = mmeHeight * hightIdx;
    unsigned bCoreOffset = mmeWidth * widthIdx;

    // aguA
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[0] = aCoreOffset;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[1] = 0 * dimW;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][0].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[0] = aCoreOffset;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[1] = 1 * dimW;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][1].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[0] = aCoreOffset;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[1] = 2 * dimW;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][0].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[0] = aCoreOffset;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[1] = 3 * dimW;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][1].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1A = (dimC / aPortsHeight) - 1;

    // aguB
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[1] = 0 * dimK;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][2].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[1] = 1 * dimK;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_MASTER][3].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[1] = 2 * dimK;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][2].roiBaseOffset[4] = 0;

    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[0] = bCoreOffset;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[1] = 3 * dimK;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[2] = 0;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[3] = 0;
    desc->aguIn[Mme::MME_SLAVE][3].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1B = (dimC / bPortsHeight) - 1;

    // aguOut - since A is non transposed C ports are not interleaved.
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[0] = bCoreOffset;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[1] = (0 + aCoreOffset) * dimK;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[2] = 0;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[3] = 0;
    desc->aguOut[Mme::MME_MASTER].roiBaseOffset[4] = 0;

    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[0] = bCoreOffset;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[1] = (euHeight + aCoreOffset) * dimK;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[2] = 0;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[3] = 0;
    desc->aguOut[Mme::MME_SLAVE].roiBaseOffset[4] = 0;

    desc->spatialSizeMinus1Cout = (dimW / cTotalSpatialPortsNr) - 1;

    desc->conv.associatedDims[0].dimA   = 0;  // DIM_C
    desc->conv.associatedDims[0].dimB   = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[0].dimOut = 1;  // DIM_C
    desc->conv.associatedDims[1].dimA   = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[1].dimB   = 0;  // DIM_K
    desc->conv.associatedDims[1].dimOut = 0;  // DIM_K
    desc->conv.associatedDims[2].dimA   = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimB   = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[2].dimOut = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimA   = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimB   = Mme::c_mme_max_tensor_dims;  // dont care
    desc->conv.associatedDims[3].dimOut = Mme::c_mme_max_tensor_dims;  // dont care

    // outer loop
    desc->outerLoop.associatedDims.dimA   = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimB   = Mme::c_mme_max_tensor_dims;  // skf pattern
    desc->outerLoop.associatedDims.dimOut = Mme::c_mme_max_tensor_dims;  // skf pattern

    // Rate Limiter
    desc->rateLimiter.aguA   = 4;
    desc->rateLimiter.aguB   = 4;
    desc->rateLimiter.aguOut = 4;

    // TODO: pcu - not needed
    // TODO: slaveSyncObject0Addr - not needed
    // TODO: powerLoop - not needed.
}
}  // namespace gaudi3
