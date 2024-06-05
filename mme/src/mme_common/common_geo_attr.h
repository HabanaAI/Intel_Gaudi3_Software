#ifndef MME__COMMON_GEO_ATTR_H
#define MME__COMMON_GEO_ATTR_H

#include <optional>
#include "llvm/small_vector.h"
#include "include/mme_common/mme_common_enum.h"

namespace MmeCommon
{
//  this struct describes how the different unit are splits. units are either ports or whole MMEs.
//  fcd units are contiguous, each unit works on the next chunk of FCD.
//  spatial units are either all interleaved or contiguous on the first spatial dimension
//  batch units are used for 2X optimizations.

//  struct describes a 3 dimensional array of units - the number of units for an operand
//  must be equal to fcd*spatial*batch.

//  so if for example fcd = 4 and sp = 2 describes an operand which is configured as follows:
//  +---+---+---+---+
//  | 4 | 5 | 6 | 7 |
//  +---+---+---+---+
//  | 0 | 1 | 2 | 3 |
//  +---+---+---+---+

class MmeHalReader;

struct GeometryGrid
{
    unsigned fcd = 1;
    unsigned spatial = 1;
    unsigned batch = 1;
    unsigned cd = 1;

    GeometryGrid idxToGrid(unsigned idx) const;  // given an index, return the corresponding position inside the grid
};

class CommonGeoAttr
{
private:
    bool transA;  //  is A operand transposed
    bool transB;  //  is B operand transposed
    bool m_primaryTensors = true;  //  on queries use primary tensros

public:
    CommonGeoAttr(const MmeLayerParams& params, const MmeHalReader& mmeHal);
    virtual ~CommonGeoAttr() = default;
    void setPrimaryTensors(bool usePrimaryTensor) { m_primaryTensors = usePrimaryTensor; };
    //  general HW defined constant
    virtual unsigned getAccHeight() const = 0;  //  HW ACC height in rows
    virtual unsigned getEuWidth() const = 0;  //  HW EU width in elements
    virtual unsigned getEuHeight() const = 0;  //  HW EU height in elements
    virtual unsigned getMmeWidth() const = 0;  //  MME width in elements, could be geometry dependant
    virtual unsigned getMmeHeight() const = 0;  //  MME height in elements, could be geometry dependant
    virtual unsigned getPortSize(EMmeInternalOperand operand) const = 0;  //  port size in elements
    virtual unsigned getTeHeight() const = 0;  //  number of rows in a single TE
    virtual unsigned getEuFacingPortSize(EMmeInternalOperand operand) const;
    virtual unsigned getCoresPerMmeNr() const = 0;  // number of cores in a single MME
    virtual bool doPortAdvanceSpatially(EMmeInternalOperand operand) const = 0;
    virtual unsigned getMmeNr() const  // number of MME units
    {
        return mmeGrid.fcd * mmeGrid.spatial * mmeGrid.batch * mmeGrid.cd;
    };
    virtual bool isGeometryPortConstrained() const { return false; };

    //  geometry specific sizes
    virtual unsigned getCoreConcurrency() const;  // number of gemms that will be calculated concurrently in a core
    virtual unsigned getMmeConcurrency() const;  // number of gemms that will be calculated concurrently by a single MME
    virtual unsigned
    getCoreCdConcurrency() const;  // number of partial outputs that will be calculated concurrently in a core
    virtual unsigned
    getMmeCdConcurrency() const;  // number of partial outputs that will be calculated concurrently by a single MME
    virtual unsigned getGeometryWidth() const;  //  overall MME width in elements, geometry dependant
    virtual unsigned getGeometryHeight() const;  // overall MME height in elements, geometry dependant
    virtual unsigned
    getGeometryConcurrency() const;  // number of gemms that will be calculated concurrently by the MMEs
    virtual unsigned getEffectiveBatchConcurrency() const { return getGeometryConcurrency(); };
    unsigned getGeometryCdConcurrency() const;  // Number of concurrent engines that split the CD
    virtual GeometryGrid
    coreIdxToGrid(unsigned coreIdx) const;  //  return the core placement in the grid according to the geometry
    virtual GeometryGrid
    mmeIdxToGrid(unsigned mmeIdx) const;  //  return the MME placement in the grid according to the geometry
    GeometryGrid getMmeGrid() const { return mmeGrid; };

    //  descriptor bits set by geometry
    virtual bool getBgemmBit() const { return false; };
    virtual bool getNonShareABit() const { return false; };
    virtual bool getDoubleAccumsBit() const;
    virtual bool getHx2Bit() const { return false; };
    virtual bool isHybridPattern() const { return false; };

    //  port amount and configurations
    unsigned getCoreSpatialPorts(EMmeInternalOperand operand) const;
    unsigned getCoreFcdPorts(EMmeInternalOperand operand) const;
    unsigned getCoreBatchPorts(EMmeInternalOperand operand) const;
    unsigned getCoreCdPorts(EMmeInternalOperand operand) const;
    // Sometimes coreGrid affects different operands in different ways, for example when the operand is transposed.
    // This function translates the core index into the effective placement of the core for a specific operand.
    GeometryGrid coreIdxToEffectiveGrid(EMmeInternalOperand operand, unsigned coreIdx) const;
    // This function returns a grid of the number of effective cores in each dim for a specific operand.
    GeometryGrid getEffectiveCoreGrid(EMmeInternalOperand operand) const;
    // This function return the number of input spatial ports that feed the same EU port
    virtual unsigned getCoreSpatialEuPort(EMmeInternalOperand operand) const { return 1; };
    unsigned getMmeSpatialPorts(EMmeInternalOperand operand) const;
    unsigned getMmeFcdPorts(EMmeInternalOperand operand) const;
    unsigned getMmeBatchPorts(EMmeInternalOperand operand) const;
    unsigned getMmePortsNr(EMmeInternalOperand operand) const;
    unsigned getChipFcdPorts(EMmeInternalOperand operand) const;
    unsigned getChipSpatialPorts(EMmeInternalOperand operand) const;
    unsigned getChipBatchPorts(EMmeInternalOperand operand) const;
    unsigned getChipPortsNr(EMmeInternalOperand operand) const;
    unsigned getCorePortsNr(EMmeInternalOperand operand) const;
    unsigned getMmeInterleavedSpatialPortsNr(EMmeInternalOperand operand) const;
    unsigned getInterleavedSpatialPortsNr(EMmeInternalOperand operand) const;
    bool isTransposed(EMmeInternalOperand operand) const;
    bool isPortStartOffset(EMmeInternalOperand operand) const;
    bool isSpatiallyInterleavedAcrossMmes(EMmeInternalOperand operand) const;
    bool isSpatiallyInterleavedAcrossCores(EMmeInternalOperand operand) const;
    bool isSpatiallyInterleavedInsideCore(EMmeInternalOperand operand) const;
    virtual bool isPortSharedBetweenCores(EMmeInternalOperand operand) const;
    bool supportsConcurrency() const;
    bool isOperandBroadcasted(EMmeInternalOperand operand, MmeDimsIndex dim) const;
    bool isOperandFullyBroadcasted(EMmeInternalOperand operand) const;
    MmeDimsIndex getConcurrentDim() const;
    unsigned getBatchDimsNr() const;
    MmeDimsIndex getLastSpatialDim(EMmeInternalOperand operand) const;
    unsigned getSpatialCoresInMmeNr() const;
    unsigned getSpatialMmeNr(EMmeInternalOperand operand) const;
    unsigned getFcdMmeNr(EMmeInternalOperand operand) const;
    void setATranspose(bool val) { transA = val; }
    void setBTranspose(bool val) { transB = val; }

    MmeDimsIndex getSpInterleavingDim(EMmeInternalOperand operand) const;

    virtual bool isPortValid(EMmeInternalOperand operand,
                             unsigned core,
                             unsigned cd,
                             unsigned batch,
                             unsigned fcd,
                             unsigned sp) const
    {
        return true;
    }
    virtual bool shouldSwapMasterAndSlave(EMmeInternalOperand operand) const { return false; }
    virtual bool isAsymPortConfigMode() const { return false; }
    // workaround indication specifically for Gaudi2, in some cases even though each MME core works on a different
    // batch, due to a HW limitation data from both batches will flow to both cores. the redundant rows and cols
    // will be dealt with by the AGU configuration
    virtual bool isMmeConcurrencyRoutingWorkAround() const { return false; }
    using EMmeInternalOperandVec = llvm_vecsmall::SmallVector<EMmeInternalOperand, e_mme_op_nr>;
    const EMmeInternalOperandVec& getOperands() const;

    MmeCommon::EMmeGeometry getGeometry() const {return m_params.strategy.geometry;}

protected:
    //  initialize Grids structs
    void resetGrids();
    void init();
    void setSpInterleavingDim();
    bool shouldInterleaveOnSecondSpatialDim();
    virtual void setGrids() = 0;
    virtual void setChipConcurrency();
    virtual void setMmeConcurrency();
    virtual void setBgemmConcurrency() = 0;
    virtual void setDedwConcurrency() = 0;
    virtual void setCdConcurrency() = 0;
    void validateParams();

    const MmeLayerParams m_params;
    const MmeHalReader& m_mmeHal;
    MmeDimsIndex m_interleavedCdDim = DIM_W;

    //  ports grid in a single core
    GeometryGrid aGrid;
    GeometryGrid bGrid;
    GeometryGrid cGrid;
    //  cores grid in a single MME
    //  Gaudi2 has 2 cores - master and slave.
    //  Gaudi3 has only 1 core which consists of both master and slave.
    GeometryGrid coreGrid;
    //  MME grid over the whole chip
    GeometryGrid mmeGrid;

private:
    virtual bool isDimConcurrencyOptimizationSupported() const;
    void setConcurrentDim();
    void setDefaultConcurrentDim() const;
    MmeDimsIndex calculateDefaultConcurrentDimForGemm() const;
    MmeDimsIndex calculateConcurrentDimForGemm() const;
    MmeDimsIndex calculateConcurrentDimForNonGemm() const;

    mutable std::optional<MmeDimsIndex> m_concurrentDimOpt;
};

using upCommonGeoAttr = std::unique_ptr<CommonGeoAttr>;
}  // namespace MmeCommon

#endif //MME__COMMON_GEO_ATTR_H
