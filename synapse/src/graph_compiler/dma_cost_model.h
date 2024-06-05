#pragma once

#include "types.h"

class HalReader;
class DMANode;
class HabanaGraph;

struct DmaCost
{
    double       durationInUsec                = 0;
    double       durationUnderDescsSplitInUsec = 0;  // takes under account the split into descriptors
    unsigned int cycles                 = 0;
};

class DmaCostModel
{
public:
    DmaCostModel(const HalReader& hal, HabanaGraph* graph = nullptr);
    DmaCost  getCostModelResult(DMANode& node);
    uint64_t getSizeInFullCLs(const DMANode& node, const TensorPtr& tensor);

private:
    void   calculateDescsRemainder(DMANode& node, uint64_t& maxSizeRemainder, uint64_t& totalSizeRemainder);
    double calculateProjectedDuration(DMANode& node,
                                      double   sramSize,
                                      double   hbmSize,
                                      double   inputTensorSize,
                                      double   outputTensorSize);
    void   calculateSizes(const DMANode& node,
                          double&        sramSize,
                          double&        hbmSize,
                          double&        inputTensorSize,
                          double&        outputTensorSize);
    double calculateDurationByNumEngines(double   sramSize,
                                         double   hbmSize,
                                         double   inputTensorSize,
                                         double   outputTensorSize,
                                         unsigned numEngines);

    uint64_t roundToClMultiplication(uint64_t size) const;

    unsigned int m_hbmBw;   // [GB/sec]
    unsigned int m_sramBw;  // [GB/sec]
    unsigned int m_dmaBw;   // [GB/sec]
    unsigned int m_clSize;  // [Bytes]
    unsigned int m_numDmaEngines;
    double       m_dmaMinimalOverhead;
    HabanaGraph* m_graph;
};