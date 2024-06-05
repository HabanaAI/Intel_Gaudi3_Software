#include "dma_cost_model.h"

#include "dma_node.h"
#include "graph_compiler/passes/split_to_physical_rois.h"
#include "hal_reader/hal_reader.h"
#include "roi_splitter.h"

constexpr auto GB_TO_BYTES = 1 << 30;
constexpr auto SEC_TO_USEC = 1000000;

static double max(double x1, double x2, double x3, double x4)
{
    return std::max(std::max(x1, x2), std::max(x3, x4));
}

DmaCostModel::DmaCostModel(const HalReader& hal, HabanaGraph* graph) : m_graph(graph)
{
    m_hbmBw  = hal.getHbmBwGBps();
    m_sramBw = hal.getSramBwGBps();
    m_dmaBw  = hal.getDmaBwGBps();
    m_numDmaEngines      = hal.getNumInternalDmaEngines();
    m_clSize = hal.getCacheLineSizeInBytes();
    m_dmaMinimalOverhead = hal.getDmaMinimalOverhead();
}

uint64_t DmaCostModel::roundToClMultiplication(uint64_t size) const
{
    uint64_t remainder = size % m_clSize;
    return (remainder == 0) ? size : (size - remainder + m_clSize);
}

uint64_t DmaCostModel::getSizeInFullCLs(const DMANode& node, const TensorPtr& tensor)
{
    uint64_t totalSizeInBytes = tensor->getTotalSizeInBytes();

    if (node.isLinearDma() || node.isTranspose())
    {
        return roundToClMultiplication(totalSizeInBytes);
    }

    uint64_t dim0InBytes   = tensor->getSizeInBytes(0);
    uint64_t dim0InFullCLs = roundToClMultiplication(dim0InBytes);

    return totalSizeInBytes / dim0InBytes * dim0InFullCLs;
}

void DmaCostModel::calculateDescsRemainder(DMANode& node, uint64_t& maxRemainderSize, uint64_t& totalRemainderSize)
{
    unsigned int elementSize = node.getOutput(0)->getElementSizeInBytes();
    unsigned int origNumDesc = node.getPhysicalRois() ? node.getPhysicalRois()->size() : 0;

    unsigned int       numDesc;
    std::list<NodeROI> physicalRois;

    std::shared_ptr<Node> nodeShared = node.shared_from_this();
    if (origNumDesc == 0 && m_graph)
    {
        ROISplitter splitter;
        NodeROI     baseRoi = nodeShared->generateRoi();
        node.setParallelLevel(m_numDmaEngines);
        std::list<NodeROI> logicalRois = splitter.splitDMA(nodeShared, *m_graph, baseRoi);

        unsigned currPipelineLevel = 0;

        for (NodeROI& roi : logicalRois)
        {
            roi.pipelineLevel = currPipelineLevel;
            ++currPipelineLevel;
        }

        splitToPhysicalROIsForNode(*m_graph, nodeShared, &logicalRois, physicalRois);
        numDesc = physicalRois.size();
    }
    else
    {
        numDesc = origNumDesc;
    }

    unsigned remainder = numDesc % m_numDmaEngines;
    maxRemainderSize   = 0;
    totalRemainderSize = 0;

    if (remainder == 0)
    {
        return;
    }

    auto itRoi = origNumDesc == 0 ? physicalRois.end() : node.getPhysicalRois()->end();
    for (int i = 0; i < remainder; i++)
    {
        --itRoi;
        maxRemainderSize = std::max(multiplyElements(itRoi->size, itRoi->size + 25) * elementSize, maxRemainderSize);
        totalRemainderSize += multiplyElements(itRoi->size, itRoi->size + 25) * elementSize;
    }
}

double DmaCostModel::calculateProjectedDuration(DMANode& node,
                                                double   sramSize,
                                                double   hbmSize,
                                                double   inputTensorSize,
                                                double   outputTensorSize)
{
    uint64_t maxSizeRemainder;
    uint64_t totalSizeRemainder;
    calculateDescsRemainder(node, maxSizeRemainder, totalSizeRemainder);

    double durationAllEngines = 0;
    double totalDuration      = 0;

    bool     bothTensorsInSram  = (sramSize > outputTensorSize) ? true : false;
    bool     bothTensorsInHbm   = (hbmSize > outputTensorSize) ? true : false;
    uint64_t sramTotalRemainder = !sramSize ? 0 : (bothTensorsInSram ? totalSizeRemainder * 2 : totalSizeRemainder);
    uint64_t hbmTotalRemainder  = !hbmSize ? 0 : (bothTensorsInHbm ? totalSizeRemainder * 2 : totalSizeRemainder);

    double sramSizeOnAllEngines = (sramSize - sramTotalRemainder > 0) ? (sramSize - sramTotalRemainder) : 0;
    double hbmSizeOnAllEngines  = (hbmSize - hbmTotalRemainder > 0) ? (hbmSize - hbmTotalRemainder) : 0;
    double inputSizeOnAllEngines =
        (inputTensorSize - totalSizeRemainder > 0) ? (inputTensorSize - totalSizeRemainder) : 0;
    double outputSizeOnAllEngines =
        (outputTensorSize - totalSizeRemainder > 0) ? (outputTensorSize - totalSizeRemainder) : 0;

    if (sramSizeOnAllEngines > 0 || hbmSizeOnAllEngines > 0 || inputSizeOnAllEngines > 0 || outputSizeOnAllEngines > 0)
    {
        durationAllEngines = calculateDurationByNumEngines(sramSizeOnAllEngines,
                                                           hbmSizeOnAllEngines,
                                                           inputSizeOnAllEngines,
                                                           outputSizeOnAllEngines,
                                                           m_numDmaEngines);
    }

    double inputSizeMaxRemainder  = inputTensorSize ? maxSizeRemainder : 0;
    double outputSizeMaxRemainder = outputTensorSize ? maxSizeRemainder : 0;

    totalDuration = durationAllEngines + calculateDurationByNumEngines(sramTotalRemainder,
                                                                       hbmTotalRemainder,
                                                                       inputSizeMaxRemainder,
                                                                       outputSizeMaxRemainder,
                                                                       1);

    return totalDuration;
}

void DmaCostModel::calculateSizes(const DMANode& node,
                                  double&        sramSize,
                                  double&        hbmSize,
                                  double&        inputTensorSize,
                                  double&        outputTensorSize)
{
    TensorPtr input  = nullptr;
    TensorPtr output = node.getOutput(0);

    inputTensorSize  = 0;
    outputTensorSize = getSizeInFullCLs(node, output);

    if (!node.isMemset())
    {
        input = node.getInput(0);
    }

    hbmSize  = 0;  // in Bytes
    sramSize = 0;  // in Bytes

    if (input)
    {
        inputTensorSize = node.isBroadcast() ? outputTensorSize : getSizeInFullCLs(node, input);

        if (input->getTensorAnnotation().memory.location == TENSOR_IN_SRAM)
        {
            sramSize += inputTensorSize;
        }
        else
        {
            hbmSize += inputTensorSize;
        }
    }

    if (output->getTensorAnnotation().memory.location == TENSOR_IN_SRAM)
    {
        sramSize += outputTensorSize;
    }
    else
    {
        hbmSize += outputTensorSize;
    }
}

double DmaCostModel::calculateDurationByNumEngines(double   sramSize,
                                                   double   hbmSize,
                                                   double   inputTensorSize,
                                                   double   outputTensorSize,
                                                   unsigned numEngines)
{
    unsigned dmaBw = ((double)m_dmaBw / m_numDmaEngines) * numEngines;

    return max((double)sramSize / m_sramBw, hbmSize / m_hbmBw, inputTensorSize / dmaBw, outputTensorSize / dmaBw) /
           GB_TO_BYTES * SEC_TO_USEC;
}

DmaCost DmaCostModel::getCostModelResult(DMANode& node)
{
    DmaCost result;

    double prevDurationDescsSplit = node.getNodeAnnotation().dmaCost.durationUnderDescsSplitInUsec;
    double sramSize;
    double hbmSize;
    double inputTensorSize;
    double outputTensorSize;

    calculateSizes(node, sramSize, hbmSize, inputTensorSize, outputTensorSize);
    HB_ASSERT(m_sramBw != 0 && m_hbmBw != 0 && m_dmaBw != 0, "SRAM BW / HBM BW / DMA BW is not initialized");

    result.durationInUsec =
        calculateDurationByNumEngines(sramSize, hbmSize, inputTensorSize, outputTensorSize, m_numDmaEngines);

    if (!node.isTranspose() && node.isLinearDma())
    {
        double overheadFactor = node.isMemset() ? 0.5 : 1;
        result.durationUnderDescsSplitInUsec =
            calculateProjectedDuration(node, sramSize, hbmSize, inputTensorSize, outputTensorSize) +
            m_dmaMinimalOverhead * overheadFactor;
    }

    LOG_DEBUG(
        GC,
        "DMA Cost Model: node={}, expectedDuration = {} usec, projectedDuration (with split to descriptors) = {} usec",
        node.getNodeName(),
        result.durationInUsec,
        result.durationUnderDescsSplitInUsec);

    if (result.durationUnderDescsSplitInUsec && prevDurationDescsSplit)
    {
        // Calculate diff between cost model results: when desctipors split is projected vs when split is already known
        double diff =
            ((prevDurationDescsSplit - result.durationUnderDescsSplitInUsec) / result.durationUnderDescsSplitInUsec) *
            100;

        // Add log message if |diff| > 10%
        if (diff > 10 || diff < -10)
        {
            LOG_INFO(GC,
                     "DMA Cost Model - compare results of expected duration under descriptors split: node={}, current "
                     "= {} usec, previous = {} usec, diff(%) = {}",
                     node.getNodeName(),
                     result.durationUnderDescsSplitInUsec,
                     prevDurationDescsSplit,
                     diff);
        }
    }

    return result;
}