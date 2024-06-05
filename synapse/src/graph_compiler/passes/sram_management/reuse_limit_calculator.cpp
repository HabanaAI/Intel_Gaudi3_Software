#include "reuse_limit_calculator.h"

#include "habana_global_conf.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "node.h"
#include "slicing_brain.h"

#include <types.h>

ReuseLimitCalculator::ReuseLimitCalculator(const HalReader& halReader, const pNode& node)
: m_node(node),
  m_dimCtrl(node),
  c_mmeVecSizeElement(halReader.getMmeVectorSize() / node->getOutput(0)->getElementSizeInBytes())
{}

uint64_t ReuseLimitCalculator::getLimit() const
{
    // The limit would be set where there is "enough" reuse - i.e. the ratio between processing and traffic is at least
    // some factor.

    if (GCFG_SRAM_SLICER_REUSE_LIMIT_FACTOR.value() < 1.0f)
    {
        return SlicingBrain::knobs.maxWideSliceSizeFactor_nonCommon2D;
    }

    uint64_t minReuseLimit = minimalReuseLimit();

    // Use the bigger side so in case the limit is the whole side, we don't limit too much
    uint64_t heightSize = aggregateDimSizes(m_node->getOutput(TENSOR_OFM), m_dimCtrl.heightOutput());
    uint64_t widthSize = aggregateDimSizes(m_node->getOutput(TENSOR_OFM), m_dimCtrl.widthOutput());
    uint64_t limit = std::max(heightSize, widthSize);

    // While the minReuseLimit is as low value as this calculator would return, if the height or width
    // are smaller than it, the Pr/Tr ratio need to reflect it.
    uint64_t minimalNarrow = std::min(minReuseLimit, std::min(heightSize, widthSize));

    // Check the next ratio when using one side as the minimal size. This will allow the solvers to shape the slices
    // in narrower rectangles if they so choose.
    while (limit > minReuseLimit &&
           getPrTrRatio(limit / 2, minimalNarrow) >= GCFG_SRAM_SLICER_REUSE_LIMIT_FACTOR.value())
    {
        if (((limit / 2) < minReuseLimit) && ((limit % c_mmeVecSizeElement) == 0))
        {
            // Break when current limit is align to MME vector size and going to be decreased below minReuseLimit.
            // This will allow the solver to create more strategies with bigger slices and better MME utilization.
            break;
        }
        limit /= 2;
    }

    return std::max(limit, minReuseLimit);
}

// Don't limit too much...
uint64_t ReuseLimitCalculator::minimalReuseLimit() const
{
    // MME geometry can have up to 4 MMEs in a row. Allow at least 2 geometries.
    return 8 * c_mmeVecSizeElement;
}

uint64_t ReuseLimitCalculator::aggregateDimSizes(const pTensor& operand, const DimVector& dims)
{
    uint64_t size = 1;
    for (auto dim : dims)
    {
        size *= operand->getSizeInElements(dim);
    }
    return size;
}

double ReuseLimitCalculator::getPrTrRatio(uint64_t height, uint64_t width) const
{
    uint64_t cdSize = aggregateDimSizes(m_node->getInput(0), m_dimCtrl.commonDimOperandA());

    double processingTime = getProcessingTime(height, width, cdSize);
    double trafficTime    = getTrafficTime(height, width, cdSize);

    return processingTime / trafficTime;
}

// Processing time estimation
double ReuseLimitCalculator::getProcessingTime(uint64_t height, uint64_t width, uint64_t cdSize) const
{
    // Number of cycles is the number of operations (multiplications) divided by the number of
    // multiplications that the MME is doing in a single cycle, times the 4 MMEs.
    double cycles = cdSize * height * width / (4. * c_mmeVecSizeElement * c_mmeVecSizeElement);

    // Return the ideal processing time (100% utilization): #cycles/frequency
    return cycles / SlicingBrain::knobs.freqGHz;
}

double ReuseLimitCalculator::getTrafficTime(uint64_t height, uint64_t width, uint64_t cdSize) const
{
    // Since it's assumed that the width, height and cdSize are aggregated, the size of each
    // operand is the multiplication of it's dimensions (which is the no. of elements) times
    // element size.
    uint64_t traffic =
            height * cdSize * m_node->getInput(TENSOR_IFM)->getElementSizeInBytes() +   // operand A +
            width * cdSize * m_node->getInput(TENSOR_WEIGHT)->getElementSizeInBytes() + // operand B +
            height * width * m_node->getOutput(TENSOR_OFM)->getElementSizeInBytes();    // output

    // Return ideal traffic (dense, no cache line alignment issues):
    // Size in bytes of 2 inputs and the output divided by the HBM BW.
    return traffic / SlicingBrain::knobs.hbmAvailableBWGBps;
}
