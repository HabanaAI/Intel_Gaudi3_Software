#include <sstream>
#include "node.h"
#include "slicing_brain.h"
#include "slicing_utils.h"
#include "slicing_strategy.h"
#include "metrics_calculator.h"
#include "types.h"

SlicingStrategyPtr SlicingStrategy::createStrategy(const HalReader& halReader, const pNode& node)
{
    TensorVector inputs = node->getInputs();
    SlicingStrategyPtr s(
        new SlicingStrategy(halReader, std::make_shared<StrategySlicingData>(inputs, node->getOutput(0))));
    StrategySlicingData& data = s->getSlicingData();
    std::list<pSlicedOperand> slicedOperandInputs;
    for (int i = 0; i < data.bundleTensors.size(); i++)
    {
        slicedOperandInputs.push_back(data.bundleTensors[i]);
    }
    data.setOutputSliceBackwardMapping(
            TrivialSliceMapper::mapOutputToInputs(node, slicedOperandInputs, data.masterOperand));

    // Add all output tensors to SlicingData.bundleTensors, first output already added
    for (unsigned idx = 1; idx < node->getNumOutputs(); idx++)
    {
        data.bundleTensors.push_back(std::make_shared<SlicedOperand>(node->getOutput(idx)));

    }

    return s;
}

SlicingStrategy::SlicingStrategy(const HalReader& halReader, const StrategySlicingDataPtr& slicingData)
: m_halReader(halReader), m_slicingData(slicingData), m_graphSizeOptimized(false), m_allowUpdateNumOfBuffers(true)
{
}

SlicingStrategy::SlicingStrategy(const SlicingStrategy& rhs, bool resetAlignment)
: m_halReader(rhs.m_halReader),
  m_slicingData(rhs.getSlicingData().clone()),
  m_metrics(rhs.m_metrics),
  m_graphSizeOptimized(rhs.getGraphSizeOptimized()),
  m_allowUpdateNumOfBuffers(true)
{
    if (resetAlignment)
    {
        m_metrics.valid = false;
        this->resetAlignment();
    }
}

SlicingStrategyPtr SlicingStrategy::clone(bool resetAlignment)
{
    return SlicingStrategyPtr(new SlicingStrategy(*this, resetAlignment));
}

SlicingStrategy::Metrics& SlicingStrategy::getMetrics()
{
    if (m_metrics.valid)
        return m_metrics;
    return calculateMetrics();
}

const SlicingStrategy::Metrics& SlicingStrategy::getMetrics() const
{
    return m_metrics;
}

SlicingStrategy::Metrics& SlicingStrategy::calculateMetrics()
{
    MetricsCalculator calculator(m_halReader, this, &m_metrics);
    return calculator.calculate();
}

SlicingStrategy::Metrics& SlicingStrategy::recalculateSramCapacity()
{
    MetricsCalculator calculator(m_halReader, this, &m_metrics);
    return calculator.recalculateSramCapacityOnly();
}

void SlicingStrategy::printLog(int logLevel, const synapse::LogManager::LogType& logName) const
{
    if (!log_level_at_least(logName, logLevel)) return;

    SYN_LOG(logName, logLevel, "Slicing Strategy SRAM Capacity: {}", getMetrics().SRAMCapacity);

    const StrategySlicingData& data = getSlicingData();
    for (int i = 0; i < data.bundleTensors.size(); i++)
    {
        const pSlicedOperand& input      = data.bundleTensors[i];
        const pTensor&        origTensor = input->originalTensor;
        SYN_LOG(logName,
                logLevel,
                "Operand [{}] {} : {}, Sliced : {}, Num of slices: {}, Buffers: {}, inSram: {}, alignedToCL:{}",
                i,
                origTensor->getName(),
                toString(input->finalShape.begin(), input->finalShape.begin() + input->originalTensor->getDim(), 'x'),
                toString(input->chunkDimensions.data(), input->chunkDimensions.data() + origTensor->getDim(), 'x'),
                SlicedOperandUtils::nofSlices(input),
                input->numOfBuffers,
                input->resideInSRAM,
                input->alignWithCacheLine);
    }
    const pSlicedOperand& output = data.masterOperand;
    SYN_LOG(logName,
            logLevel,
            "Prime MME Output {} : {}, Sliced : {}, Num of slices: {}, Buffers: {}, inSram: {}, alignedToCL:{}",
            output->originalTensor->getName(),
            toString(output->finalShape.begin(), output->finalShape.begin() + output->originalTensor->getDim(), 'x'),
            toString(output->chunkDimensions.data(),
                     output->chunkDimensions.data() + output->originalTensor->getDim(),
                     'x'),
            SlicedOperandUtils::nofSlices(output),
            output->numOfBuffers,
            output->resideInSRAM,
            output->alignWithCacheLine);
}

StrategySlicingData& SlicingStrategy::getSlicingData()
{
    return *m_slicingData;
}

const StrategySlicingData& SlicingStrategy::getSlicingData() const
{
    return *m_slicingData;
}

void SlicingStrategy::setDoubleBuffer(bool val)
{
    getMetrics().isDoubleBuffered = val;
    getSlicingData().updateNumOfOperandBuffers(val);
}

bool SlicingStrategy::sramSlicedOperandsDoubleBuffered() const
{
    for (const auto& op : getSlicingData().getSlicedOperands())
    {
        for (uint32_t dim = 0; dim < op->originalTensor->getDim(); ++dim)
        {
            if (SlicedOperandUtils::isSlicedOnDimension(op, dim) && op->resideInSRAM && (op->numOfBuffers == 1))
            {
                return false;
            }
        }
    }
    return true;
}

void SlicingStrategy::alignNumBuffers()
{
    // in some cases we can't update the num of buffers since the operands sizes might exceed the max sram capacity
    // it is the solver responsibilty to allow or forbid changing the number of operand buffers
    if (allowUpdateNumOfBuffers())
    {
        getSlicingData().updateNumOfOperandBuffers(m_metrics.isDoubleBuffered);
    }

    if (m_metrics.isDoubleBuffered)
    {
        for (const auto& operand : getSlicingData().getSlicedOperands())
        {
            if (operand->numOfBuffers > 1)
                return;
        }
        // isDoubleBuffered is set to true, but no operand is sliced or in SRAM
        m_metrics.isDoubleBuffered = false;
    }
    // else, all operands num of buffers would be set to 1 anyway - nothing to do
}

void SlicingStrategy::resetAlignment()
{
    // reset alignWithCacheLine member in order not to prevent from cloned strategy to fit into sram
    for (const auto &op : getSlicingData().getSlicedOperands())
    {
        op->alignWithCacheLine = false;
    }
}

void SlicingStrategy::alignWalkingPattern()
{
    auto& slicingData = getSlicingData();
    if ((slicingData.traversalPattern.size() > 1) &&
        (slicingData.getWalkingDir() == StrategySlicingData::WalkingDir::TopToBottom) &&
        (SlicedOperandUtils::getNumOfSlicedDims(slicingData.masterOperand) < 2))
    {
        // Trivially or single dim sliced output means that the walking pattern is LTR.
        // If it is set to Top-to-Bottom, reverse it so the metrics and stitching will have the real picture.
        std::reverse(slicingData.traversalPattern.begin(), slicingData.traversalPattern.end());
    }
}

void SlicingStrategy::alignShapeTensorsSlicing(const pBundle& bundle)
{
    for (auto& bundleOperand : getSlicingData().bundleTensors)
    {
        if (bundleOperand->originalTensor->getTensorType() == OUTPUT_DESCRIBING_SHAPE_TENSOR)
        {
            // TODO [SW-53001]: Remove this W/A. It's a temporary solution to a design bug in kernels that get ODST and
            // request packing
            if (bundle->type() == SCALAR_PIPE)
            {
                bundleOperand->resideInSRAM = false;
                return;
            }

            pNode shapeConsumer = nullptr;
            for (const auto& node : getSlicingData().getStrategyNodes(bundle))
            {
                for (const auto& tensor : node->getInputs())
                {
                    if (tensor == bundleOperand->originalTensor)
                    {
                        shapeConsumer = node;
                        break;
                    }
                }
            }
            HB_ASSERT(shapeConsumer != nullptr,
                      "Did not find shape consumer for tensor {}",
                      bundleOperand->originalTensor->getName());

            pTensor describedTensor = nullptr;
            for (const auto& tensor : shapeConsumer->getOutputs())
            {
                if (tensor->getAllSizesInElements() == bundleOperand->originalTensor->getAllSizesInElements())
                {
                    describedTensor = tensor;
                    break;
                }
            }
            HB_ASSERT(describedTensor != nullptr,
                      "Did not find describedTensor consumer for tensor {}",
                      bundleOperand->originalTensor->getName());

            pSlicedOperand describedByShape = nullptr;
            // Find the operand the shape tensor is describing
            for (const auto& slicedOperand : getSlicingData().getSlicedOperands())
            {
                if (slicedOperand->originalTensor == describedTensor)
                {
                    describedByShape = slicedOperand;
                }
            }

            HB_ASSERT(describedByShape != nullptr, "Did not find operand for tensor {}", describedTensor->getName());

            auto shapeTensor              = bundleOperand->originalTensor;
            *bundleOperand                = *describedByShape;
            bundleOperand->originalTensor = shapeTensor;
            bundleOperand->resideInSRAM   = false;
        }
    }
}

// this comparison is relevant only before bundle expansion process
bool SlicingStrategy::compareInitialStrategy(const SlicingStrategy& other, bool exactMatch) const
{
    return (getSlicingData().compareInitialSlicing(other.getSlicingData(), exactMatch));
}

SlicingStrategy& SlicingStrategy::setInputIsInSRAM(unsigned idx, bool val)
{
    getSlicingData().bundleTensors[idx]->resideInSRAM = val;
    return *this;
}

SlicingStrategy& SlicingStrategy::setOutputIsInSRAM(bool val)
{
    getSlicingData().masterOperand->resideInSRAM = val;
    return *this;
}

// the goal of this function is to create a "unique" string from a slicing data, so that during the "solving" stage,
// when trying to merge the new strategies created by the graph size optimization solver, we will not add new strategies
// that in fact represent the same slicing as an exiting one.
std::string SlicingStrategy::getSlicingDataString(bool exactMatch) const
{
    std::stringstream ss;
    for (auto dim : getSlicingData().traversalPattern)
    {
        ss<<dim<<"_";
    }
    for (auto op : getSlicingData().getSlicedOperands())
    {
        ss<<op->toString()<<";";
    }
    return ss.str();
}
