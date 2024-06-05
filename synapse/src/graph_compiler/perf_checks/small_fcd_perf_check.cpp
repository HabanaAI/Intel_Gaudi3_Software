
#include "small_fcd_perf_check.h"
#include "habana_graph.h"
#include "include/mme_common/mme_brain.h"
#include "types.h"

constexpr TSize hugeTensor   = (1 << 30);  // 1   GB
constexpr TSize bigTensor    = (1 << 28);  // 256 MB
constexpr TSize smallTensor  = (1 << 20);  // 1   MB

constexpr unsigned hugeNumberOfReads = 100;
constexpr unsigned manyReads         = 10;

GcPerf::GcPerfCheckPtr SmallFcdPerfCheck::createPerfCheck(const HabanaGraph& g)
{
    return GcPerf::GcPerfCheckPtr(new SmallFcdPerfCheck(g));
}

SmallFcdPerfCheck::SmallFcdPerfCheck(const HabanaGraph& g) : GcPerf::GcPerfCheck(g)
{
    const auto& hal = g.getHALReader();
    HB_ASSERT_PTR(hal);
    m_cacheLineSizeInBytes = hal->getCacheLineSizeInBytes();
}

unsigned SmallFcdPerfCheck::getNumberOfReads(const TensorPtr& t) const
{
    unsigned ret = 0;
    for (const auto& consumer : m_graph.getTensorConsumers(t))
    {
        if (!consumer) continue;
        ++ret;
        if (m_graph.runsOnMME(consumer))
        {
            const auto& perfAttr = consumer->getNodeAnnotation().mmeMetaData.mmePerfAttr;
            if (perfAttr)
            {
                if (t == consumer->getInput(0))  // is operand A
                {
                    ret += std::max<unsigned>(1, perfAttr->fetchNrA) - 1;
                }
                else if (t == consumer->getInput(1))  // is operand B
                {
                    ret += std::max<unsigned>(1, perfAttr->fetchNrB) - 1;
                }
            }
            else
            {
                LOG_WARN(GC, "can't find mme perf attribute for {}", consumer->getNodeName());
            }
        }
    }
    return ret;
}

// check if a tensor with small fcd will read more than one time
void SmallFcdPerfCheck::performReadChecks(const TensorPtr& t, const TSize& fcdSize) const
{
    // If the Fcd is bigger than cacheline we skip the rest of the checks.
    if (fcdSize >= m_cacheLineSizeInBytes) return;

    auto sizeInBytes = t->getDenseSizeInBytes();
    if (sizeInBytes <= smallTensor) return;  // If the tensor is small we ignore it.

    auto numOfReads = getNumberOfReads(t);
    if (numOfReads == 1) return;  // If the tensor will be read only one time we have nothing to do.

    auto logLevel = GcPerf::LogLevel::LOW;  // For the case that 2 <= numOfReads < manyReads.
    if (numOfReads >= hugeNumberOfReads || (sizeInBytes >= bigTensor && numOfReads >= manyReads))
    {
        logLevel = GcPerf::LogLevel::HIGH;
    }

    PERF_REPORT(logLevel,
                "Tensor: {} - with size {} MB and effective FCD {} B, will be read {} times from {}",
                t->getName(),
                bToMb(sizeInBytes),
                fcdSize,
                numOfReads,
                t->inSram() ? "SRAM" : "DRAM");
}

// check if the producer write a tensor with low utilization
void SmallFcdPerfCheck::performWriteChecks(const NodePtr& node, const TensorPtr& t, const TSize& fcdSize) const
{
    auto sizeInBytes = t->getDenseSizeInBytes();
    if (sizeInBytes <= smallTensor) return;  // If the tensor is small we ignore it.

    auto     logLevel           = GcPerf::LogLevel::LOW;  // For the case that smallTensor < sizeInBytes < bigTensor.
    unsigned numberOfCacheLines = 1;
    if (sizeInBytes >= bigTensor)
    {
        logLevel           = GcPerf::LogLevel::HIGH;
        numberOfCacheLines = (sizeInBytes >= hugeTensor) ? 3 : 2;
    }

    if (fcdSize < m_cacheLineSizeInBytes * numberOfCacheLines)
    {
        PERF_REPORT(logLevel,
                    "Tensor: {} - The node {} write tensor of size {} MB and the effective FCD is {} B to {}",
                    t->getName(),
                    node->getNodeName(),
                    bToMb(sizeInBytes),
                    fcdSize,
                    t->inSram() ? "SRAM" : "DRAM");
    }
}

void SmallFcdPerfCheck::run(const NodePtr& node) const
{
    for (const auto& output : node->getOutputs())
    {
        auto [denseShape, denseStrides] = mergeDenseDimensions(output);
        const auto& fcdSize             = denseShape.getMaxSize(0) * output->getElementSizeInBytes();
        if (denseShape.getDim() == 1 ||               // the tensor is dense
            fcdSize % m_cacheLineSizeInBytes == 0 ||  // the tensor size that is dense in memory is aligned to cacheline
            denseStrides[1] % m_cacheLineSizeInBytes == 0)  // the tensor already aligned to cacheline
            continue;

        performWriteChecks(node, output, fcdSize);
        performReadChecks(output, fcdSize);
    }
}