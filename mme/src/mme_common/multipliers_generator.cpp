#include "multipliers_generator.h"
#include "index_space_dimensions.h"
#include "spatial_dims_mapping.h"

namespace MmeCommon::Brain::SolutionMultipliers
{
MultipliersGenerator::MultipliersGenerator(const MmeLayerParams& params,
                                           const AccessPattern& accessPattern,
                                           const MultiplierArray& granularity,
                                           const MultiplierArray& previousMultipliers,
                                           AspectFactoryPtr&& aspectFactory)
: m_params(params),
  m_maxMultipliers(calculateMaxMultipliers(accessPattern, granularity)),
  m_operandAccess(accessPattern, granularity),
  m_previousMultipliers(previousMultipliers),
  m_aspectFactory(std::move(aspectFactory)),
  m_solution(previousMultipliers)
{
}

const MultiplierArray& MultipliersGenerator::getSolution() const
{
    return m_solution;
}

MultiplierArray MultipliersGenerator::calculateMaxMultipliers(const AccessPattern& accessPattern,
                                                              const MultiplierArray& granularity)
{
    MME_ASSERT(accessPattern.indexSpace.size() == granularity.size(),
               "Unexpected granularity and index space different ranks");

    MultiplierArray maxMultipliers(granularity.size());
    for (Dim dim = 0; dim < granularity.size(); dim++)
    {
        MME_ASSERT(granularity.at(dim) > 0, "Invalid granularity value");
        maxMultipliers.at(dim) = div_round_up(accessPattern.indexSpace.at(dim), granularity.at(dim));
    }
    return maxMultipliers;
}

void MultipliersGenerator::inflateUpTo(AspectName aspectName, size_t desiredSize, EMmeOperand operand)
{
    resetToPrevious(aspectName);

    if (desiredSize >= aspectFullSize(aspectName, operand))
    {
        // If the desired size is larger then the full size, the rounding below may not acheive it. So,
        // set the multiplier to the maximum value.
        setMaxMultiplier(aspectName);
    }
    else
    {
        for (auto idxSpcDim : idxSpaceAspect(aspectName))
        {
            auto curSize = aspectCurrentSize(aspectName, operand);
            if (curSize < desiredSize)
            {
                auto inflation = div_round_down(desiredSize, curSize);
                inflate(idxSpcDim, inflation);
            }
        }
    }
}

void MultipliersGenerator::inflateAtLeastTo(AspectName aspectName, uint64_t desiredSize, EMmeOperand operand)
{
    resetToPrevious(aspectName);

    // Since the rounding below is upwards (ceiling), no need for a special treatment for the
    // case of desiredSize >= full size.
    for (auto idxSpcDim : idxSpaceAspect(aspectName))
    {
        auto curSize = aspectCurrentSize(aspectName, operand);
        if (curSize < desiredSize)
        {
            auto inflation = div_round_up(desiredSize, curSize);
            inflate(idxSpcDim, inflation);
        }
    }
}

void MultipliersGenerator::inflate(Dim idxSpcDim, uint64_t factor)
{
    m_solution.at(idxSpcDim) = std::min(m_solution.at(idxSpcDim) * factor, uint64_t(m_maxMultipliers.at(idxSpcDim)));
}

void MultipliersGenerator::setMultipliers(MultiplierArray initialSolution)
{
    MME_ASSERT(m_solution.size() == initialSolution.size(), "initial solution size doesnt match solution size");
    for (int dim = 0; dim < m_solution.size(); dim++)
    {
        m_solution.at(dim) = initialSolution.at(dim);
    }
}

void MultipliersGenerator::setMaxMultiplier(AspectName aspectName)
{
    for (auto dim : idxSpaceAspect(aspectName))
    {
        m_solution.at(dim) = m_maxMultipliers.at(dim);
    }
}

void MultipliersGenerator::resetToPrevious(AspectName aspect)
{
    for (auto dim : idxSpaceAspect(aspect))
    {
        m_solution.at(dim) = m_previousMultipliers.at(dim);
    }
}

uint64_t MultipliersGenerator::aspectFullSize(AspectName aspect, EMmeOperand operand) const
{
    uint64_t aspectSize = 1;
    const auto& sizes = m_params.getOperand(operand).sizes;
    for (auto tensorDim : operandAspect(aspect, operand))
    {
        aspectSize *= sizes.at(tensorDim);
    }
    return aspectSize;
}

uint64_t MultipliersGenerator::aspectCurrentSize(AspectName aspect, EMmeOperand operand) const
{
    uint64_t aspectSize = 1;
    const auto& fullSizes = m_params.getOperand(operand).sizes;
    for (auto idxSpcDim : idxSpaceAspect(aspect))
    {
        for (auto tensorDim : m_operandAccess.mappedTensorDims(operand, idxSpcDim))
        {
            // The multipliers * granuleSize may produce a size that is slightly bigger than the full size.
            // This is because the granularity is not necessarily an integer division of the full size.
            auto dimCurrentSize =
                std::min(uint64_t(fullSizes.at(tensorDim)),
                         m_solution.at(idxSpcDim) * m_operandAccess.granularityByTensorDim(operand, tensorDim));
            aspectSize *= dimCurrentSize;
        }
    }
    return aspectSize;
}

IndexSpaceAspect MultipliersGenerator::idxSpaceAspect(AspectName aspectName) const
{
    return m_aspectFactory->create(aspectName);
}

OperandAspect MultipliersGenerator::operandAspect(AspectName aspectName, EMmeOperand operand) const
{
    return m_aspectFactory->create(aspectName, operand);
}

namespace MmeAspects
{
MultipliersGenerator::MultipliersGenerator(const MmeLayerParams& params,
                                           const AccessPattern& accessPattern,
                                           const MultiplierArray& granularity,
                                           const MultiplierArray& previousMultipliers)
: SolutionMultipliers::MultipliersGenerator(params,
                                            accessPattern,
                                            granularity,
                                            previousMultipliers,
                                            AspectFactoryPtr(new PhysicalAspects::Factory(&params)))
{
}
}  // namespace MmeAspects
}  // namespace MmeCommon::Brain::SolutionMultipliers
