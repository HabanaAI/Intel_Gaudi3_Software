#ifndef _TRANSPOSENODESCREATOR_H_
#define _TRANSPOSENODESCREATOR_H_

#include <memory>
#include <vector>

#include "transpose_node.h"
#include "transpose_permutation.h"
#include "compilation_hal_reader.h"

namespace gc
{
class EngineSelector;
}

class Node;
class Tensor;

enum TransposeStrategyID
{
    UNUSED,
    TRANSPOSE_SHAPE,
    TRANSPOSE_VIA_RESHAPE,
    LOGICAL_TRANSPOSE,
    STATIC_TRANSPOSE,
    PHYSICAL_TRANSPOSE
};

namespace fmt
{
template<>
struct formatter<std::optional<std::string>> : fmt::formatter<std::string>
{
    template<typename FormatContext>
    auto format(const std::optional<std::string>& opt, FormatContext& ctx)
    {
        if (opt)
        {
            fmt::formatter<std::string>::format(*opt, ctx);
            return ctx.out();
        }
        return fmt::format_to(ctx.out(), "nullopt");
    }
};
}  // namespace fmt

class TransposeCostModel
{
public:
    virtual uint64_t getCost(const NodeVector& extractedNodes) const = 0;
    virtual uint64_t getCost(const TensorPtr& input, const TransposePermutationArray& permutation) const = 0;
};

class TransposeNodeStrategy
{
public:
    virtual ~TransposeNodeStrategy() = default;
    virtual bool       canBeUsed(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const = 0;
    virtual NodeVector extract(const TransposeNodeParams& transposeNodeParams, const HalReaderPtr& hal) const   = 0;
    virtual std::string_view    strategyName() const                                            = 0;
    virtual TransposeStrategyID getStrategyID() const                                           = 0;
    NodeVector                  handleAliasedOutputTranspose(const TransposeNodeParams& transposeNodeParams,
                                                             const HalReaderPtr&        hal) const;
    static NodeVector           createReshapeWithExtractTranspose(const TensorPtr&                 in,
                                                                  const TensorPtr&                 out,
                                                                  const TransposePermutationArray& permutation,
                                                                  const std::string&               name);
    virtual uint64_t            calculateCost(const TransposeNodeParams& transposeNodeParams,
                                              const HalReaderPtr&        hal) const = 0;  // in cycles
};
class TransposeNodesCreator
{
public:
    TransposeNodesCreator()  = default;
    ~TransposeNodesCreator() = default;

    NodeVector getTransposeNodes(const TransposeNode& transpose, TransposeStrategyID strategyToSkip = UNUSED) const;
    std::pair<NodeVector, uint64_t> getTransposeNodesAndCost(const TransposeNode& transpose,
                                                             TransposeStrategyID  strategyToSkip = UNUSED) const;
    NodeVector getTransposeNodesByParams(const TransposeNodeParams& transposeNodeParams,
                                         TransposeStrategyID        strategyToSkip = UNUSED) const;
    uint64_t   getTransposeCostByParams(const TransposeNodeParams& transposeNodeParams,
                                        TransposeStrategyID        strategyToSkip = STATIC_TRANSPOSE) const;

    std::shared_ptr<TransposeCostModel> getCostModel() const;
    const HalReaderPtr                  getHalReader() const { return m_halReader; }

    static constexpr uint32_t FULLY_UTILIZED_MAX_SUPPORTED_DIMS = 4;

    TransposeNodesCreator(const TransposeNodesCreator&) = delete;
    TransposeNodesCreator& operator=(const TransposeNodesCreator&) = delete;

private:
    const TransposeNodeStrategy* getWinningStrategy(const TransposeNodeParams& transposeNodeParams,
                                                    TransposeStrategyID        strategyToSkip) const;

    const HalReaderPtr m_halReader = CompilationHalReader::getHalReader();
};

using LogicTransposeStrategies = std::array<const TransposeNodeStrategy*, 4>;

#endif // _TRANSPOSENODESCREATOR_H_
