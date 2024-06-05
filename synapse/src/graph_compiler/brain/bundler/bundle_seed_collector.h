#pragma once

#include "node.h"
#include "bundle_expander.h"
#include "layered_brain_bundle.h"
#include <string_view>

class HabanaGraph;

namespace gc::layered_brain::bundler
{
using BundleExpanders    = std::list<BundleExpanderPtr>;
using BundleAndExpanders = std::pair<BundlePtr, BundleExpanders>;
class SeedCollector;
using SeedCollectorPtr = std::shared_ptr<SeedCollector>;

/**
 * @brief Interface defining an object that collects all bundle seeds of some sort in graph.
 *        behavior is defined by implementing collect().
 *
 */
class SeedCollector
{
public:
    using CollectorEnumValueType = uint8_t;

    enum class Type : CollectorEnumValueType
    {
        ATTENTION,
        MULTI_MME,
        SINGLE_BATCH_GEMM,
        SINGLE_GEMM,
        SINGLE_CONV,
        NUM_TYPES
    };

    struct TypeCompare
    {
        inline bool operator()(const SeedCollector::Type& lhs, const SeedCollector::Type& rhs) const
        {
            // Lower enum val ==> higher priority
            return static_cast<CollectorEnumValueType>(lhs) < static_cast<CollectorEnumValueType>(rhs);
        }
    };

    explicit SeedCollector(HabanaGraph& graph) : m_graph(graph), m_ruleLibrary(RuleLibrary::instance()) {}

    /**
     * @brief Collect seeds and return them in a vector of <bundle, expander list> pairs
     *
     */
    virtual std::vector<BundleAndExpanders> collect(bool iterative = false) = 0;

    /**
     * @brief Returns enum corresponding to the seed collector type
     *
     */
    virtual Type getType() const = 0;

    /**
     * @brief Forces derived classes to define a dtor
     *
     */
    virtual ~SeedCollector() = default;

protected:
    HabanaGraph&       m_graph;
    const RuleLibrary& m_ruleLibrary;
};

static inline std::string_view toString(SeedCollector::Type type)
{
    switch (type)
    {
        case SeedCollector::Type::SINGLE_CONV:
            return "SingleConvCollector";
        case SeedCollector::Type::SINGLE_BATCH_GEMM:
            return "SingleBatchGemmCollector";
        case SeedCollector::Type::SINGLE_GEMM:
            return "SingleGemmCollector";
        case SeedCollector::Type::MULTI_MME:
            return "MultiMmeCollector";
        case SeedCollector::Type::ATTENTION:
            return "AttentionCollector";
        default:
            HB_ASSERT(false, "Unexpected enum {}", type);
            return "Invalid";  // dummy
    }
}

}  // namespace gc::layered_brain::bundler
