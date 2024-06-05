#pragma once

#include "node.h"
#include "layered_brain_bundle.h"

namespace gc::layered_brain::bundler
{
using RuleEnumType = uint8_t;
enum class RuleType : RuleEnumType
{
    NOT_NDIM,
    NOT_DYNAMIC_SHAPE,
    NOT_BUNDLED,
    BUNDLED,
    HAS_ACCESS_PATTERN,
    BINDING_INPUT_REUSE,
    NO_BINDING_INPUT_REUSE,
    OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
    NO_OVERLAP_OR_OFFSET_IN_ACCESS_PATTERN,
    HAS_UNIQUE_INPUTS,
    SUPPORTED_LOGICAL_OPERATION,
    IS_LOGICAL_OPERATION,
    RUNS_ON_TPC,
    NOT_RUNS_ON_TPC,
    RUNS_ON_MME,
    NOT_RUNS_ON_MME,
    NOT_RUNS_ON_DMA,
    IS_ISOLATED,
    NO_ANCESTORS_IN_BUNDLE,
    SUPPORTED_GEMM,
    IS_GEMM,
    SUPPORTED_BATCH_GEMM,
    SUPPORTED_MULTI_GEMM_SEED,
    SUPPORTED_ATTENTION_SEED,
    IS_CONV,
    VALID_BATCH_GEMM,
    IS_BATCH_GEMM,
    HAS_SHARED_INPUT_IN_BUNDLE,
    SINGLE_SHARED_MME_OPERAND_IN_BUNDLE,
    NO_SHARED_INPUT_IN_BUNDLE,
    NO_MULTIPLE_OUTPUTS_IN_BUNDLE,
    NO_PRODUCER_BPG_CYCLE,
    NO_CONSUMER_BPG_CYCLE,
    NO_SHARED_MME_OPERAND_BPG_CYCLE,
    NOT_MAX_CANDIDATES,
    IS_VALID_SEED_SIZE,
    LAST_CANDIDATE_TPC,
    LAST_CANDIDATE_MME,
    EXACTLY_2_TPC_PRODUCERS,
    NO_MME_PRODUCERS_BUNDLED,
    NO_MME_CONSUMERS_BUNDLED,
    NO_TPC_CONSUMERS,
    NOT_SERIALIZE_DESERIALIZE_NODE,
    MME_INPUTS_LARGER_THAN_OUTPUTS,
    MME_OUTPUTS_LARGER_THAN_INPUTS,
    NUM_BREAK_RULES
};

struct RuleContext
{
    using CandidatesPtr = std::shared_ptr<const std::vector<NodePtr>>;
    RuleContext(const HabanaGraph&   g,
                const NodePtr&       pn = nullptr,
                const BundlePtr&     pb = nullptr,
                const CandidatesPtr& pc = nullptr)
    : graph(g), node(pn), bundle(pb), curCandidates(pc)
    {
    }
    const HabanaGraph&  graph;
    const NodePtr       node;
    const BundlePtr     bundle;
    const CandidatesPtr curCandidates;
};

class IRule
{
public:
    IRule(const std::string& name, RuleType type) : m_name(name), m_type(type) { HB_ASSERT(!name.empty(), ""); }

    virtual bool apply(const RuleContext& ctx) const = 0;

    std::string name() const { return m_name; }
    RuleType    type() const { return m_type; }

    virtual ~IRule() = default;

protected:
    const std::string m_name;
    const RuleType    m_type;
};
using RulePtr = std::shared_ptr<IRule>;

class RuleLibrary
{
public:
    ~RuleLibrary()                  = default;
    RuleLibrary(const RuleLibrary&) = delete;
    RuleLibrary(RuleLibrary&&)      = delete;
    RuleLibrary operator=(const RuleLibrary&) = delete;
    RuleLibrary operator=(RuleLibrary&&) = delete;

    static const RuleLibrary& instance();
    RulePtr                   getRule(RuleType type) const;
    std::vector<RulePtr>      getRules(const std::vector<RuleType>& types) const;

private:
    void appendRule(const RulePtr& rule);
    RuleLibrary();
    struct RuleCompare
    {
        bool operator()(const RuleType& lhs, const RuleType& rhs) const
        {
            return static_cast<RuleEnumType>(lhs) < static_cast<RuleEnumType>(rhs);
        }
    };
    std::map<RuleType, RulePtr, RuleCompare> m_factoryMap;
};

}  // namespace gc::layered_brain::bundler
