#pragma once
#include "bundle_break_rules.h"
#include "bundle_seed_collector.h"
#include "types.h"
#include <cstdint>

class HabanaGraph;
namespace gc::layered_brain::bundler
{
class MmeNodeCollector : public SeedCollector
{
public:
    explicit MmeNodeCollector(HabanaGraph& graph);
    static bool bundleOnlyIsolatedMmeNodes();
    virtual ~MmeNodeCollector() = default;

    virtual Type                            getType() const override = 0;
    virtual std::vector<BundleAndExpanders> collect(bool iterative = false) override;

protected:
    void injectBreakRules(const std::vector<RuleType>& rules);
    bool passesCommonRules(const NodePtr& n) const;
    bool passesBreakRules(const NodePtr& n, const BundlePtr& bundleSeed = nullptr) const;
    template<bool IsIterative>
    BundleExpanders getExpanders(const NodeSet& seed, const BundlePtr& bundle) const;
    bool            shouldAddConditionalExpander(const NodePtr& n) const;
    void            logSeed(const BundlePtr& seed) const;

private:
    void                       applyExpandersOrder(BundleExpanders& expanders) const;
    const std::vector<RulePtr> m_commonRules;
    std::vector<RulePtr>       m_breakRules;
};

class SingleBatchGemmCollector : public MmeNodeCollector
{
public:
    explicit SingleBatchGemmCollector(HabanaGraph& graph);
    Type getType() const override { return Type::SINGLE_BATCH_GEMM; }
};

class SingleGemmCollector : public MmeNodeCollector
{
public:
    explicit SingleGemmCollector(HabanaGraph& graph);
    Type getType() const override { return Type::SINGLE_GEMM; }
};

class SingleConvCollector : public MmeNodeCollector
{
public:
    explicit SingleConvCollector(HabanaGraph& graph);
    Type getType() const override { return Type::SINGLE_CONV; }
};

class MultiMmeCollector : public MmeNodeCollector
{
public:
    explicit MultiMmeCollector(HabanaGraph& graph);
    virtual ~MultiMmeCollector()                                     = default;
    virtual Type                            getType() const override = 0;
    virtual std::vector<BundleAndExpanders> collect(bool iterative = false) override;

private:
    struct MmeGroup
    {
        using Id = uint64_t;
        MmeGroup(const TensorPtr& t, const std::vector<NodePtr>& nodes, uint64_t id)
        : m_sharedOperand(t), m_nodes(nodes), m_id(id)
        {
        }

        TensorPtr            m_sharedOperand;
        std::vector<NodePtr> m_nodes;
        Id                   m_id;
    };

    std::vector<MmeGroup> gatherMmeGroups() const;
};

class MultiGemmCollector : public MultiMmeCollector
{
public:
    explicit MultiGemmCollector(HabanaGraph& graph);
    Type getType() const override { return Type::MULTI_MME; }
};

class AttentionCollector : public SeedCollector
{
public:
    explicit AttentionCollector(HabanaGraph& graph);
    virtual ~AttentionCollector() = default;
    Type                            getType() const override { return SeedCollector::Type::ATTENTION; }
    std::vector<BundleAndExpanders> collect(bool iterative = false) override;

protected:
    bool getAttentionBundle(const BundlePtr& bundle, const NodePtr& tpcSeed) const;
    void logSeed(const BundlePtr& seed) const;
    bool passesTpcSeedRules(const NodePtr& n) const;

    const std::vector<RulePtr> m_tpcSeedRules;
};

}  // namespace gc::layered_brain::bundler