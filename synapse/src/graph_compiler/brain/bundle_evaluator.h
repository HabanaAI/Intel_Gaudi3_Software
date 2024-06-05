#pragma once

#include <cstdint>
#include <string_view>

#include "brain_data.h"
#include "habana_graph.h"
#include "bundler/layered_brain_bundle.h"
#include "slicer/slicer.h"
#include "slicer/slicing_details.h"

namespace gc::layered_brain
{
using EvalResultValue = uint32_t;
enum class EvalResult : EvalResultValue
{
    UNEVALUATED,
    LOW,
    VALID,
    HIGH
};
std::string_view toString(const EvalResult& type);
struct Evaluation
{
    template<typename T>
    struct BundleMetric
    {
        T          value         = 0;
        EvalResult eval          = EvalResult::UNEVALUATED;
        NodePtr    offendingNode = nullptr;
        bool       valid() const { return eval == EvalResult::VALID; }
        bool       unevaluated() const { return eval == EvalResult::UNEVALUATED; }
    };

    BundleMetric<float>    mmeUtil;
    BundleMetric<float>    mmeBw;
    BundleMetric<uint64_t> cacheUtil;
    BundleMetric<uint64_t> nofSlices;
    BundleMetric<uint64_t> perforationMultiplier;
    BundleMetric<float>    perforationUtil;
    std::string            evalString() const;
    bool                   allMetricsValid() const;
};

using EvalPtr      = std::shared_ptr<Evaluation>;
using ConstEvalPtr = std::shared_ptr<const Evaluation>;
using MaxUtilPerMME = std::unordered_map<NodePtr, decltype(Evaluation::mmeUtil.value)>;

class BundleEvaluator
{
public:
    BundleEvaluator(const BundlePtr&              bundle,
                    const ConstSlicingDetailsPtr& slicingDetails,
                    const MaxUtilPerMME&          maxUtilPerMME);
    ConstEvalPtr postSlicingEvaluation(const HabanaGraph& slicedGraph) const;
    ConstEvalPtr preSlicingEvaluation() const;
    static MaxUtilPerMME collectMaxUtils(const BundlePtr& bundle, const StrategyVector& strategies);

private:
    void handleNofSlices(Evaluation::BundleMetric<uint64_t>& nofSlices) const;
    void handleMmeBw(Evaluation::BundleMetric<float>& mmeBw) const;
    void handleCacheUtil(Evaluation::BundleMetric<uint64_t>& cacheUtil, uint64_t cacheUsed, uint64_t cacheSize) const;
    void handleMmeUtil(Evaluation::BundleMetric<float>& mmeUtil) const;
    void handlePerforationMultiplier(Evaluation::BundleMetric<uint64_t>& perforationMultiplier) const;
    void handlePerforationUtil(Evaluation::BundleMetric<float>& perforationUtil) const;

    static EvalResult evaluateNofSlices(uint64_t nofSlices);
    EvalResult        evaluateMmeBw(float bw) const;
    EvalResult        evaluateMmeUtil(const NodePtr& mme, float util) const;

    const NodeSet& getBundledMmeNodes() const;
    const BundleData& getBundleData(const HabanaGraph& slicedGraph) const;
    NodeSet        acquireBundledMmeNodes(const BundlePtr& bundle);

    const BundlePtr&                                                 m_bundle;
    NodeSet                                                          m_bundleMmeNodes;
    const ConstSlicingDetailsPtr                                     m_slicingDetails;
    const MaxUtilPerMME                                              m_mmeToMaxUtil;
};
}  // namespace gc::layered_brain