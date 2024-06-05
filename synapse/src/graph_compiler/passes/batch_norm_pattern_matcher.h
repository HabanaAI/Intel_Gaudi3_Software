#pragma once

#include "habana_graph.h"
#include "synapse_common_types.h"

// Current Instance-Norm TPC kernel implementation for both FWD and BWD is single stage.
// Since the norm plane is {W,H}, TPC disables split on these dims, so the parallelization
// can be done on the rest of the dims i.e. B and C for FWD and only C for BWD.
// Since 2-stage kernel for instance-norm BWD is currently not available, to handle low FCD
// and low batch size cases, instance-norm BWD is reduced to multiple batch-norms with B=1
// for optimal performance.
// This class identifies this pattern.
// Example for B=2:
//
// ┌────────┐                          ┌────────────────────────────┐
// │  IFM   ├──────────────┐           │                            │
// └────────┘              │           │                       ┌────▼────────┐
//                         │           │                    ┌──►             │
//                   ┌─────▼────┐      │                    │  │             │
//                   │split_IFM ├──────┼────────────────────┤  │             │                 ┌───────────────────────┐
//                   └──────────┘      │                    │  │    BN1      ├─────────────────►                       │              ┌──────────┐
// ┌────────┐                          │                ┌───┼──►             │                 │ concat_GRAD_OUT       ├─────────────►│GRAD_OUT  │
// │GRAD_IN ├──────────────┐           │                │   │  │             ├──────────────┐  │                       │              └──────────┘
// └────────┘              │           │           ┌────┼───┼──►             │              │  └──────────▲────────────┘
//                   ┌─────▼────┐      │           │    │   │  └─────▲───────┴──────────┐   │             │
//                   │split_GRAD├──────┼───────────┼────┤   │        │                  │   │             │
//                   └──────────┘      │           │    │   │        │              ┌───┼───┼─────────────┘
// ┌────────┐                          │     ┌─────┼────┼───┼────────┘              │   │   │
// │ MEAN   ├──────────────┐           │     │     │    │   │                       │   │   │
// └────────┘              │           │     │     │    │   │                       │   │   │  ┌───────────────────────┐
//                   ┌─────▼────┐      │     │     │    │   │  ┌─────────────┐      │   │   │  │                       │              ┌──────────┐
//                   │split_MEAN├──────┼─────┼─────┤    │   └──►             ├──────┘   │   └──► concat_GRAD_BETA      ├──────────────►GRAD_BETA │
//                   └──────────┘      │     │     │    │      │             │          │      │                       │              └──────────┘
// ┌────────┐                          │     │     │    └──────►             │          │      └─────────▲─────────────┘
// │ ISTD   ├──────────────┐           │     │     │           │    BN2      │          │                │
// └────────┘              │           │     │     └───────────►             ├──────────┼────────────────┘
//                   ┌─────▼────┐      │     │                 │             │          │
//                   │split_ISTD├──────┴─────┼─────────────────►             ├─────┐    │
//                   └──────────┘            │                 └─────▲───────┘     │    │
//                                           │                       │             │    │      ┌───────────────────────┐
//                                           │                       │             │    └──────►                       │              ┌──────────┐
// ┌────────┐                                │                       │             │           │ concat_GRAD_GAMMA     ├──────────────►GRAD_GAMMA│
// │GAMMA   ├────────────────────────────────┴───────────────────────┘             │           │                       │              └──────────┘
// └────────┘                                                                      │           └──────────▲────────────┘
//                                                                                 │                      │
//                                                                                 └──────────────────────┘
//
class InstanceNormToBatchNormPatternMatcher
{
public:
    InstanceNormToBatchNormPatternMatcher(const HabanaGraph& graph);

    // If the node is BN that fits the above pattern return <true, concurrency-level>
    // where concurrency-level = batch-size (number of BNs created from a single instance-norm).
    // Otherwise, return <false, 0>.
    std::pair<bool, TSize> matchPattern(const NodePtr& node) const;

private:
    enum BnBwdInputs
    {
        IFM_INPUT_IDX     = 0,
        GRAD_IN_INPUT_IDX = 1,
        MEAN_INPUT_IDX    = 2,
        ISTD_INPUT_IDX    = 3,
        GAMMA_INPUT_IDX   = 4,
        NUM_INPUTS        = 5
    };

    enum BnBwdOutputs
    {
        GRAD_OUT_OUTPUT_IDX   = 0,
        GRAD_BETA_OUTPUT_IDX  = 1,
        GRAD_GAMMA_OUTPUT_IDX = 2,
        NUM_OUTPUTS           = 3
    };

    bool matchBatchNormSubPattern(const NodePtr& node) const;

    std::optional<NodePtr> matchSplitSubPattern(const NodePtr& node, unsigned inputIdx) const;

    std::optional<NodePtr> matchConcatSubPattern(const NodePtr& node, unsigned outputIdx) const;

    const std::array<std::string, 2> m_supportedBNGuids = {"batch_norm_bwd_bf16", "batch_norm_bwd_f32"};

    // Map from producers+consumers (for inputs+outputs) to BN nodes.
    // <IFM producer, GRAD-IN producer, MEAN producer, ISTD producer, GAMMA tensor, GRAD-OUT consumer, GRAD_BETA
    // consumer, GRAD_GAMMA consumer>
    std::map<std::tuple<NodePtr, NodePtr, NodePtr, NodePtr, TensorPtr, NodePtr, NodePtr, NodePtr>, NodeVector>
        m_operandsToNodes;

    // Map from BN node to concurrency-level, contains the BN nodes that fit the pattern above.
    std::unordered_map<NodePtr, TSize> m_matchedNodes;

    const HabanaGraph& m_graph;
};