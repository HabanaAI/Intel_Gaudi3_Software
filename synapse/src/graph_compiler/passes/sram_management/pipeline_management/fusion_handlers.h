#pragma once
#include "pipeline_management/fusion_candidates_collector.h"

class FusionHandler
{
public:
    explicit FusionHandler(const CandidateInfo& candidate) : m_candidate(candidate) {};
    const CandidateInfo& getCandidate() { return m_candidate; };
    virtual bool         fuse(const NodePtr& directive) = 0;

protected:
    CandidateInfo m_candidate;
};

class TpcFusionHandler : public FusionHandler
{
public:
    static bool isValidForFusion(HabanaGraph& g, const NodePtr& directive, const CandidateInfo& candidate);
    TpcFusionHandler(HabanaGraph& g, const CandidateInfo& candidate, TpcRecompileDb& db)
    : FusionHandler(candidate), m_graph(g), m_tpcFusionDb(db) {};
    bool fuse(const NodePtr& directive) override;

protected:
    bool
    recompileKernel(const TPCNodePtr& node, unsigned tensorIdxToDuplicate, void*& resultElf, unsigned& resultElfSize);
    bool handleTpcDoubleStore(const NodePtr& directive, const CandidateInfo& candidate);
    bool isValidLlvmTensorIdPostRecompile(const TPCNodePtr& tpcCandidate, unsigned tensorIdToDuplicate, void*& elf, unsigned elfSize) const;

    HabanaGraph&    m_graph;
    TpcRecompileDb& m_tpcFusionDb;
};

using FusionHandlerPtr = std::shared_ptr<FusionHandler>;