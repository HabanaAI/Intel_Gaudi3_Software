#include "strategy_filter.h"

namespace gc::layered_brain::slicer
{

bool StrategyFilter::operator()(const StrategyPtr& strategy) const
{
    if (cpdFromStrategy(strategy).hasConflict()) return false;
    return true;
}

ConflictingPerforationDetector StrategyFilter::cpdFromStrategy(const StrategyPtr& strategy) const
{
    ConflictingPerforationDetector cpd {};

    for (const auto& n : m_keyNodes)
    {
        const auto& bvds        = m_bundleViewContainer->getNodesBVDs({n});
        const auto& perforation = strategy->getPerforationBVDForNode(n);
        cpd.addNodeBVDs(perforation, bvds);
    }

    return cpd;
}

void ConflictingPerforationDetector::addNodeBVDs(const std::optional<BundleViewId>& perforationBVD,
                                                 const BVDSet&                      nodeDimsBVDs)
{
    if (perforationBVD)
    {
        m_nodeBVDs.push_back(NodeBVDs {*perforationBVD, nodeDimsBVDs});
    }
}

// true if there is some conflict in the perforation bvd assignment
bool ConflictingPerforationDetector::hasConflict() const
{
    if (m_nodeBVDs.empty()) return false;

    if (!validPerforations() || !disjointPerforations()) return true;

    return false;
}

// true if all the perforation bvds appear in the node bvds collection
bool ConflictingPerforationDetector::validPerforations() const
{
    for (const auto& nBvds : m_nodeBVDs)
    {
        if (nBvds.nodeDimsBVDs.find(nBvds.perforationBVD) == nBvds.nodeDimsBVDs.end())
        {
            return false;
        }
    }
    return true;
}

// When multiple perforation choices are taken (some nodes are perforated over one bvd and some nodes are perforated on
// another), this choice will be called 'disjoint' if there are no shared bvds that could have been chosen for all the
// nodes OR there are shared bvds, but none of them was chosen for any of the nodes.
// E.g. some nodes are perforated on bvd-1, but could have been perforated on bvd-2.
// If there are some other nodes that are perforated on bvd-2, then this selection is not disjointed.
// If there are some other nodes that are perforated on bvd-3, then even if bvd-2 is an alternative for them, they are
// still considered disjointed.
// Either everyone uses a shared alternative, or none of them do.
bool ConflictingPerforationDetector::disjointPerforations() const
{
    auto alternativePerforations = calcAlternatives();

    while (!alternativePerforations.empty())
    {
        const auto [perforationBvd1, alternatives1] = *alternativePerforations.begin();
        alternativePerforations.erase(perforationBvd1);
        for (const auto& [perforationBvd2, alternatives2] : alternativePerforations)
        {
            // perforationBvd1 != perforationBvd2 since they are keys of a map
            auto sharedAlternatives = intersect(alternatives1, alternatives2);
            if (contains(sharedAlternatives, perforationBvd1) || contains(sharedAlternatives, perforationBvd2))
                return false;
        }
    }
    return true;
}

// For every perforation bvd, b0, finds alternatives bvds {b0, b1, b2, ..}, such that all the nodes that are perforated
// on b0 can also be perforated instead on b1, b2, etc. Note: b0 _will_ be included in the alternatives set.
ConflictingPerforationDetector::Alternatives ConflictingPerforationDetector::calcAlternatives() const
{
    Alternatives alternativePerforations;
    for (const auto& nBvds : m_nodeBVDs)
    {
        bool inserted = alternativePerforations.insert({nBvds.perforationBVD, nBvds.nodeDimsBVDs}).second;
        if (!inserted)
        {
            alternativePerforations[nBvds.perforationBVD] =
                intersect(alternativePerforations[nBvds.perforationBVD], nBvds.nodeDimsBVDs);
        }
    }
    return alternativePerforations;
}

BVDSet ConflictingPerforationDetector::intersect(const BVDSet& s1, const BVDSet& s2)
{
    BVDSet result;
    for (const auto& e1 : s1)
    {
        if (contains(s2, e1)) result.insert(e1);
    }
    return result;
}

bool ConflictingPerforationDetector::contains(const BVDSet& s, const BundleViewId& e)
{
    return s.find(e) != s.end();
}
}  // namespace gc::layered_brain::slicer