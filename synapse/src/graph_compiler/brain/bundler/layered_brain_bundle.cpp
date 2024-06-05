#include "layered_brain_bundle.h"
#include "layered_brain.h"
#include "bundle_plane_graph.h"
#include "bundle.h"

namespace gc::layered_brain
{
Bundle::Bundle(HabanaGraph& g) : m_graph(g), m_idx(nextBundleIndex()) {}

Bundle::~Bundle()
{
    rejectCandidates();
}

BundlePtr Bundle::create(HabanaGraph& g)
{
    return std::make_shared<Bundle>(g);
}

BundleIndex Bundle::index() const
{
    return m_idx;
}

const NodeSet& Bundle::getNodes() const
{
    return m_bundleNodes;
}

bool Bundle::hasCandidates() const
{
    return !m_candidates.empty();
}

BundleIndex Bundle::nextBundleIndex()
{
    return ::Bundle::getNextBundleIndex();
}

void Bundle::addCandidate(const NodePtr& n)
{
    m_candidates.push(n);
    add(n);
}

void Bundle::rejectCandidates()
{
    if (m_candidates.empty()) return;
    auto* pBpg = m_graph.getBPGraph();
    HB_ASSERT_PTR(pBpg);
    while (!m_candidates.empty())
    {
        auto& candidate = m_candidates.top();
        pBpg->unbundleNode(candidate);
        candidate->getNodeAnnotation().bundleInfo.unset();
        m_bundleNodes.erase(candidate);
        m_candidates.pop();
    }
}

void Bundle::acceptCandidates()
{
    while (!m_candidates.empty())
    {
        m_candidates.pop();
    }
}

void Bundle::add(const NodePtr& n)
{
    const auto bundleIndex = index();
    BundleInfo bi(bundleIndex, UNDEFINED);
    n->getNodeAnnotation().bundleInfo.set(bi);

    auto* pBpg = m_graph.getBPGraph();
    HB_ASSERT_PTR(pBpg);
    const bool success = pBpg->addNodeToBundle(n, bi);
    HB_ASSERT(success, "Failed adding node {}[{}] to bundle {}", n->getNodeName(), n->getNodeTypeStr(), index());
    m_bundleNodes.insert(n);
}

void Bundle::dismantle()
{
    if (m_bundleNodes.empty())
    {
        HB_ASSERT(m_candidates.empty(), "Expecting empty candidates container if bundle nodes container is empty");
        return;
    }

    // dismantle bundle including candidates in BPG
    auto* pBpg = m_graph.getBPGraph();
    HB_ASSERT_PTR(pBpg);
    pBpg->removeBundle(*m_bundleNodes.begin());

    // unset bundle info of all nodes and candidates
    std::for_each(m_bundleNodes.begin(), m_bundleNodes.end(), [](const auto& n) {
        n->getNodeAnnotation().bundleInfo.unset();
    });

    // empty candidates container
    while (!m_candidates.empty())
        m_candidates.pop();

    // clear bundle nodes container
    m_bundleNodes.clear();
}

}  // namespace gc::layered_brain