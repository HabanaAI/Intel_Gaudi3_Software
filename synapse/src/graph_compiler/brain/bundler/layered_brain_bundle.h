#pragma once
#include <stack>
#include "layered_brain.h"
#include "types.h"

namespace gc::layered_brain
{
class Bundle;
using BundlePtr = std::shared_ptr<Bundle>;
class Bundle
{
public:
    explicit Bundle(HabanaGraph& g);
    ~Bundle();
    Bundle(const Bundle&) = delete;
    Bundle(Bundle&&)      = delete;
    Bundle& operator=(const Bundle&) = delete;
    Bundle& operator=(Bundle&&) = delete;

    /**
     * @brief Returns a pointer to an empty bundle with a valid bundle index
     *
     */
    static BundlePtr create(HabanaGraph& g);

    /**
     * @brief Returns the bundle index
     *
     */
    BundleIndex index() const;

    /**
     * @brief Get a constant reference to bundle nodes and candidates container
     *
     */
    const NodeSet& getNodes() const;

    /**
     * @brief Get a copy of bundle nodes in a container of choice
     *
     */
    template<typename T>
    T getNodesCopy() const
    {
        const auto& nodes = getNodes();
        return T(nodes.begin(), nodes.end());
    }

    /**
     * @brief True if bundle has candidates pending
     *        accept/reject otherwise false
     */
    bool hasCandidates() const;

    /**
     * @brief Add node as a bundle candidate
     *
     */
    void addCandidate(const NodePtr& n);

    /**
     * @brief Adds all nodes in input container to bundle as candidates
     *
     */
    template<typename NodesContainer>
    void addCandidate(const NodesContainer& nodes)
    {
        std::for_each(nodes.begin(), nodes.end(), [this](auto& n) { addCandidate(n); });
    }

    /**
     * @brief Discard bundle candidates from the bundle
     * @note If there are no candidates, noop
     */
    void rejectCandidates();

    /**
     * @brief Commit on bundle candidates as valid bundle nodes
     * @note If there are no candidates, noop
     */
    void acceptCandidates();

    /**
     * @brief Add input node n directly as a valid bundle node w/o
     *        requiring user to commit on the candidate
     */
    void add(const NodePtr& n);

    /**
     * @brief Adds all nodes in input container to directly to
     *        bundle as valid bundle candidates w/o requiring
     *        user to commit on each candidate
     */
    template<typename NodesContainer>
    void add(const NodesContainer& nodes)
    {
        std::for_each(nodes.begin(), nodes.end(), [this](auto& n) { add(n); });
    }

    /**
     * @brief Dismantle the bundle effectively removing
     *        all bundle nodes and candidates
     */
    void dismantle();

private:
    /**
     * @brief Returns the current next valid bundle index
     *        and increments for next request
     */
    static BundleIndex nextBundleIndex();

    std::stack<NodePtr> m_candidates;
    HabanaGraph&        m_graph;
    BundleIndex         m_idx;
    NodeSet             m_bundleNodes;
};

}  // namespace gc::layered_brain