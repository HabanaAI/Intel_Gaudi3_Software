#pragma once

#include "node.h"
#include "tensor.h"
#include "types.h"
#include "section_id_generator.h"

class HabanaGraph;

// TensorCoherenceMapping encapsulates the memory coherency ordering for tensors.
class TensorCoherenceMapping
{
public:
    using TensorCoherence = std::unordered_map<uint64_t, TensorVector>;
    using SectionType     = SectionIDGenerator::AllocationManagementType;

    // registers memory coherence ordering for graph 'g'
    TensorCoherenceMapping(const HabanaGraph& g);

    // find all tensors that overlap with 't' and are produced after it.
    TensorVector findNextCoherencyTensors(const TensorPtr& t) const;

    // find all tensors that overlap with 't' and are produced before it.
    TensorVector findPreviousCoherencyTensors(const TensorPtr& t) const;

    // calculate blocked nodes of 'blockingNode' according to memory coherence using connectivity of graph g
    // 'g' may be different than the original graph used to create the memory coherence
    NodeSet calculateBlockedNodes(const HabanaGraph& g, const NodePtr& blockedNode) const;

    // calculate blocking nodes of 'blockedNode' according to memory coherence using connectivity of graph g
    // 'g' may be different than the original graph used to create the memory coherence
    NodeSet calculateBlockingNodes(const HabanaGraph& g, const NodePtr& blockedNode) const;

    // get internal data structure for tensor memory coherence
    const std::vector<TensorCoherence>& getAllSectionsTensorCoherence() const { return m_mappedTensors; }

    // get internal data structure for tensor memory coherence for a specific type
    const TensorCoherence& getAllSectionsTensorCoherence(SectionType type) const { return m_mappedTensors[type]; }

    // debug function to print internal data structure
    void printMemoryCoherence() const;

    // validate the data structure is legal according to assumptions taken
    void validateMemoryCoherence() const;

    // validate that every coherent tensor readers/writers have a correct defined path between them.
    bool validatePostGraphMemoryCoherence() const;

    // check if graph contains legacy RAW dependencies that may disable some optimizations
    bool doReadAfterWriteExternalDependenciesExist() const;

    // should 't' be registered
    static bool isMemoryCoherencyTensor(const TensorPtr& t)
    {
        return !t || t->isUserManagedDram() || t->isPartOfRMWSection();
    }

    // check if t is registered
    bool doesCoherencyTensorExist(const TensorPtr& t) const;

    class CoherencyComparator
    {
    public:
        CoherencyComparator(const TensorCoherenceMapping& coherenceMapping);
        bool operator()(const TensorPtr& t1, const TensorPtr& t2) const;

    private:
        const TensorCoherenceMapping& m_coherenceMapping;
    };
    bool overlapsWithOthersInSection(const TensorPtr& t) const;
    bool previousOverlapsWithOthersInSection(const TensorPtr& t) const;

private:
    TensorVector findCoherencyTensors(const TensorPtr& t, bool previousTensors) const;

    static SectionType     getTensorCoherencyType(const TensorPtr& t);
    bool                   skipTensor(const TensorPtr& t) const;
    TensorCoherence&       getTensorCoherence(const TensorPtr& t);
    const TensorCoherence& getTensorCoherence(const TensorPtr& t) const;
    uint64_t               getTensorSectionId(const TensorPtr& t) const;
    void                   printMemoryCoherence(const TensorCoherence& allSectionsTensors) const;
    void                   validateMemoryCoherence(const TensorCoherence& allSectionsTensors) const;

    // used for post graph validation
    using RealTensorMap = std::map<TensorPtr, TensorSet, TensorComparator>;  // {RealTensor --> all alias tensors}
    RealTensorMap buildRealTensorMapping(SectionType type) const;
    bool          validatePostGraphCoherencyTensors(const RealTensorMap& postGraphMapping,
                                                    const TensorPtr&     blockingRealTensor,
                                                    const TensorPtr&     blockedRealTensor) const;
    bool          validatePostGraphCoherencyMapping(SectionType type, const RealTensorMap& postGraphMapping) const;
    bool          validatePostGraphMemoryCoherence(SectionType type) const;
    bool          validateWriteAfterWrite(const TensorPtr& blocking, const TensorPtr& blocked) const;
    bool          validateWriteAfterRead(const TensorPtr& blocking, const TensorPtr& blocked) const;
    bool          validatePostGraphConnectivity(SectionType type, const RealTensorMap& postGraphMapping) const;
    static bool   isMemoryCoherencyTensor(const TensorPtr& t, SectionType type);

    std::vector<TensorCoherence> m_mappedTensors;
    const HabanaGraph&           m_g;
};
