#pragma once

namespace eager_mode
{
class DescGeneratorHal;
class Node2DescContainer;
class SingleNode2Desc;

class SyncSchemeManagerBase
{
public:
    explicit constexpr SyncSchemeManagerBase(const DescGeneratorHal& descGenHal) : m_descGenHal(descGenHal) {}

    void         generateNodesArcSyncScheme(Node2DescContainer& multiNode2Desc) const;
    virtual void generateWorkDistributionContexts(Node2DescContainer& multiNode2Desc) const = 0;

protected:
    void generateNodesArcSyncScheme(SingleNode2Desc& singleNode2Desc) const;

protected:
    const DescGeneratorHal& m_descGenHal;
};

}  // namespace eager_mode