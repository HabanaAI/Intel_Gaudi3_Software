#pragma once

#include "cache_types.h"
#include "node.h"
#include "types.h"

namespace gc::layered_brain
{
class NodeAccessCacheSetter
{
public:
    explicit NodeAccessCacheSetter(const NodePtr& node) : m_node(node) {}

    // Configuration
    NodeAccessCacheSetter& input(size_t inputIdx)
    {
        m_index      = inputIdx;
        m_isInputIdx = true;
        return *this;
    }

    NodeAccessCacheSetter& output(size_t outputIdx)
    {
        m_index      = outputIdx;
        m_isInputIdx = false;
        return *this;
    }

    NodeAccessCacheSetter& directive(CacheDirective dir)
    {
        m_directive = dir;
        return *this;
    }

    NodeAccessCacheSetter& cacheClass(CacheClass cls)
    {
        m_class = cls;
        return *this;
    }

    NodeAccessCacheSetter& cmAction(CacheMaintenanceAction cma)
    {
        m_cma = cma;
        return *this;
    }

    NodeAccessCacheSetter& mcid(LogicalMcid mcid)
    {
        m_mcid = mcid;
        return *this;
    }

    // Apply the access settings
    void set()
    {
        HB_ASSERT(m_index, "Directive set of unspecified {} index", m_isInputIdx ? "input" : "output");

        auto& metadata = targetMetaData(*m_index);
        if (m_directive) metadata.cacheDirective = *m_directive;
        if (m_class) metadata.cacheClass = *m_class;
        if (m_cma) metadata.cmAction = *m_cma;
        if (m_mcid) metadata.mcid = *m_mcid;
    }

private:
    NodePtr m_node {};

    std::optional<size_t> m_index {};
    bool                  m_isInputIdx {true};  // if false, index referres to outputs

    std::optional<CacheDirective>         m_directive {};
    std::optional<CacheClass>             m_class {};
    std::optional<CacheMaintenanceAction> m_cma {};
    std::optional<LogicalMcid>            m_mcid {};

    CacheMetaData& targetMetaData(size_t idx)
    {
        if (m_isInputIdx)
        {
            return inputMD(idx);
        }
        else
        {
            return outputMD(idx);
        }
    }
    CacheMetaData& inputMD(size_t inputIdx) { return m_node->getNodeAnnotation().inputsCacheMetaData.at(inputIdx); }
    CacheMetaData& outputMD(size_t outputIdx) { return m_node->getNodeAnnotation().outputsCacheMetaData.at(outputIdx); }
};

}  // namespace gc::layered_brain