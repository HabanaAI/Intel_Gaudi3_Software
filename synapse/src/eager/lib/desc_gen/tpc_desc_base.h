#pragma once

// eager includes (relative to src/eager/lib/)
#include "desc_gen/desc_base.h"

class TPCNode;

namespace eager_mode
{
class EagerGraph;

class TpcDescGeneratorBase : public DescGeneratorBase
{
public:
    TpcDescGeneratorBase(EagerGraph& graph, const EagerNode& node) : DescGeneratorBase(graph, node) {}
    TPCNode&      getNode() const { return *m_node.get<TPCNode>(); }
    static size_t calcNumberPatchableTensors(const EagerNode& node);
    bool          generateDesc() override final;

private:
    inline void splitTpcNodeDims();

protected:
    virtual bool       generateTpcDesc() = 0;
    std::list<NodeROI> m_rois;
};

}  // namespace eager_mode
