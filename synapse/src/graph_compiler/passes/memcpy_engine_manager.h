#pragma once

#include "types.h"
#include "habana_graph.h"

class MemcpyEngineManager
{
public:
    virtual ~MemcpyEngineManager() = default;
    virtual bool selectEngine(HabanaGraph& graph);

protected:
    virtual bool validateMemCopy(const NodePtr& node) const;
    NodePtr createTpcMemset(const TensorVector& inputs, const TensorVector& outputs, const std::string& name) const;

private:
    NodeList        createConcreteNode(const HabanaGraph& graph, const NodePtr& node) const;
    NodeList        lower64bMemcpyTo32b(const NodePtr& semanticNode) const;
    NodeList        createConcreteCopyNode(const HabanaGraph& graph, const NodePtr& semanticNode) const;
    virtual NodePtr createConcreteSetNode(const HabanaGraph& graph, const NodePtr& semanticNode) const;
    virtual NodePtr
    getDefaultCopyNode(const TensorVector& inputs, const TensorVector& outputs, const std::string& name) const;
    virtual NodePtr getDefaultCopyNdNode(const TensorVector& inputs,
                                         const TensorVector& outputs,
                                         const std::string&  name,
                                         synDataType         dtype) const;
    void            updateNodeAnnotation(const NodePtr& semanticNode, const NodeList& copySequence) const;
};
