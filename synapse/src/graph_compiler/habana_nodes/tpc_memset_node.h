#pragma once

#include "habana_nodes.h"
#include "node_visitor.h"
#include "perf_lib_layer_params.h"

class TPCMemsetNode : public TPCNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    virtual pNode clone() const override;
    virtual bool  validateNode() const override;
    virtual bool  isMemset() const override;
    virtual void  setGUID(const StringViewWithHash& guidAndHash) override;

private:
    TPCMemsetNode(const TensorVector&        in,
                  const TensorVector&        out,
                  std::string_view           name,
                  ns_ConstantKernel::Params& params);

    static pNode createNode(const TensorVector& in,
                            const TensorVector& out,
                            UserParams          userParams,
                            std::string_view    guid,
                            std::string_view    name);
};
