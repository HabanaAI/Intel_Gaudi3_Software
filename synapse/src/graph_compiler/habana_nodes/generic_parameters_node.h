#ifndef _GENERIC_PARAMETERS_NODE_
#define _GENERIC_PARAMETERS_NODE_

#include <memory>
#include "node.h"

// This class only implements generic parameters storage.
// It should not be used directly, but only inherited from.
// Hence all constructors are protected.
class GenericParametersNode : public Node
{
protected:
    using BaseClass = Node;

    GenericParametersNode(const TensorVector& inputs,
                          const TensorVector& outputs,
                          std::string_view    name,
                          eNodeType           type                = TYPE_DEBUG,
                          UserParams          params              = nullptr,
                          unsigned            paramsSize          = 0,
                          bool                createNodeIoManager = true);

    GenericParametersNode(const GenericParametersNode& other);

    virtual ~GenericParametersNode();

    GenericParametersNode& operator=(const GenericParametersNode& other);

public:
    UserParams          getParams() const;
    unsigned            getParamsSize() const;
    virtual void        storeParamsInBuffer(UserParams params, unsigned size);
    void                printParamsRawData() const override;

    SifNodeParams getShapeInferenceFunctionUserParams() override;
    size_t        getShapeInferenceFunctionUserParamsSize() const override;

    virtual void setParams(UserParams userParams, unsigned int userParamsSize) override;

private:
    using ParamsBufferType = std::unique_ptr<void, void (*)(void*)>;
    ParamsBufferType m_params;
    unsigned         m_paramsSize;
};

#endif  // _GENERIC_PARAMETERS_NODE_
