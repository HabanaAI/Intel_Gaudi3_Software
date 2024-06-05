#include <cstring>
#include <generic_parameters_node.h>

GenericParametersNode::GenericParametersNode(const TensorVector& inputs,
                                             const TensorVector& outputs,
                                             std::string_view    name,
                                             eNodeType           type,   /* = DEBUG */
                                             UserParams          params, /* = nullptr */
                                             unsigned            paramsSize /* = 0 */,
                                             bool                createNodeIoManager /* = true */)
: Node(inputs, outputs, name, type, SHAPE_FUNC_MAX_ID, createNodeIoManager), m_params(nullptr, nullptr)
{
    setParams(params, paramsSize);
}

void GenericParametersNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    storeParamsInBuffer(userParams, userParamsSize);
}

GenericParametersNode::~GenericParametersNode()
{
}

GenericParametersNode::GenericParametersNode(const GenericParametersNode& other) : Node(other), m_params(nullptr, nullptr)
{
    setParams(other.getParams(), other.m_paramsSize);
}

GenericParametersNode& GenericParametersNode::operator=(const GenericParametersNode& other)
{
    Node::operator=(other);
    storeParamsInBuffer(other.getParams(), other.m_paramsSize);
    return *this;
}

static void dummy_deleter(void*) {}
static void array_deleter(void* ptr) { delete[] static_cast<char* >(ptr); }

void GenericParametersNode::storeParamsInBuffer(UserParams params, unsigned paramsSize)
{
    if (paramsSize == 0)
    {
        // We do not own the parameters, just store a pointer
        m_params = ParamsBufferType(params, dummy_deleter);
    }
    else
    {
        // We own the parameters, make a local copy
        void* local_copy = new unsigned char[paramsSize];
        std::memcpy(local_copy, params, paramsSize);
        m_params = ParamsBufferType(local_copy, array_deleter);
    }
    m_paramsSize = paramsSize;
}

UserParams GenericParametersNode::getParams() const
{
    return static_cast<UserParams>(m_params.get());
}

unsigned GenericParametersNode::getParamsSize() const
{
   return m_paramsSize;
}

void GenericParametersNode::printParamsRawData() const
{
    if (m_paramsSize > 0)
    {
        BaseClass::printParamsRawData(m_params.get(), m_paramsSize);
    }
}

SifNodeParams GenericParametersNode::getShapeInferenceFunctionUserParams()
{
    return static_cast<SifNodeParams>(getParams());
}

size_t GenericParametersNode::getShapeInferenceFunctionUserParamsSize() const
{
    return getParamsSize();
}