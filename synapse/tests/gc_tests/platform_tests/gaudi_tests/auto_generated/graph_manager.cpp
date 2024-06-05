#include "graph_manager.h"
#include "../gc_autogen_test.h"

ManagedTensor::ManagedTensor(const std::string& tensorName, const sizeVector& sizes, synDataType type, bool isPersistent, SynGaudiAutoGenTest* pTest) :
                             m_tensorName(tensorName), m_sizes(sizes), m_type(type), m_isPersistent(isPersistent), m_pTest(pTest)
{
    if (m_isPersistent)
    {
        if (pTest->hbmAlloc(getTotalSizeInBytes(), &m_dramAddress, m_tensorName.c_str()) != synSuccess)
        {
            std::string messageError = "Allocator Error at tensor ";
            messageError += m_tensorName;
            throw std::runtime_error(messageError);
        }
    }
    m_tensorHandle = pTest->createTensor(m_sizes.size(), m_type, m_sizes.data(), m_isPersistent, m_tensorName.c_str());
    if (m_isPersistent)
    {
        uint64_t byteOffset = 0;
        synTensorGetSection(m_tensorHandle, &m_sectionHandle, &byteOffset);
    }
}

ManagedTensor::~ManagedTensor()
{
    // for persistent tensors we associate them to a graph upon creation.
    // so the responsibility for the destruction is on synGraphDestroy.
    if (!m_isPersistent)
    {
        synTensorDestroy(m_tensorHandle);
    }
    else
    {
        synSectionDestroy(m_sectionHandle);
    }
    m_pTest->hbmFree(getDramAddress(), getName().c_str());
}

pManagedTensor ManagedTensor::createManagedTensor(const std::string& tensorName, const sizeVector& sizes,
                                   synDataType type, bool isPersistent, SynGaudiAutoGenTest* pTest)
{
    return std::make_shared<ManagedTensor>(tensorName, sizes, type, isPersistent, pTest);
}

unsigned ManagedTensor::getDataTypeSize() const
{
    switch (m_type)
    {
        case syn_type_fixed:
        case syn_type_uint8:
            return 1;

        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
            return 2;

        case syn_type_single:
        case syn_type_int32:
        case syn_type_uint32:
            return 4;

        default:
            assert(0);
            return 0;

    }
}

uint64_t ManagedTensor::getNumberOfElements() const
{
    unsigned returnValue = 1;
    for (unsigned elements : m_sizes)
    {
        returnValue *= elements;
    }
    return returnValue;
}

uint64_t ManagedTensor::getTotalSizeInBytes() const
{
    return getNumberOfElements() * getDataTypeSize();
}

synTensor ManagedTensor::getTensorHandle() const
{
    return m_tensorHandle;
}


synDataType ManagedTensor::getDataType() const
{
    return m_type;
}

uint64_t ManagedTensor::getDramAddress() const
{
    return m_dramAddress;
}

sizeVector ManagedTensor::getSizes() const
{
    return m_sizes;
}

const std::string& ManagedTensor::getName() const
{
    return m_tensorName;
}

void ManagedNode::createNode()
{
    createSynapseNode(nullptr, 0);
}

void ManagedNode::createSynapseNode(void* params, unsigned int sizeParams)
{
    std::vector<synTensor> inputArray;

    for (const auto& managedTensor : m_inputs)
    {
        inputArray.push_back(managedTensor->getTensorHandle());
    }

    std::vector<synTensor> outputArray;
    for (const auto& managedTensor : m_outputs)
    {
        outputArray.push_back(managedTensor->getTensorHandle());
    }
    synStatus status;
    status = synNodeCreate(m_handle, inputArray.data(), outputArray.data(), m_inputs.size(), m_outputs.size(),
                           params, sizeParams, m_operationName.c_str(), m_nodeName.c_str(), nullptr, nullptr);

    if (status != synSuccess)
    {
        std::string errorMessage = "Failed to create node ";
        errorMessage += m_nodeName;
        throw (std::runtime_error(errorMessage));
    }
}
