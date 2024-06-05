#pragma once
#include "synapse_api.h"
#include <vector>
#include <list>
#include "assert.h"
#include "string.h"
#include <memory>
#include <functional>

class SynGaudiAutoGenTest;

typedef std::vector<unsigned> sizeVector;

class ManagedTensor;
typedef std::shared_ptr<ManagedTensor> pManagedTensor;

class ManagedTensor
{
public:
    ManagedTensor(const std::string& tensorName, const sizeVector& sizes,
                  synDataType type, bool isPersistent, SynGaudiAutoGenTest* pTest);
    ~ManagedTensor();
    static pManagedTensor createManagedTensor(const std::string& tensorName, const sizeVector& sizes,
                                              synDataType type, bool isPersistent, SynGaudiAutoGenTest* pTest);

    synTensor getTensorHandle() const;
    unsigned getDataTypeSize() const;
    uint64_t getTotalSizeInBytes() const;
    uint64_t getNumberOfElements() const;
    const std::string& getName() const;
    uint64_t getDramAddress() const;
    synDataType getDataType() const;
    sizeVector getSizes() const;

private:
    std::string m_tensorName;
    sizeVector m_sizes;
    synDataType m_type;
    bool m_isPersistent = false;
    uint64_t m_dramAddress = 0;
    synTensor m_tensorHandle = nullptr;
    synSectionHandle m_sectionHandle = nullptr;
    SynGaudiAutoGenTest* m_pTest;

};


class ManagedNode
{
public:
    ManagedNode(const std::list<pManagedTensor>& inputs, const std::list<pManagedTensor>& outputs,
                const std::string& nodeName, const std::string& operationName, synGraphHandle handle)
                : m_inputs(inputs)
                , m_outputs(outputs)
                , m_nodeName(nodeName)
                , m_operationName(operationName)
                , m_handle(handle)
    {
    }
    virtual ~ManagedNode() = default;

    virtual void createNode();

protected:
    void createSynapseNode(void* params, unsigned int sizeParams);

private:
    std::list<pManagedTensor> m_inputs;
    std::list<pManagedTensor> m_outputs;
    std::string m_nodeName;
    std::string m_operationName;
    synGraphHandle m_handle;
};

template<class T_PARAM>
class ManagedNodeWithParams : public ManagedNode
{
public:
    ManagedNodeWithParams(const std::list<pManagedTensor>& inputs, const std::list<pManagedTensor>& outputs,
                const std::string& nodeName, const std::string& operationName, synGraphHandle handle,
                const T_PARAM nodeParams = T_PARAM())
                : ManagedNode(inputs, outputs, nodeName, operationName, handle)
                , m_params(nodeParams)
    {
    }
    virtual ~ManagedNodeWithParams() {};

    void createNode() override
    {
        createSynapseNode((void*)&m_params, sizeof(T_PARAM));
    }
private:
    T_PARAM m_params;
};
