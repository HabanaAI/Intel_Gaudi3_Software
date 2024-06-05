#pragma once

#include "syn_object.hpp"
#include "syn_section.hpp"
#include "syn_tensor.hpp"
#include "syn_node.hpp"
#include "syn_recipe.hpp"

namespace syn
{
class GraphBase : public SynObject<synGraphHandle>
{
public:
    Section createSection() const
    {
        auto handlePtr = createHandle<synSectionHandle>(synSectionDestroy);
        SYN_CHECK(synSectionCreate(handlePtr.get(), 0, *m_handle));
        return Section(handlePtr);
    }

    Tensor createTensor(const synTensorType type, const std::string& name)
    {
        auto handlePtr = createHandle<synTensor>(synTensorDestroy);
        SYN_CHECK(synTensorHandleCreate(handlePtr.get(), handle(), type, name.c_str()));
        return Tensor(handlePtr, type);
    }

    std::vector<synTensor> getTensorsHandles(const Tensors& tensors)
    {
        std::vector<synTensor> handles;
        for (const auto& t : tensors)
        {
            handles.push_back(t ? t.handle() : nullptr);
        }
        return handles;
    }

    Node createNode(const Tensors&                  inputs,
                    const Tensors&                  outputs,
                    const UserParams&               params,
                    const std::string&              guid,
                    const std::string&              name,
                    const std::vector<std::string>& inputLayouts  = {},
                    const std::vector<std::string>& outputLayouts = {})
    {
        auto handlePtr = createHandle<synNodeId>();
        SYN_CHECK(synNodeCreateWithId(handle(),
                                      getTensorsHandles(inputs).data(),
                                      getTensorsHandles(outputs).data(),
                                      inputs.size(),
                                      outputs.size(),
                                      static_cast<const void*>(params.data()),
                                      params.size(),
                                      guid.c_str(),
                                      name.c_str(),
                                      handlePtr.get(),
                                      toConstChar(inputLayouts, true).data(),
                                      toConstChar(outputLayouts, true).data()));
        return Node(handlePtr);
    }

    Node createNode(const Tensors&                  inputs,
                    const Tensors&                  outputs,
                    const void*                     pUserParams,
                    const unsigned                  paramsSize,
                    const std::string&              guid,
                    const std::string&              name,
                    const std::vector<std::string>& inputLayouts  = {},
                    const std::vector<std::string>& outputLayouts = {})
    {
        auto handlePtr = createHandle<synNodeId>();
        SYN_CHECK(synNodeCreateWithId(handle(),
                                      getTensorsHandles(inputs).data(),
                                      getTensorsHandles(outputs).data(),
                                      inputs.size(),
                                      outputs.size(),
                                      pUserParams,
                                      paramsSize,
                                      guid.c_str(),
                                      name.c_str(),
                                      handlePtr.get(),
                                      toConstChar(inputLayouts, true).data(),
                                      toConstChar(outputLayouts, true).data()));
        return Node(handlePtr);
    }

    void setNodeDependency(const Nodes& blocking, const Nodes& blocked)
    {
        SYN_CHECK(synNodeDependencySet(handle(),
                                       getHandles<synNodeId>(blocking).data(),
                                       getHandles<synNodeId>(blocked).data(),
                                       blocking.size(),
                                       blocked.size()));
    }

    Recipe compile(const std::string& recipeName, const std::string& pBuildLog = "") const
    {
        auto handlePtr = createHandle<synRecipeHandle>(synRecipeDestroy);
        SYN_CHECK(synGraphCompile(handlePtr.get(), handle(), recipeName.c_str(), nullptr));
        return Recipe(handlePtr);
    }

    void setNodePrecision(const std::string& nodeGuid, synDataType precision)
    {
        SYN_CHECK(synNodeTypeSetPrecision(handle(), nodeGuid.c_str(), precision));
    }

    synDeviceType getDeviceType() const
    {
        synDeviceType deviceType;
        SYN_CHECK(synGraphGetDeviceType(handle(), &deviceType));
        return deviceType;
    }

    void setDeterministic(const Node& node, bool deterministic)
    {
        SYN_CHECK(synNodeSetDeterministic(handle(), node.getId(), deterministic));
    }

    bool getDeterministic(const Node& node)
    {
        bool deterministic = false;
        SYN_CHECK(synNodeGetDeterministic(handle(), node.getId(), &deterministic));
        return deterministic;
    }

    void setRoundingMode(const Node& node, synRoundingMode roundingMode)
    {
        SYN_CHECK(synNodeSetRoundingMode(handle(), node.getId(), roundingMode));
    }

    /*DEPRECATED*/
    void setGraphAttributes(const std::vector<synGraphAttribute>& attributes,
                            const std::vector<uint64_t>&          values,
                            uint32_t                              size)
    {
        SYN_CHECK(synGraphSetAttribute(handle(), attributes.data(), values.data(), size));
    }

    void setGraphAttributesV2(const std::vector<synGraphAttribute>&    attributes,
                              const std::vector<synGraphAttributeVal>& values,
                              uint32_t                                 size)
    {
        SYN_CHECK(synGraphSetAttributes(handle(), attributes.data(), values.data(), size));
    }


protected:
    GraphBase() = default;
    GraphBase(const std::shared_ptr<synGraphHandle>& handle) : SynObject(handle) {}
};

class Graph : public GraphBase
{
public:
    Graph() = default;

private:
    Graph(synDeviceType type) : GraphBase(createHandle<synGraphHandle>(synGraphDestroy))
    {
        SYN_CHECK(synGraphCreate(handlePtr(), type));
    }

    friend class Context;  // Context class requires access to Graph private constructor
};

class EagerGraph : public GraphBase
{
public:
    EagerGraph() = default;

private:
    EagerGraph(synDeviceType type) : GraphBase(createHandle<synGraphHandle>(synGraphDestroy))
    {
        SYN_CHECK(synGraphCreateEager(handlePtr(), type));
    }

    friend class Context;  // Context class requires access to EagerGraph private constructor
};
}  // namespace syn
