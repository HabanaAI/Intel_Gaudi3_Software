#pragma once

#include "synapse_api.h"

#include <vector>
#include "infra/defs.h"

class TestTensorsContainer
{
public:
    TestTensorsContainer() = default;

    TestTensorsContainer(unsigned numOfTensors)
    {
        // Each tensor will have its own section
        m_tensors.resize(numOfTensors);
        m_sections.resize(numOfTensors);
        m_tensorNames.resize(numOfTensors);
    }

    TestTensorsContainer& operator=(TestTensorsContainer const&) = delete;

    ~TestTensorsContainer() { destroy(); }

    void initialize(unsigned numOfTensors)
    {
        if (m_tensors.size() == 0)
        {
            m_tensors.resize(numOfTensors);
            m_sections.resize(numOfTensors);
            m_tensorNames.resize(numOfTensors);
        }
    }

    synTensor& tensor(unsigned tensorIndex) { return m_tensors.at(tensorIndex); }

    synSectionHandle section(unsigned tensorIndex) { return m_sections.at(tensorIndex); }

    size_t size() const { return m_tensors.size(); }

    const synTensor* tensors() const { return m_tensors.data(); }

    const synSectionHandle* sections() const { return m_sections.data(); }

    void
    setTensor(unsigned tensorIndex, const std::string& rTensorName, synTensor tensor, synSectionHandle sectionHandle)
    {
        m_tensors.at(tensorIndex)     = tensor;
        m_sections.at(tensorIndex)    = sectionHandle;
        m_tensorNames.at(tensorIndex) = rTensorName;
    }

    void setSection(unsigned sectionIndex, synSectionHandle sectionHandle)
    {
        m_sections.at(sectionIndex) = sectionHandle;
    }

    void destroy()
    {
        // tensors and sections get destroyed in graph destruction
        m_tensors.clear();
        m_sections.clear();
    }

private:
    std::vector<synTensor>        m_tensors;
    std::vector<synSectionHandle> m_sections;
    std::vector<std::string>      m_tensorNames;
};