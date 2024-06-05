#pragma once

#include "synapse_api.h"
#include "../infra/test_types.hpp"

#include <vector>
#include "infra/defs.h"

class TestSectionsContainer
{
public:
    TestSectionsContainer() = default;
    TestSectionsContainer(unsigned numOfSections) { m_sections.resize(numOfSections); }

    ~TestSectionsContainer() { destroy(); }

    void initialize(unsigned numOfSections)
    {
        if (m_sections.size() == 0)
        {
            m_sections.resize(numOfSections);
        }
    }

    void setSection(unsigned sectionIndex, synSectionHandle sectionHandle)
    {
        m_sections.at(sectionIndex) = sectionHandle;
    }

    synSectionHandle& section(unsigned tensorIndex) { return m_sections.at(tensorIndex); }

    void destroy()
    {
        for (unsigned i = 0; i < m_sections.size(); ++i)
        {
            if (section(i) != nullptr)
            {
                synStatus status = synSectionDestroy(section(i));
                ASSERT_EQ(status, synSuccess) << "Failed to synSectionDestroy i = " << i;
            }
        }
        m_sections.clear();
    }

private:
    std::vector<synSectionHandle> m_sections;
};
