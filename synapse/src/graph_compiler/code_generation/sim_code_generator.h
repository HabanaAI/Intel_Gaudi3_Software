#pragma once

#include "code_generator.h"

class SimCodeGenerator : public CodeGenerator
{
public:
    SimCodeGenerator(HabanaGraph* graph);
    SimCodeGenerator(const SimCodeGenerator& other, HabanaGraph* graph, bool cloneAllocators = false);
    SimCodeGenerator& operator=(const SimCodeGenerator& other);
    CodeGeneratorPtr clone(HabanaGraph* graph, bool cloneAllocators = false) const override;

    virtual ~SimCodeGenerator() = default;

    MemoryAllocator& getWorkspaceAllocator() override { return *m_dramAllocator; }
    MemoryAllocator& getAllocatorForProgramData() override { return *m_dramAllocator; }
    MemoryAllocator& getSramAllocator() override { return *m_sramAllocator; }
    const MemoryAllocator& getWorkspaceAllocator() const override { return *m_dramAllocator; }
    const MemoryAllocator& getAllocatorForProgramData() const override { return *m_dramAllocator; }
    const MemoryAllocator& getSramAllocator() const override { return *m_sramAllocator; }

    void         initAllocators() override { m_allocatorsWereInit = true; }
    unsigned     getQueueID(HabanaDeviceType type, unsigned id) override { return 0; }
    virtual void fillQueuesWithDmaNode(NodePtr node) override {};

    std::unique_ptr<MemoryAllocator> m_dramAllocator = nullptr;
    std::unique_ptr<MemoryAllocator> m_sramAllocator = nullptr;
};