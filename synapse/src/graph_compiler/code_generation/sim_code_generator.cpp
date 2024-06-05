
#include "sim_code_generator.h"
#include "code_generation/code_generator_factory.h"

CodeGeneratorPtr CodeGeneratorFactory::instantiateSimCodeGenerator(HabanaGraph* graph)
{
    return std::make_unique<SimCodeGenerator>(graph);
}

SimCodeGenerator::SimCodeGenerator(HabanaGraph* graph) : CodeGenerator(graph)
{
    m_dramAllocator = createAllocator(MEMORY_SLAB_ALLOCATOR, "DRAM_SIM");
}

SimCodeGenerator::SimCodeGenerator(const SimCodeGenerator& other, HabanaGraph* graph, bool cloneAllocators /*false*/)
: CodeGenerator(other, graph), m_dramAllocator {createAllocator(MEMORY_SLAB_ALLOCATOR, "DRAM_SIM")}
{

}

SimCodeGenerator& SimCodeGenerator::operator=(const SimCodeGenerator& other)
{
    if(this != &other)
    {
        CodeGenerator::operator=(other);
        SimCodeGenerator tmp(other, other.m_graph);
        std::swap(m_dramAllocator, tmp.m_dramAllocator);
        std::swap(m_sramAllocator, tmp.m_sramAllocator);
    }
    return *this;
}

CodeGeneratorPtr SimCodeGenerator::clone(HabanaGraph* graph, bool cloneAllocators /*false*/) const
{
    return CodeGeneratorPtr(new SimCodeGenerator(*this, graph, cloneAllocators));
}
