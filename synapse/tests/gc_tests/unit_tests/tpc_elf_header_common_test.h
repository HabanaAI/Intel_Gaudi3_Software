#pragma once

#include "compilation_hal_reader.h"
#include "defs.h"
#include "synapse_common_types.h"
#include "tpc_kernel_lib_interface.h"
#include "types.h"
#include "tpc_node.h"
#include "kernel_instantiation_wrapper.h"
#include "tpc_elf_api.hpp"

class TestParams
{
public:
    struct ElfParams
    {
        unsigned version;
        unsigned specialFunctionUsed;
        bool     unsetSmallVlm;
    };

    TestParams() = delete;

    TestParams(const ElfParams& elfParams, bool smallVlmRequired)
    : m_elfParams(elfParams), m_smallVlmRequired(smallVlmRequired)
    {
    }

    ElfParams m_elfParams;
    bool      m_smallVlmRequired;
};

template<class GraphType>
class TPCCustomElfHeaderNodeCommon : public TPCNode
{
public:

    TPCCustomElfHeaderNodeCommon()
    : TPCNode(TensorVector {}, TensorVector {}, "customGuid"), m_programHeader{0}, m_halSetter(&m_g)
    {
    }

    TPCCustomElfHeaderNodeCommon(TpcElfTools::TPCProgramHeader& programHeader)
    : TPCNode(TensorVector {}, TensorVector {}, "customGuid"), m_programHeader(programHeader), m_halSetter(&m_g)
    {
    }

    tpc_lib_api::GlueCodeReturn instantiate(KernelInstantiationWrapper& instance)
    {
        instance.setProgrammingHeader(m_programHeader);

        m_instanceWrapper = instance;
        return tpc_lib_api::GLUE_SUCCESS;
    }

    static TPCCustomElfHeaderNodeCommon<GraphType>* create(TpcElfTools::TPCProgramHeader& programHeader)
    {
        TPCCustomElfHeaderNodeCommon<GraphType>* node(new TPCCustomElfHeaderNodeCommon<GraphType>(programHeader));
        return node;
    }

    virtual ~TPCCustomElfHeaderNodeCommon() {}

private:
    TpcElfTools::TPCProgramHeader m_programHeader;
    GraphType                     m_g;
    CompilationHalReaderSetter    m_halSetter;
};