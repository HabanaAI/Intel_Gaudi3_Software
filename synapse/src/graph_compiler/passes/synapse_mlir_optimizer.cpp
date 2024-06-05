#include "gaudi_graph.h"
#include "graph_comparator.hpp"
#include "habana_global_conf.h"
#include "habana_pass.h"
#include "ir_translation/ir_to_synapse_translator.hpp"
#include "ir_translation/synapse_graph_wrapper.hpp"
#include "synapse_mlir_optimizer.h"
#include "utils.h"

const std::string synapseMLIRLibName           = "libSynapseMLIR.so";
const std::string synapseMLIROptimizerFuncName = RUN_SYNAPSE_MLIR_OPTIMIZER_ENTRY_POINT_NAME;

SynapseMLIRSharedObject::~SynapseMLIRSharedObject()
{
    // TODO remove explict call for destroy once the synapse_mlir SO is initialized at graph/synapse init
    destroy();
}

bool SynapseMLIRSharedObject::init()
{
    libHandle handle = LoadSharedObject(synapseMLIRLibName.c_str());
    if (!handle)
    {
        LOG_WARN(GC_TRANSLATION, "Failed to load {} shared object", synapseMLIRLibName);
        return false;
    }

    fnHandle funcHandle = GetFunction(handle, synapseMLIROptimizerFuncName.c_str());
    if (!funcHandle)
    {
        LOG_WARN(GC_TRANSLATION, "Failed to load {} function at {} shared object", synapseMLIROptimizerFuncName,
                 synapseMLIRLibName);
        return false;
    }
    m_pSynapseMlirFunc = (pfnRunSynapseMLIROptimizer)funcHandle;
    return true;
}

bool SynapseMLIRSharedObject::destroy()
{
    m_pSynapseMlirFunc = nullptr;
    void* handle       = dlopen(synapseMLIRLibName.data(), RTLD_LAZY | RTLD_NOLOAD);
    if (handle)
    {
        UnloadSharedObject(handle);
    }
    return true;
}

bool SynapseMLIROptimizer::runSynapseMLIROptimizer(pfnRunSynapseMLIROptimizer synapseMlirEntryPoint)
{
    if (!synapseMlirEntryPoint)
    {
        LOG_WARN(GC_TRANSLATION, "Failed to get function pointer from shared object instance");
        return false;
    }
    LOG_INFO(GC_TRANSLATION, "Calling Synapse_MLIR optimizations with function {}", synapseMLIROptimizerFuncName);
    // Calling synapse_mlir lib flow
    gc_protocol::SynMLIRReturnCode_t mlirRetVal = synapseMlirEntryPoint(&m_synapseGraphWrapper, &m_pMlirGraphWrapper);
    LOG_DEBUG(GC_TRANSLATION, "Return value of {} is : {}", synapseMLIROptimizerFuncName, mlirRetVal);

    if (mlirRetVal != gc_protocol::SYN_MLIR_SUCCESS)
    {
        LOG_WARN(GC_TRANSLATION, "Synapse MLIR optimizations failed with return code {}", mlirRetVal);
        return false;
    }
    if (!m_pMlirGraphWrapper)
    {
        LOG_WARN(GC_TRANSLATION, "Synapse MLIR returned a null graph wrapper");
        return false;
    }
    return true;
}

bool SynapseMLIROptimizer::translateOptimizedGraph()
{
    m_pMlirToSynapseTranslator = std::make_shared<IRToSynapseTranslator>(*m_pMlirGraphWrapper);
    if (GCFG_SYNAPSE_MLIR_MODE.value() == SynapseMLIROptimizerModeEnabledValidation)
    {
        m_originalGraphCopy = m_synapseGraphWrapper.getSynapseGraph().clone();
    }
    // Translate MLIR Ops and tensors to Synapse Nodes & tensors
    bool retVal = m_pMlirToSynapseTranslator->startTranslationToSynapse(&m_synapseGraphWrapper.getSynapseGraph());
    LOG_INFO(GC_TRANSLATION, "Finished translation from MLIR to Synapse");
    if (!retVal)
    {
        LOG_WARN(GC_TRANSLATION, "Translation from MLIR to Synapse failed");
        return false;
    }
    return true;
}

bool SynapseMLIROptimizer::validateOptimizedGraph()
{
    LOG_DEBUG(GC_TRANSLATION, "Validating correctness of optimized graph");
    HB_ASSERT(m_originalGraphCopy != nullptr, "original graph copy is null");
    GraphComparator graphComparator;
    return graphComparator.compareGraphs(m_synapseGraphWrapper.getSynapseGraph(), *m_originalGraphCopy);
}

void SynapseMLIROptimizer::clearMLIRResources()
{
    // invokes MLIRGraphWrapper destructor in synapse_mlir
    if (m_pMlirGraphWrapper != nullptr)
    {
        delete(m_pMlirGraphWrapper);
        m_pMlirGraphWrapper = nullptr;
    }
}
/*
 * Call optimization passes from synapse_mlir lib.
 * After optimizations have finished - update the synapse graph with the changes (currently WIP)
 */
bool synapseMLIROptimizer(HabanaGraph& g)
{
    if (GCFG_SYNAPSE_MLIR_MODE.value() == SynapseMLIROptimizerModeDisabled)
    {
        LOG_TRACE(GC_TRANSLATION, "Synapse mlir optimizer is disabled, not running");
        return true;
    }
    if (g.isEmpty())
    {
        LOG_TRACE(GC_TRANSLATION, "Synapse graph is empty, not running synapse mlir optimizer");
        return true;
    }
    SynapseMLIRSharedObject sharedObject;
    if (!sharedObject.init()) return false;

    SynapseMLIROptimizer       optimizer(g);
    pfnRunSynapseMLIROptimizer func = sharedObject.getSynapseMlirFunc();
    if (!optimizer.runSynapseMLIROptimizer(func))
    {
        return false;
    }
    if (!optimizer.translateOptimizedGraph())
    {
        return false;
    }
    // release resources allocated by MLIR lib as they aren't needed anymore
    optimizer.clearMLIRResources();

    if (GCFG_SYNAPSE_MLIR_MODE.value() == SynapseMLIROptimizerModeEnabledValidation &&
        !optimizer.validateOptimizedGraph())
    {
        LOG_WARN(GC_TRANSLATION, "Optimized graph validation failed");
        return false;
    }
    // TODO SW-90490 infer which nodes need to be replaced and perform the replacement, this can also be done within
    // mlirToSynapseTranslator

    sharedObject.destroy();
    return true;
}
