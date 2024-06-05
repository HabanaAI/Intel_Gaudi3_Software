#pragma once
#include  "gc_protocol.hpp"
class SynapseGraphWrapper;
class MLIRToSynapseTranslator;

enum SynapseMLIROptimizerMode
{
    SynapseMLIROptimizerModeDisabled = 0,
    SynapseMLIROptimizerModeEnabled,
    SynapseMLIROptimizerModeEnabledValidation
};

class SynapseMLIRSharedObject
{
public:
    ~SynapseMLIRSharedObject();
    bool                       init();
    bool                       destroy();
    pfnRunSynapseMLIROptimizer getSynapseMlirFunc() { return m_pSynapseMlirFunc; };

private:
    pfnRunSynapseMLIROptimizer m_pSynapseMlirFunc     = nullptr;
};

class SynapseMLIROptimizer
{
public:
    SynapseMLIROptimizer(HabanaGraph& g) : m_synapseGraphWrapper(g, g.getCompilationMode() == CompilationMode::Eager) {};
    ~SynapseMLIROptimizer() { clearMLIRResources(); };
    bool runSynapseMLIROptimizer(pfnRunSynapseMLIROptimizer synapseMlirEntryPoint);
    bool translateOptimizedGraph();
    bool validateOptimizedGraph();
    // clear resources that were allocated in MLIR side, such as context, block and ops
    void clearMLIRResources();

private:
    // synapseGraphWrapper contains habana graph, and will be passed to synapse_mlir lib
    SynapseGraphWrapper m_synapseGraphWrapper;
    // pMLIRGraphWrapper will be filled by synapse_mlir, will contain MLIR graph (ops and tensors)
    gc_protocol::ProtocolGraph* m_pMlirGraphWrapper = nullptr;
    // mlirToSynapseTranslator will translate MLIR graph that resides in m_synapseGraphWrapper, to Synapse
    std::shared_ptr<IRToSynapseTranslator> m_pMlirToSynapseTranslator = nullptr;
    // used for validations after translation
    HabanaGraphPtr m_originalGraphCopy = nullptr;
};
