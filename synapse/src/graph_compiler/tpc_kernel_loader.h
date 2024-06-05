#pragma once

#include <memory>
#include "synapse_common_types.h"
#include "types.h"
#include "infra/settable.h"
#include "passes/cast_nodes_handler.h"

tpc_lib_api::DeviceId deviceTypeToDeviceID(synDeviceType deviceType);

class HabanaGraph;

class TpcKernelLoader
{
public:
    explicit TpcKernelLoader(HabanaGraph* graph);

    virtual ~TpcKernelLoader();

    /**
     * Try recover from tpc kernel initialization error
     * Return false if can't recover
     */
    virtual bool tryRecover(const pNode& node, tpc_lib_api::GlueCodeReturn initRes);

    /**
     * Call after all tpc nodes were initialized
     * Return true if tpc nodes initialization should be triggered once more
     */
    virtual bool finalizeRecover();

    bool load();
    bool load(pNode& node);
    bool allocate(bool bAddNOPKernel = false);
    bool allocate(TPCNode& node);

protected:
    virtual Settable<deviceAddrOffset> allocateKernelWorkspace(unsigned requiredSize);
    virtual Settable<deviceAddrOffset> allocatePrintfWorkspace(unsigned requiredSize);
    virtual void setPrintfTensorOffset(Tensor& printfTensor, deviceAddrOffset addr);

    bool _isLastNode(const std::shared_ptr<Node>& node) const;

    bool _loadFullGraph();
    bool _loadSingleNode(pNode& node);
    bool _loadAndRecover(pNode& node);

    bool _allocateFullGraph(bool bAddNOPKernel = false);
    bool _allocateSingleNode(TPCNode& node);
    bool _addNOPKernel();

    bool         _adaptNodeOutputTensor(const pNode& node) const;
    virtual void _reset();

    CastNodeHandler       m_castNodeHandler;
    HabanaGraph*          m_graph;
    std::shared_ptr<char> m_printfBuffer;
    bool                  m_reRunTpcInit;
};
