#pragma once

#include <vector>
#include "recipe.h"
#include "types.h"
#include "tensor_shape.h"

struct MultiSifTensorInfo
{
    bool m_isInternal;

    // for internal tensors, just a unique internal index
    // for external tensors, the index of the fused node input or output
    unsigned m_index;
    // Shape (for internal tensors)
    // (needed to pass to the SIF)
    TensorShape m_shape;

    // This input tensor is taken from fused node output and not from input
    bool m_takeFromOutput = false;
};

struct MultiSifSingleNodeInfo
{
    std::vector<MultiSifTensorInfo> m_inputs;
    std::vector<MultiSifTensorInfo> m_outputs;
    std::vector<unsigned char>      m_nodeParams;
    sm_function_id_t                m_sifID;
    uint64_t                        m_sifVersion;
    std::string                     m_nodeName;
    std::string                     m_nodeGUID;

    std::vector<tpc_lib_api::NodeTensorPermutation> m_inputPermutations;
    std::vector<tpc_lib_api::NodeTensorPermutation> m_outputPermutations;
};

struct MultiSifNodeInfo
{
    std::vector<MultiSifSingleNodeInfo> m_nodes;
    unsigned                            m_internalTensorsNr;
};

tpc_lib_api::GlueCodeReturn multiSifRun(synDeviceType     deviceType,
                                        MultiSifNodeInfo* multiSifNodeInfo,
                                        SifParams*        sifParams,
                                        SifOutputs*       sifOutputs,
                                        bool              inferMax);

class MultiSifNodeInfoHelper
{
public:
    MultiSifNodeInfoHelper() {}

    void setMultiSifInfo(std::shared_ptr<MultiSifNodeInfo> multiSifInfoPtr)
    {
        m_multiSifNodeInfo = multiSifInfoPtr;
    }

    std::shared_ptr<MultiSifNodeInfo> getMultiSifInfo() const
    {
        return m_multiSifNodeInfo;
    }

protected:
    std::shared_ptr<MultiSifNodeInfo> m_multiSifNodeInfo;
};