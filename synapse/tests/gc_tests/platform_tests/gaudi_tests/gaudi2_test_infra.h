#pragma once

#include "infra/gc_synapse_test.h"
#include <tensor.h>

#include "test_types.hpp"

class Gaudi2Graph;

static const size_t                DEFAULT_SIZES = 4;
static const std::vector<unsigned> DEFAULT_DIMS  = {16, 16, 16, 16};

typedef std::vector<synQuantizationParams> QuantParamsVector;

class SynGaudi2TestInfra : public SynTest
{
protected:
    typedef size_t GRAPH_INDEX;
    typedef size_t TENSOR_INDEX;

    SynGaudi2TestInfra();
    ~SynGaudi2TestInfra() override = default;

    static const TENSOR_INDEX INVALID_TENSOR_INDEX;
    static const std::string  INVALID_RECIPE_FILE_NAME;
    static const std::string  FILE_NAME_SUFFIX;

private:
    struct TensorInfo
    {
        synTensor         m_tensor;
        char*             m_data;
        size_t            m_dataSize;
        const synDataType m_dataType;
        const std::string m_name;
        const TensorUsage m_usage;
        const bool        m_isWeights;
        const bool        m_isStatic;
        const size_t      m_elementsNum;
        const unsigned    m_sectionIndex;
    };

    const unsigned INVALID_SECTION = std::numeric_limits<unsigned>::max();

    std::vector<TensorInfo> m_tensors;
    synStreamHandle         m_streamHandleDownload;
    synStreamHandle         m_streamHandleCompute;
    synStreamHandle         m_streamHandleUpload;
    synEventHandle          m_eventHandle;
    bool                    m_isNewApiMode;
};
