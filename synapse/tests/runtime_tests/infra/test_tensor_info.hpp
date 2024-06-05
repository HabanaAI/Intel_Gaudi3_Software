#pragma once

#include "synapse_api_types.h"
#include "../infra/test_types.hpp"
#include <vector>

// Bit-mask
enum class TensorFlag
{
    NONE = 0x0,
    // The Tensor is part of a const-section
    CONST_SECTION = 0x1
};

struct TensorInfo
{
    // Each launch requires the same Tensor's infomation (at least, at the moment)
    synTensorType m_tensorType = DATA_TENSOR;
    synDataType   m_dataType   = syn_type_na;
    unsigned      m_dimsAmount = 0;
    uint64_t      m_tensorSize = 0;
    std::string   m_tensorName = "";
    uint64_t      m_tensorId   = 0;
    bool          m_isConst    = false;
    bool          m_isSfg      = false;

    TSize m_tensorDimsSize[HABANA_DIM_MAX]    = {0};
    TSize m_tensorMinDimsSize[HABANA_DIM_MAX] = {0};

    uint32_t m_tensorFlags = (uint32_t)TensorFlag::NONE;

    // In case there are several tensors in a single section
    unsigned m_sectionIndex  = INVALID_SECTION_ID;
    uint64_t m_sectionOffset = 0;

    TestSectionType m_sectionType = TestSectionType::NON_CONST_SECTION;
};

typedef std::vector<TensorInfo> TensorInfoVec;
