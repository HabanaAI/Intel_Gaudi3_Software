#pragma once

#include "defs.h"

#include "synapse_api_types.h"
#include "synapse_common_types.h"
#include "filesystem.h"

#include <cstdint>
#include "../utils/gtest_synapse.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

enum class synTestPackage
{
    DEFAULT,
    CI,
    ASIC,
    ASIC_CI,
    SIM,
    DEATH,
    SIZE
};

// Excluding DEATH package
#define ALL_TEST_PACKAGES synTestPackage::CI, synTestPackage::SIM, synTestPackage::ASIC, synTestPackage::ASIC_CI

enum class TestSectionType
{
    NON_CONST_SECTION    = 0,
    CONST_SECTION        = 1,
    CONST_TENSOR_SECTION = 2
};

struct SectionInfo
{
    uint64_t m_sectionSize    = 0;
    bool     m_isConstSection = false;
};
typedef std::vector<SectionInfo> SectionInfoVec;

typedef std::vector<void*> SectionMemoryVec;

template<typename PKT_TYPE, int OP_CODE>
void generatePacketOpCode(uint8_t*& pBuffer)
{
    auto pBinary = reinterpret_cast<PKT_TYPE*>(pBuffer);
    std::memset(pBinary, 0, sizeof(PKT_TYPE));

    pBinary->opcode = OP_CODE;
    pBuffer += sizeof(PKT_TYPE);
}

#define TEST_RESOURCE_PATH std::string(fs::temp_directory_path() / "syn_test_resource/")