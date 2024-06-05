#pragma once

#include "graph_compiler/types.h"
#include "graph_optimizer_test.h"
#include "define.hpp"

class Gaudi3PipelineManagementTest : public GraphOptimizerTest
{
protected:
    void SetUp() override;
    void TearDown() override;

    // create an arbitrary tensor
    pTensor createTensor(std::vector<TSize> shape, synDataType dataType, bool isPersistent = true);

private:
    unsigned m_memorySectionId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;
};
