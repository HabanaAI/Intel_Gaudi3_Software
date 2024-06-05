#include "gaudi_memset_scheduling_test_common.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "compilation_hal_reader.h"
#include "hal_reader/gaudi2/hal_reader.h"

class Gaudi2MemsetSchedulingTest : public MemsetSchedulingTest<Gaudi2Graph>
{
};

TEST_F(Gaudi2MemsetSchedulingTest, reduction_memset_scheduling)
{
    CompilationHalReader::setHalReader(Gaudi2HalReader::instance());
    reduction_memset_scheduling();
}
