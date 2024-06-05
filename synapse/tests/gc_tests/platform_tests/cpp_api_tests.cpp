#include "data_collector.h"
#include "data_provider.h"
#include "infra/gc_base_test.h"
#include "infra/gtest_macros.h"
#include "launcher.h"
#include "node_factory.h"
#include "synapse_common_types.h"

using namespace gc_tests;

GC_TEST_F(SynTrainingCompileTest, DISABLED_create_graph)
{
    syn::Graph invalidGraph;
    EXPECT_FALSE(invalidGraph);
    syn::Graph graph = m_ctx.createGraph(m_deviceType);
    EXPECT_TRUE(graph);
}

GC_TEST_F_INC(SynTrainingCompileTest, DISABLED_create_eager_graph, synDeviceGaudi2)
{
    syn::EagerGraph invalidGraph;
    EXPECT_FALSE(invalidGraph);
    syn::EagerGraph graph = m_ctx.createEagerGraph(m_deviceType);
    EXPECT_TRUE(graph);
}

GC_TEST_F(SynTrainingCompileTest, DISABLED_create_tensor)
{
    syn::Graph graph = m_ctx.createGraph(m_deviceType);

    const std::string name = "DATA_TENSOR";

    syn::Tensor tensor = graph.createTensor(synTensorType::DATA_TENSOR, name);
    EXPECT_EQ(name, tensor.getName());
}