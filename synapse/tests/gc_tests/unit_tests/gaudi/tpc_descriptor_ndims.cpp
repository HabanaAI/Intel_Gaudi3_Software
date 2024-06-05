#include "compilation_hal_reader.h"
#include "defs.h"
#include "math_utils.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "platform/gaudi/graph_compiler/gaudi_graph.h"
#include "platform/gaudi/graph_compiler/passes.h"
#include "synapse_common_types.h"
#include "tensor.h"
#include "tpc_kernel_loader.h"
#include "transpose_node.h"
#include "node.h"
#include "transpose_utils.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include <algorithm>
#include "platform/gaudi/graph_compiler/descriptor_generator.h"
#include "types.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "gaudi_code_generator.h"

namespace gaudi
{
class TpcTest
: public GraphOptimizerTest
, public testing::WithParamInterface<int>
{
    void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }
};

TEST_P(TpcTest, ndims)
{
    GaudiGraph      g;
    TpcKernelLoader loader(&g);
    NSizeArray      sizes;
    auto            dims = GetParam();
    for (int i = 0; i < dims; i++)
    {
        sizes[i] = i % 4 + 1;
    }

    TensorPtr in   = std::make_shared<Tensor>(dims, sizes.data(), syn_type_uint16);
    TensorPtr out  = std::make_shared<Tensor>(dims, sizes.data(), syn_type_uint8);
    auto      node = NodeFactory::createNode({in}, {out}, nullptr, NOP_KERNEL_NAME, "cast");
    GraphEditor::addNode(g, node);
    loader.load(node);

    auto               tpcNode = static_cast<TPCNode*>(node.get());
    auto               roi     = node->generateRoi();
    std::list<NodeROI> rois {roi};

    std::list<DescAndMask<gaudi::TpcDesc>>             output;
    CompilationHalReaderSetter                         compHalReaderSetter(&g);
    DescriptorGenerator::generateTpcDescriptors(*tpcNode, rois, 0, output);
    ASSERT_EQ(output.size(), 1);
    auto&            desc = output.begin()->first;
    std::vector<int> sizesDesc;
    std::vector<int> stridesDesc;
    for (int i = 0; i < div_round_up(dims, gaudi::TpcDesc::c_max_tensor_dims); i++)
    {
        sizesDesc.push_back(desc.m_tensors[i].dim_0_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_1_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_2_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_3_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_4_size.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_0_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_1_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_2_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_3_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_4_stride.v);
    }
    for (int i = 0; i < dims; i++)
    {
        EXPECT_EQ(sizesDesc[i], sizes[i]);
        EXPECT_EQ(stridesDesc[i], in->getStrideInElements(i)) << i;
    }
}

TEST_P(TpcTest, full_compile)
{
    GaudiGraph      g;
    TpcKernelLoader loader(&g);
    NSizeArray      sizes;
    auto            dims = GetParam();
    for (int i = 0; i < dims; i++)
    {
        sizes[i] = i % 3 + 1;
    }
    synMemoryDescriptor persistentMemoryDesc(true);
    TensorPtr           in = std::make_shared<Tensor>(dims, sizes.data(), syn_type_uint16);

    in->setTensorInSram();
    in->setMemoryDescriptor(persistentMemoryDesc);
    in->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1);
    TensorPtr out = std::make_shared<Tensor>(dims, sizes.data(), syn_type_uint8);
    out->setTensorInSram();
    out->setMemoryDescriptor(persistentMemoryDesc);
    out->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR);
    auto node = NodeFactory::createNode({in}, {out}, nullptr, NOP_KERNEL_NAME, "cast");
    GraphEditor::addNode(g, node);
    loader.load(node);

    g.compile();
    auto nodes = g.getExeSortedNodes();
    ASSERT_EQ(nodes.size(), 1);

    auto tpcDescriptors = downcaster<GaudiCodeGenerator>(g.getCodeGenerator().get())->getTPCNodeDescriptorsWrappers(nodes.front());
    ASSERT_EQ(tpcDescriptors.size(), 1);
    auto&            desc = tpcDescriptors.begin()->getDescriptor();
    std::vector<int> sizesDesc;
    std::vector<int> stridesDesc;
    for (int i = 0; i < div_round_up(dims, gaudi::TpcDesc::c_max_tensor_dims); i++)
    {
        sizesDesc.push_back(desc.m_tensors[i].dim_0_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_1_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_2_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_3_size.v);
        sizesDesc.push_back(desc.m_tensors[i].dim_4_size.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_0_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_1_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_2_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_3_stride.v);
        stridesDesc.push_back(desc.m_tensors[i].dim_4_stride.v);
    }
    for (int i = 0; i < dims; i++)
    {
        EXPECT_EQ(sizesDesc[i], sizes[i]);
        EXPECT_EQ(stridesDesc[i], in->getStrideInElements(i)) << i;
    }
}
INSTANTIATE_TEST_SUITE_P(, TpcTest, ::testing::Range(6, HABANA_DIM_MAX));
}  // namespace gaudi