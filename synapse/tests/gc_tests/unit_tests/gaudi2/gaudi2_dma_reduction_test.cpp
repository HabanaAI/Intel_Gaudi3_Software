#include "gaudi2_types.h"
#include "synapse_test.hpp"
#include "synapse_api.h"

#include "graph_optimizer_test.h"
#include "node.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "platform/gaudi2/graph_compiler/command_queue.h"
#include "platform/gaudi2/graph_compiler/queue_command.h"
#include "platform/gaudi2/graph_compiler/descriptor_generator.h"
#include "platform/gaudi2/graph_compiler/gaudi2_graph.h"
#include "platform/gaudi2/graph_compiler/gaudi2_code_generator.h"
#include "platform/gaudi2/graph_compiler/block_data.h"
#include "graph_compiler/descriptor_wrapper.h"

using namespace gaudi2;

class Gaudi2DmaDescriptorTest : public GraphOptimizerTest
{
};

class TestQueue : public gaudi2::DmaDescQueue
{
public:
    TestQueue() : gaudi2::DmaDescQueue(0, 0, 0, false) {}

    DescriptorShadow::AllRegistersProperties regProp(NodePtr n, const DescriptorWrapper<DmaDesc>& dw)
    {
        return registersPropertiesForDesc(n, dw);
    }
};

class TestWMR : public gaudi2::WriteManyRegisters
{
public:
    TestWMR() = default;
    uint32_t getValue(unsigned i) { return gaudi2::WriteManyRegisters::getValue(i); }
};

TEST_F(Gaudi2DmaDescriptorTest, dma_desc_with_reduction)
{
#define NO_OF_TESTS (4)
    Gaudi2Graph         graph;
    DescriptorGenerator dg(downcaster<Gaudi2CodeGenerator>(graph.getCodeGenerator().get()));

    const TSize              tensorDims[NO_OF_TESTS] = {5, 5, 5, 5};
    const ReductionOperation redOps[NO_OF_TESTS]     = {REDUCTION_MIN,
                                                    REDUCTION_SET,
                                                    REDUCTION_MAX0,
                                                    ENUM_MAX_REDUCTION_OPERATIONS};
    const synDataType        dataTypes[NO_OF_TESTS]  = {syn_type_float, syn_type_int16, syn_type_uint8, syn_type_float};
    const unsigned           expectedDataTypes[NO_OF_TESTS] = {7, 1, 3, 0};

    for (size_t test = 0; test < NO_OF_TESTS; test++)
    {
        TensorPtr inputTensor(new Tensor(4U, tensorDims, dataTypes[test]));
        inputTensor->setSramOffset(128 * 8);

        TensorPtr outputTensor(new Tensor(4U, tensorDims, dataTypes[test]));
        outputTensor->setDramOffset(128 * 4);
        outputTensor->getTensorAnnotation().tensorReductionInfo.isReductionEnabled =
            ReductionInfo::isRMWReduction(redOps[test]) ? true : false;
        outputTensor->getTensorAnnotation().tensorReductionInfo.reductionOperation = redOps[test];

        std::string nodeName("DMA_node");
        DMANode     dmaNode(inputTensor, outputTensor, nodeName, DMA_TYPE_INTERNAL);
        NodeROI     nodeROI = dmaNode.generateRoi();
        TensorROI   tensorROI;
        for (size_t i = 0; i < 4U; i++)
        {
            tensorROI.m_layout.m_size[i]         = inputTensor->getSizeInElements(i);
            tensorROI.m_layout.spatialStrides[i] = inputTensor->getStrideInBytes(i);
            tensorROI.m_parentTensor             = inputTensor;
        }
        nodeROI.inputRois.push_back(tensorROI);

        for (size_t i = 0; i < 4U; i++)
        {
            tensorROI.m_layout.m_size[i]         = outputTensor->getSizeInElements(i);
            tensorROI.m_layout.spatialStrides[i] = outputTensor->getStrideInBytes(i);
            tensorROI.m_parentTensor             = outputTensor;
        }
        nodeROI.outputRois.push_back(tensorROI);

        std::list<NodeROI> physicalRois;
        physicalRois.push_back(nodeROI);

        gaudi2::DescriptorGenerator::DmaDescriptorsList descriptors;

        ASSERT_NO_THROW(dg.generateDmaDescriptors(dmaNode, physicalRois, descriptors));
        ASSERT_TRUE(descriptors.size() == 1);

        DmaDesc      desc = descriptors.front().desc;
        ValidityMask<DmaDesc> mask = descriptors.front().mask;

        // verify reduction register is masked and reduction parameters are correct
        ASSERT_FALSE(mask[MASK_OFFSET(DmaDesc, axuser.hb_wr_reduction) - 1]);
        ASSERT_TRUE(mask[MASK_OFFSET(DmaDesc, axuser.hb_wr_reduction)]);
        ASSERT_FALSE(mask[MASK_OFFSET(DmaDesc, axuser.hb_wr_reduction) + 1]);

        if (ReductionInfo::isRMWReduction(redOps[test]))
        {
            ASSERT_EQ(desc.axuser.hb_wr_reduction.ind, 1U);
            ASSERT_EQ(desc.axuser.hb_wr_reduction.dtype, expectedDataTypes[test]);
            ASSERT_EQ(desc.axuser.hb_wr_reduction.op, redOps[test] & 0x3);
            ASSERT_EQ(desc.axuser.hb_wr_reduction.max, (redOps[test] == REDUCTION_MAX0) ? 1U : 0U);
        }
        else
        {
            ASSERT_EQ(desc.axuser.hb_wr_reduction.ind, 0U);
            ASSERT_EQ(desc.axuser.hb_wr_reduction.dtype, 0);
            ASSERT_EQ(desc.axuser.hb_wr_reduction.op, 0U);
            ASSERT_EQ(desc.axuser.hb_wr_reduction.max, 0U);
        }

        // verify the other register block is not affected
        ASSERT_EQ(desc.ctx.src_tsize_0.val, inputTensor->getSizeInBytes(0));
        ASSERT_EQ(desc.ctx.src_tsize_1.val, inputTensor->getSizeInElements(1));
        ASSERT_EQ(desc.ctx.src_tsize_2.val, inputTensor->getSizeInElements(2));
        ASSERT_EQ(desc.ctx.src_tsize_3.val, inputTensor->getSizeInElements(3));
        unsigned expectedStride = dataTypeToSizeInBits(dataTypes[test]) / 8;
        ASSERT_EQ(desc.ctx.src_stride_1.val, expectedStride);
        expectedStride *= inputTensor->getSizeInElements(0);
        ASSERT_EQ(desc.ctx.src_stride_2.val, expectedStride);
        expectedStride *= inputTensor->getSizeInElements(1);
        ASSERT_EQ(desc.ctx.src_stride_3.val, expectedStride);
        expectedStride *= inputTensor->getSizeInElements(2);
        ASSERT_EQ(desc.ctx.src_stride_4.val, expectedStride);

        ASSERT_EQ(desc.ctx.dst_tsize_0.val, outputTensor->getSizeInBytes(0));
        ASSERT_EQ(desc.ctx.dst_tsize_1.val, outputTensor->getSizeInElements(1));
        ASSERT_EQ(desc.ctx.dst_tsize_2.val, outputTensor->getSizeInElements(2));
        ASSERT_EQ(desc.ctx.dst_tsize_3.val, outputTensor->getSizeInElements(3));
        expectedStride = dataTypeToSizeInBits(dataTypes[test]) / 8;
        ASSERT_EQ(desc.ctx.dst_stride_1.val, expectedStride);
        expectedStride *= outputTensor->getSizeInElements(0);
        ASSERT_EQ(desc.ctx.dst_stride_2.val, expectedStride);
        expectedStride *= outputTensor->getSizeInElements(1);
        ASSERT_EQ(desc.ctx.dst_stride_3.val, expectedStride);
        expectedStride *= outputTensor->getSizeInElements(2);
        ASSERT_EQ(desc.ctx.dst_stride_4.val, expectedStride);

        TestQueue ddq;

        DescriptorWrapper<DmaDesc> dw(desc, mask);

        DescriptorShadow ds(ddq.regProp(std::make_shared<DMANode>(dmaNode), dw));

        ddq.addLoadDesc(nullptr, DescSection(desc), nullptr, 0, &ds);

        const auto& cmds = ddq.getCommands(false);

        TestWMR* cmd0 = static_cast<TestWMR*>(cmds[0].get());  // reduction reg
        ASSERT_NE(cmd0, nullptr);
        ASSERT_EQ(cmd0->GetFirstReg(),
                  getRegForLoadDesc(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL, 0) +
                      offsetof(block_axuser_dma_core_ctx, axuser.hb_wr_reduction));  // 0xb810
        ASSERT_EQ(cmd0->GetCount(), 1U);
        ASSERT_EQ(cmd0->getValue(0), desc.axuser.hb_wr_reduction._raw);

        TestWMR* cmd1 = static_cast<TestWMR*>(cmds[1].get());
        ASSERT_NE(cmd1, nullptr);
        ASSERT_EQ(cmd1->GetFirstReg(),
                  getRegForLoadDesc(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL, 0) +
                      offsetof(block_axuser_dma_core_ctx, ctx.idx));  // 0xb86c
        ASSERT_EQ(cmd1->GetCount(), 3U);

        TestWMR* cmd2 = static_cast<TestWMR*>(cmds[2].get());
        ASSERT_NE(cmd2, nullptr);
        ASSERT_EQ(cmd2->GetFirstReg(),
                  getRegForLoadDesc(DEVICE_DMA_DRAM_SRAM_BIDIRECTIONAL, 0) +
                      offsetof(block_axuser_dma_core_ctx, ctx.wr_comp_addr_hi));  // 0xb8bc
        ASSERT_EQ(cmd2->GetCount(), 12U);
    }
}
