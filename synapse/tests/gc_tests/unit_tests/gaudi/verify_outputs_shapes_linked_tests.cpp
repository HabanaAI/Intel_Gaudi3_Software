#include "compilation_hal_reader.h"
#include "gaudi_graph.h"
#include "graph_optimizer_test.h"
#include "node_factory.h"
#include "hal_reader/gaudi1/hal_reader.h"
#include "tensor.h"
#include <perf_lib_layer_params.h>
#include <platform/gaudi/graph_compiler/passes.h>

class DynamicShapesMemsetOutput : public GraphOptimizerTest
{
public:
    TensorPtr createTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes = {})
    {
        if (minSizes.empty())
        {
            return std::make_shared<Tensor>(maxSizes.size(), maxSizes.data(), syn_type_float);
        }
        else
        {
            return std::make_shared<Tensor>(maxSizes.size(), maxSizes.data(), syn_type_float, minSizes.data());
        }
    }

    TensorPtr createPersistentTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes = {})
    {
        auto tensor = createTensor(maxSizes, minSizes);
        synMemoryDescriptor memDesc(true /* persistent */);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000000);
        tensor->map();
        return tensor;
    }

    TensorPtr createShapeTensor(const std::vector<TSize>& maxSizes, const std::vector<TSize>& minSizes)
    {
        return std::make_shared<Tensor>(maxSizes.size(),
                                        maxSizes.data(),
                                        syn_type_float,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        minSizes.data(),
                                        SHAPE_TENSOR);
    }

/*    template< typename CONTAINER >
    void checkTensorMinSizes(const pTensor& tensor, CONTAINER sizes)
    {
        auto minSizeIter = sizes.begin();
        for (int dim = 0; dim < tensor->getDim(); dim++)
        {
            ASSERT_EQ(*minSizeIter++, tensor->getMinimalSizeInElements(dim))
                                    << "Tensor " << tensor->getName() << " unexpected min size in dimension " << dim;
        }
    }
*/
    void setUpReduction(bool useDynamicShapes, int numberOfInputs)
    {
        const std::vector<TSize>& maxSizes = m_maxSizes;
        const std::vector<TSize>& minSizes = useDynamicShapes ? m_minSizes : m_maxSizes;

        TensorPtr shape = createShapeTensor(maxSizes, minSizes);

        for(int i = 0; i < numberOfInputs; i++)
        {
            auto tensor = createTensor(maxSizes, minSizes);
            tensor->setTensorInSram();
            m_tIn.push_back(tensor);
        }

        m_tOut = createTensor(maxSizes);
        m_tOut->setTensorInSram();

        NodePtr reduction = NodeFactory::createNode(m_tIn, {m_tOut}, nullptr, NodeFactory::reductionNodeTypeName, "reduction");
        ASSERT_TRUE(GraphEditor::addNode(m_graph, reduction));
    }

    void addReductionInputs(const NodeVector& inputNodes)
    {
        for(auto node: inputNodes)
        {
            ASSERT_TRUE(GraphEditor::addNode(m_graph, node));
        }
    }

    void runCompilationTests(bool dynamicShapeTest)
    {
        bool verificationRes = verifyMemsetOutputShapes(m_graph);
        ASSERT_EQ(verificationRes, !dynamicShapeTest) << "Pre-compilation check failed";

        ASSERT_TRUE(m_graph.compile()) << "Compilation failed";

        verificationRes = verifyMemsetOutputShapes(m_graph);
        ASSERT_TRUE(verificationRes) << "compilation passed, Is it possible the vewrification passed did not run?";
    }

protected:
    virtual void SetUp()
    {
        GraphOptimizerTest::SetUp();
        CompilationHalReader::setHalReader(GaudiHalReader::instance(synDeviceGaudi));
    }

    virtual void TearDown()
    {
        GraphOptimizerTest::TearDown();
    }

public:
    GaudiGraph m_graph;
    TensorVector                m_tIn;
    TensorPtr m_tOut;
    const std::vector<TSize> m_maxSizes = {64, 256};
    const std::vector<TSize> m_minSizes = {64, 128};

};

TEST_F(DynamicShapesMemsetOutput, BasicMemset)
{
    // Given the following graph:
    //
    // [t1] -----> (memcpy) ---------> [m_tIn[0]] ------+
    //                                                  v
    //             (memset) ---------> [m_tIn[1]] -->(Reduction) --> [m_tOut]
    //                                                  ^
    // [shape] --> (shaped_memsem) --> [m_tIn[2]] ------+
    //
    // t3 shape is expected to be linked to memsem_tIn1 or memcopy and the shape inference should propogate the min size
    // throughout the graph.

    bool isDynamicShapeTest = true;

    setUpReduction(isDynamicShapeTest, 3);

    TensorPtr t1    = createPersistentTensor(m_maxSizes, m_minSizes);
    TensorPtr shape = createShapeTensor(m_maxSizes, m_minSizes);
    NodePtr memcpy    = NodeFactory::createNode({t1}, {m_tIn[0]}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    NodePtr memset   = NodeFactory::createNode({}, {m_tIn[1]}, nullptr, NodeFactory::memsetNodeTypeName, "memset");
    NodePtr shaped_memset   = NodeFactory::createNode({shape}, {m_tIn[2]}, nullptr, NodeFactory::memsetNodeTypeName, "shaped_memset");
    addReductionInputs({memcpy, memset, shaped_memset});

    runCompilationTests(isDynamicShapeTest);
}

TEST_F(DynamicShapesMemsetOutput, NoDynamicShapes)
{
    // Given the following graph:
    //
    // [t1] --> (memcpy) --> [m_tIn[0]] -------+
    //                                          v
    //          (memset) --> [m_tIn[0]] -->(Reduction) --> [m_tOut]
    //

    bool isDynamicShapeTest = false;
    setUpReduction(isDynamicShapeTest, 2);

    TensorPtr t1    = createPersistentTensor(m_maxSizes);

    NodePtr memcpy = NodeFactory::createNode({t1}, {m_tIn[0]}, nullptr, NodeFactory::memcpyNodeTypeName, "memcpy");
    NodePtr memset = NodeFactory::createNode({}, {m_tIn[1]}, nullptr, NodeFactory::memsetNodeTypeName, "memset");
    addReductionInputs({memcpy, memset});

    runCompilationTests(isDynamicShapeTest);
}
