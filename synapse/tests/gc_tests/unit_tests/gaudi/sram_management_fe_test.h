#pragma once

#include "gaudi_graph.h"
#include "graph_compiler/types.h"
#include "graph_optimizer_test.h"
#include "graph_compiler/passes/sram_management/bundlizer.h"
#include "graph_compiler/passes/sram_management/slicing_brain.h"
#include "node_factory.h"

namespace gaudi
{
using Solution = Bundle::Solution;


// Utility class for creating SRAM Management tests.
class SRAMManagementTest : public GraphOptimizerTest
{
protected:
    void SetUp() override;
    void TearDown() override;

    void setGlobalConfForTest(hl_gcfg::GcfgItem& gConf, const std::string& stringValue) override;

    // create an arbitrary tensor
    pTensor createTensor(const std::vector<TSize>&    shape,
                         synDataType                  dataType,
                         bool                         isPersistent = true,
                         const std::vector<TSize>&    minShape     = std::vector<TSize>(),
                         synTensorType                tensorType   = DATA_TENSOR);

    // create a bundle with a single MME node.
    std::shared_ptr<Bundle> createSingleMMENodeBundle(TensorVector       inputs,
                                                      TensorVector       outputs,
                                                      const std::string& guid,
                                                      void*              params     = nullptr,
                                                      unsigned           paramsSize = 0);

    // getter for the Slicing brain
    const MMESlicingBrain& getMmeBrain() const;
    const TPCSlicingBrain& getTpcBrain() const;
    const AllBrains&       getSlicingBrains() const;

    // getter for the graph
    GaudiGraph& getGraph()
    {
        return m_graph;
    }

    // check the brain solution for expected operand size and # of operations.
    void checkSolutionSize(const Solution& solution, unsigned operandSize, unsigned operationSize) const;

    // Check the solution's chunk size for the given original operand
    void checkChunkSize(const Solution& solution, const pTensor operand, const SizeArray& expChunkSize);

    // Use MME (master) brain top strategy to solve a bundle
    void solveBundleWithStrategy(pBundle bundle);

    unsigned m_memorySectionId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;

    class ExecutionOrderChecker
    {
    public:
        ExecutionOrderChecker(unsigned opASlicingDim, unsigned opBSlicingDim)
                : m_opASlicingDim(opASlicingDim), m_opBSlicingDim(opBSlicingDim)
        {
        }

        void setOpSkipFactor(unsigned skip);
        void checkWalkRightExecutionOrder(const Solution&  solution,
                                          unsigned         expRows,
                                          unsigned         expCols,
                                          const DimVector& outputSlicedSpatialDims = {DIM_W, WEIGHT_DIM_K});

        void checkWalkRightSnakeExecutionOrder(const Solution&  solution,
                                               unsigned int     expRows,
                                               unsigned int     expCols,
                                               const DimVector& outputSlicedSpatialDims);

        void checkWalkRightSimpleExecutionOrder(const Solution&  solution,
                                                unsigned int     expRows,
                                                unsigned int     expCols,
                                                const DimVector& outputSlicedSpatialDims);

        void checkWalkDownExecutionOrder(const Solution&  solution,
                                         unsigned         expRows,
                                         unsigned         expCols,
                                         const DimVector& outputSlicedSpatialDims = {DIM_W, WEIGHT_DIM_K});

        void checkWalkDownSnakeExecutionOrder(const Solution&  solution,
                                              unsigned int     expRows,
                                              unsigned int     expCols,
                                              const DimVector& outputSlicedSpatialDims);

        void checkWalkDownSimpleExecutionOrder(const Solution&  solution,
                                               unsigned int     expRows,
                                               unsigned int     expCols,
                                               const DimVector& outputSlicedSpatialDims);

        void checkReductionOnlyExecutionOrder(const Solution& solution,
                                              unsigned int expSlices);
    private:
        const Solution::Operation& getNextOperation(std::list<Solution::Operation>::const_iterator& currentOperation);
        unsigned m_opASlicingDim;
        unsigned m_opBSlicingDim;
        unsigned m_opSkipFactor = 1;
    };

private:
    SlicingBrain::Knobs m_orgKnobs; // backup of Slicing Brain knobs before modifying them
    std::string         m_orgSramMaxCap;
    GaudiGraph          m_graph;
    AllBrains           m_brains {m_graph};
};

} // namespace gaudi
