#pragma once
#include "transpose_nodes_creator.h"
#include "dma_transpose_cost_model.h"
#include "transpose_utils.h"

class TransposeSplitter
{
public:
    // since spliting complexity is (factorial(tensors dim))^2
    // we avoid splitting high rank transposes
    static constexpr unsigned MAX_DIM_FOR_SPLITTING = 5;
    static constexpr unsigned MAX_LENGTH            = 5;
    using TransposeChain                            = llvm_vecsmall::SmallVector<TransposeNodeParams, 5>;
    // we split each transpose node with dijkstra like algorithm
    // where the nodes defined by permutation from the original input
    using Node       = TransposePermutationArray;
    using Edge       = std::pair<Node, Node>;
    using EdgeToCost = std::map<Edge, uint64_t>;

    class Path
    {
    public:
        Path(TransposeSplitter* splitter = nullptr) : m_splitter(splitter) {}

        bool addTransposeToPath(const TensorPtr&                 input,
                                const TensorPtr&                 output,
                                const TransposePermutationArray& permutation,
                                bool                             intermidiatePath);

        static Path createEmptyPath(TransposeSplitter* splitter);

        bool operator>(const Path& o) const { return m_cost != o.m_cost ? o.m_cost < m_cost : o.m_node < m_node; }
        bool operator<(const Path& o) const { return o > *this; }
        bool operator<=(const Path& o) const { return !(o < *this); }
        bool operator==(const Path& o) const { return m_chain == o.m_chain; }

        const TransposeChain&          chain() const { return m_chain; }
        const TransposeSplitter::Node& node() const { return m_node; }
        const uint64_t&                cost() const { return m_cost; }

        void replaceInput(const TensorPtr& newInput) { m_chain.front().input = newInput; }
        void replaceOutput(const TensorPtr& newOutput) { m_chain.back().output = newOutput; }

    private:
        TransposeSplitter*      m_splitter;  // An instance of transpose splitter.
        TransposeChain          m_chain;     // The chain of edges from the identity transpose to m_node.
        TransposeSplitter::Node m_node;      // The node at the end of the path.
        uint64_t                m_cost = 0;  // The path cost (includes all transposes).
        // To avoid cycles we forbid to visit nodes that already visited. In addition, when path is
        // intermidiate (m_node not equal to original transpose permutation) we also forbid to visit the final node.
        std::vector<TransposeSplitter::Node> m_forbiddenNodes;
    };

    using NodeToPath = std::map<Node, Path>;

    TransposeSplitter(const TransposeNode& transpose, const bool skipPerfReport = false);
    ~TransposeSplitter() = default;
    std::pair<NodeVector, uint64_t> splitTransposeViaCostModel();

private:
    void        calculateBestSplitting();

    std::optional<Path> createLastTranspose(const TensorPtr& input, Path path) const;
    void                initializeState();  // Since we may skip splitting, this function is "lazy constructor".
    std::optional<TransposeSplitter::Path>
    createIntermidiateTranspose(const TensorPtr& input, Path path, const TransposePermutationArray& permutation);

    // Returns an access to the best splitting.
    Path&       getBestSplittingHandler();
    const Path& getBestSplittingHandler() const;
    bool        shouldSkipNewPath(const std::optional<Path>& newPath) const;

    const TransposeNode& m_originalTranspose;
    // Since the transpose creator apply manipulations on the tensors we using dummy input and output.
    const TensorPtr       m_dummyInput;
    const TensorPtr       m_dummyOutput;
    const bool                  m_skipPerfReport;
    uint64_t              m_costThreshold = 0;
    const TransposeNodesCreator m_creator;

    // Caches the cheapest path discovered so far for each node (permutation).
    NodeToPath m_nodesMap;
    // Caches weights for lazy cost calculation.
    EdgeToCost m_edgesMap;
};
