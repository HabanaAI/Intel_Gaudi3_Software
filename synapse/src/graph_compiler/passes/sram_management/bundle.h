#pragma once
#include <vector>
#include "graph_compiler/types.h"
#include "tensor.h"
#include "log_manager.h"
#include <atomic>

#define SLC_TRACE(...) LOG_TRACE(SRAM_SLICE, __VA_ARGS__)
#define SLC_DEBUG(...) LOG_DEBUG(SRAM_SLICE, __VA_ARGS__)
#define SLC_INFO(...)  LOG_INFO(SRAM_SLICE, __VA_ARGS__)
#define SLC_ERR(...)   LOG_ERR(SRAM_SLICE, __VA_ARGS__)
#define SLC_WARN(...)  LOG_WARN(SRAM_SLICE, __VA_ARGS__)

class PostSlicingOpHandler;
using PostSlicingOpHandlerPtr = std::shared_ptr<PostSlicingOpHandler>;
class MmeSlicingStrategy;

const unsigned c_sharedMultiBufLevel = 4;

class Bundle
{
public:
    explicit Bundle(BundleType type = UNDEFINED) : m_index(getNextBundleIndex()), m_type(type) {}

    unsigned          index() const { return m_index; }
    virtual void      addNode(pNode);
    void              removeNode(pNode node);
    const NodeVector& getNodes() const;
    std::string       getName() const;
    BundleType        type() const { return m_type; }
    static uint32_t   getNextBundleIndex();

    // Holds the decisions taken by the brain for the use of
    // the slicing backend when it actually creates the slices.
    struct Solution
    {
        // Decoration of a tensor with the slicing information for it.
        struct SlicedOperand
        {
            using pSlicedOperand = std::shared_ptr<SlicedOperand>;

            explicit SlicedOperand(pTensor tensor)
            : originalTensor(tensor)
            {
                if (tensor)
                {
                    chunkDimensions = tensor->getAllSizesInElements();
                    finalShape = tensor->getAllSizesInElements();
                    finalElementType = tensor->getElementType();
                }
                else
                {
                    chunkDimensions.fill(0);
                    finalShape.fill(0);
                    finalElementType = syn_type_na;
                }
                overlapElementsCount.fill(0);
                offsetBefore.fill(0);
                offsetAfter.fill(0);
                extraLeftoverAfter.fill(0);
                minValidSliceSize.fill(1);
            }
            SlicedOperand(const SlicedOperand& other) = default;
            pTensor     originalTensor;
            SizeArray   chunkDimensions;                 // shape of each chunk, including overlap if exists.
            OffsetArray overlapElementsCount;            // number of elements per dim, which overlap with the previous chunk.
            OffsetArray offsetBefore;                    // number of elements per dim, which are part of the tensor data due to offset before (such as padding)
            OffsetArray offsetAfter;                     // number of elements per dim, which are part of the tensor data due to offset after (such as padding)
            OffsetArray extraLeftoverAfter;              // number of last elements per dim, that are not part of the last slice.
            bool        countPaddingOnlySlice = true;    // If the last slice contains only padding lines - create it according to this flag.
            bool        requiresTensorView = false;      // TEMP FLAG! true if the solver creates slices with overlap, or on multiple dimensions (split fails to handle)
            SizeArray   minValidSliceSize;               // minimal required input data to produce output.
            SizeArray   finalShape;                      // If solver will decide to reshape the operand for flattened.
            bool        resideInSRAM      = false;       // Should chunks be copied to SRAM
            uint32_t    numOfBuffers      = 1;           // The amount of space in SRAM reserved for chunks of this
                                                         // operand. 1 - single buffered, 2 - double buffered, ...
            synDataType finalElementType  = syn_type_na; // the final data type of the operand. It can be different than
                                                         // the original tensor, i.e in case of reduction output it will
                                                         // be forced to fp32
            bool        alignWithCacheLine = false;      // align FCD to cache line for mme performance optimization
            PostSlicingOpHandlerPtr postSlicingHandler = nullptr;  // Modify the sliced node according to the operand slices, after the new node and tensors are created

            bool sharedChainMultiBuf = false;

            bool operator== (const SlicedOperand& other) const;
            bool operator!= (const SlicedOperand& other) const;

            std::string toString();

            void resetSlicingData();
            void copyShapeData(const SlicedOperand& other);
            bool hasOverlap() const;
            bool hasOffset() const;
            bool isFirstSliceSmaller() const;

            // Return true if there is a dimension in the operand, in which the last slice is extended to cover extra
            // leftover tensor elements
            bool hasExtraSizedSlices() const;

            struct SliceOperandComp
            {
                bool operator()(const pSlicedOperand& lhs, const pSlicedOperand& rhs) const;
            };
        };
        using pSlicedOperand = std::shared_ptr<SlicedOperand>;

        // An activation of an operation over slices of the operand.
        struct Operation
        {
            struct SliceReference;
            using pSliceReference = std::shared_ptr<SliceReference>;

            // Reference to a single slice out of the full original tensor
            struct SliceReference
            {
                explicit SliceReference(pSlicedOperand slicedOp)
                    : operand(slicedOp) { coordinates.fill(0); }

                SliceReference(const SliceReference& other) = default;

                struct Hasher
                {
                    size_t operator()(const pSliceReference& s) const;
                };

                struct IsEqual
                {
                    bool operator()(const pSliceReference& obj1, const pSliceReference& obj2) const;
                };

                // The following members identify a specific slice in the sliced tensor
                pSlicedOperand operand;
                CoordArray coordinates; //of the slice, not memory or elements
            };

            explicit Operation(pNode node) : originalNode(node) {}

            pNode                        originalNode;
            std::vector<pSliceReference> inputs;
            std::vector<pSliceReference> outputs;
        };

        std::vector<pSlicedOperand> operands;    // All sliced tensors
        std::list<Operation> operations;  // All activations over specific slices, in execution order.
    };

    Solution& getSolution() { return m_solution; }

    const Solution& getSolution() const { return m_solution; }

    // Prevent copying bundles
    Bundle(const Bundle&) = delete;
    Bundle& operator=(const Bundle&) = delete;

private:
    NodeVector m_nodes;
    Solution   m_solution;
    unsigned   m_index;

    static std::atomic<uint32_t> s_nextBundleIndex;
    const BundleType m_type;
};

using pBundle = std::shared_ptr<Bundle>;

using SlicedOperand = Bundle::Solution::SlicedOperand;
using pSlicedOperand = Bundle::Solution::pSlicedOperand;
using SliceReference = Bundle::Solution::Operation::SliceReference;
using pSliceReference = Bundle::Solution::Operation::pSliceReference;

class SlaveOperands
{
public:
    enum OpernadType
    {
        NonSharedInput ,
        Output,
        Shape
    };

    using OperandContainer = std::map<OpernadType, pSlicedOperand>;

    pSlicedOperand getInput();
    void setInput(pSlicedOperand operand);
    pSlicedOperand getOutput();
    void setOutput(pSlicedOperand operand);
    pSlicedOperand getShapeOperand();
    void setShapeOperand (pSlicedOperand operand);

    bool empty();
    void copy(const SlaveOperands& other);

    OperandContainer::iterator begin() noexcept;
    OperandContainer::const_iterator begin() const noexcept;
    OperandContainer::iterator end() noexcept;
    OperandContainer::const_iterator end() const noexcept;

    std::vector<pSlicedOperand> getSlaveOperands();

private:
        OperandContainer m_operands;
};

struct BundleExpansion
{
    enum Role
    {
        FirstRole = 0,

        // The order of the roles define their priority
        WideInputProducer = FirstRole,
        NarrowInputProducer,
        OutputConsumer,
        SharedInputConsumer,
        SlaveInputProducer,
        SlaveOutputConsumer,
        // Must be last
        NumOfRoles
    };

    BundleExpansion() = default;
    virtual ~BundleExpansion() = default;
    BundleExpansion(const BundleExpansion& other)
    {
        role = other.role;
        stitchedOperand = std::make_shared<SlicedOperand>(*other.stitchedOperand);
        nodeToStitch = other.nodeToStitch;
        reshapeNode = other.reshapeNode;
        bundleNode = other.bundleNode;
        additionalSRAMCapacity = other.additionalSRAMCapacity;
        slaveOperands.copy(other.slaveOperands);
        winningStrategyForSlaveBundle = other.winningStrategyForSlaveBundle;
    }

    void operator=(const BundleExpansion&) = delete;

    static std::string role2String(const Role& role);
    // Single place to aggregate global config and temporary disabling of specific role expansion
    static bool isExpansionEnabledForRole(BundleExpansion::Role role);

    /* Dependant-role:
     * Role R_i depends on R_j if R_j possibly can be discovered only if R_i discovered. */
    static bool isDependentRole(BundleExpansion::Role role);

    static bool isProducer(BundleExpansion::Role role);

    static BundleExpansion::Role masterToSlaveEquivalentRole(BundleExpansion::Role role);

    BundleExpansion::Role     role = BundleExpansion::NumOfRoles;    /* What part is the candidate play in the bundle */
    pSlicedOperand            stitchedOperand;                       /* The sliced tensor in the bundle that we should stitch to */
    // TODO: [SW-25932] Refactor BundleExpansion slaveOperands
    SlaveOperands slaveOperands;                                     /* The other operands of the slave node. This member is */
                                                                     /* not empty for role==SharedInputConsumer, i.e. slave */
    std::list<std::shared_ptr<BundleExpansion>> dependentCandidates; /* The candidates that depend on this candidate - FE:
                                                                      * slave consumer or producer depends on the slave */

    /* The winning strategy of the slave bundle containing the slave MME (nodeToStitch) with possibly
     * TPC producer / consumer (dependentCandidates).
     * This strategy will be used to evaluate the invalid slave candidates in the strategy cost-model.
     * This member is relevant for role==SharedInputConsumer only. */
    std::shared_ptr<MmeSlicingStrategy> winningStrategyForSlaveBundle;

    pNode                     nodeToStitch;                          /* The candidate node to stitch to bundle */
    pNode                     reshapeNode;                           /* A reshape node between the candidate and the
                                                                      * sliced tensor (optional) */
    pNode                     bundleNode;                            /* The node that's already in the bundle that
                                                                      * produces or consumes the stitched operand */
    uint32_t                  additionalSRAMCapacity = 0;            /* How much more SRAM will be needed to add this
                                                                      * candidate to the bundle */
};
using pBundleExpansion = std::shared_ptr<BundleExpansion>;
