#pragma once

#include "mme_node.h"
#include "multi_node.h"
#include "node_visitor.h"

using Labels             = std::vector<int>;
using OperandLabels      = std::vector<Labels>;
using LabelCounts        = std::vector<int>;
using OperandLabelCounts = std::vector<LabelCounts>;
using LabelToDimSizes    = std::vector<unsigned>;

// Dummy axis label used to denote an ellipsis in an input or output subscript.
constexpr int kEllipsisLabel = -1;

// Each dimension is categorized into exactly one of five types based on
// whether its corresponding label is present in the input and/or the output
// subscripts.
enum DimensionType
{
    // Reduce dimensions are present in exactly one input; and not in the output
    // and are summed over prior to Tensor contraction.
    kReduce = 0,
    // Batch dimensions are those present in two inputs as well as the output.
    // They are part of the batch dimensions during Tensor contraction.
    // Such dimensions may be broadcasting dimensions (those mapping to
    // ellipsis)
    // or explicit batch dimensions corresponding to named axis labels.
    kBroadcasting = 1,
    kBatch        = 2,
    // Free dimensions are present in exactly one of the inputs, and also the
    // output. These are non-contracted axes in the Tensor contraction.
    kFree = 3,
    // Contract dimensions are present in two inputs, but not the output. These
    // dimensions are contracted in Tensor contraction.
    kContract = 4,

};

class EinsumNode : public MultiNode
{
    DEFINE_VISITOR_METHOD
    friend class NodeFactory;

public:
    typedef MultiNode BaseClass;

    virtual bool validateNode() const override;

    virtual NodePtr clone() const override;

    virtual NodeList extract() override;

    virtual bool validateNodeForGraph(const HabanaGraph& g) const override;

    virtual void setParams(UserParams userParams, unsigned userParamsSize) override;

    void printParamsRawData() const override;

protected:
    SifNodeParams getShapeInferenceFunctionUserParams() override;

    size_t getShapeInferenceFunctionUserParamsSize() const override;

private:
    synEinsumParams            m_params;
    bool                       m_parsing_status;
    OperandLabels              m_input_labels;   // maps dimension to label
    Labels                     m_output_labels;  // maps dimension to label
    std::vector<DimensionType> m_label_types;
    OperandLabelCounts         m_input_label_counts;
    LabelCounts                m_output_label_counts;
    std::vector<bool>          m_input_has_ellipsis;
    bool                       m_output_has_ellipsis = false;
    LabelToDimSizes            m_label_to_dim_sizes;  // maps label to dimension size
    std::vector<uint8_t>       m_sifMetadataBuffer;

    EinsumNode(const TensorVector& inputs, const TensorVector& outputs, UserParams params, std::string_view name);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              std::string_view    guid,
                              std::string_view    name);

    bool IsSupportedEinsumEquation(std::string_view einsumEquation) const;

    void ReduceOperand(TensorPtr& input, unsigned inputIdx, Labels& free_labels, TensorPtr& output, NodeList* retNodes);

    DimensionType GetDimensionType(bool is_removed, bool is_unique);

    void MapToLabels(const std::string& subscript, Labels* labels, std::unordered_map<char, int>* label_mapping);

    bool ParseEinsumEquationHelper(const std::string&        equation,
                                   std::vector<std::string>* input_subscripts,
                                   std::string*              output_subscript) const;

    bool ParseEquation();

    void InsertBroadcastLabels(int          num_bcast_dims,
                               int          num_named_labels,
                               int          ellipsis_axis,
                               Labels*      labels,
                               LabelCounts* label_counts);
    bool RecordLabelToDimension(const int label, const int axis, const TensorPtr& input);

    bool ProcessDimensions();

    void PermuteLabels(const std::vector<int>& permutation, Labels& labels);

    bool ShouldTranspose(const TensorPtr& input, const std::vector<int>& permutation);

    void AddTransposeNodeIfNeeded(const TensorPtr&  input,
                                  std::vector<int>& permutation,
                                  TensorPtr&        output,
                                  NodeList*         retNodes,
                                  std::string       transposeName);

    void AddReduceNode(const TensorPtr&      input,
                       std::vector<unsigned> axesToReduce,
                       TensorPtr&            output,
                       NodeList*             retNodes,
                       std::string           baseName);

    bool ReshapeNeeded(const TensorPtr& input, SizeArray* outputSizes, int newTensorRank);

    void AddReshapeNode(const TensorPtr&   input,
                        const TensorPtr&   shape,
                        SizeArray*         outputSizes,
                        SizeArray*         outputMinSizes,
                        unsigned           outputTensorRank,
                        TensorPtr&         output,
                        NodeList*          retNodes,
                        const std::string& reshapeName);

    void AddBatchGemmNode(const TensorPtr& lhs,
                          const TensorPtr& rhs,
                          TensorPtr&       output,
                          NodeList*        retNodes,
                          std::string      batchGemmName);

    void ContractOperands(const TensorVector& inputs, TensorPtr& output, NodeList* retNodes);

    void ExpandOutput(const TensorVector&  inputs,
                      const TensorPtr&     contractionOutput,
                      const OperandLabels& free_labels,
                      TensorPtr&           output,
                      Labels&              result_labels,
                      NodeList*            retNodes);

    void AddDynamicReshapeShapeNode(const TensorPtr&   input,
                                    std::string*       reshapeEq,
                                    SizeArray*         outputSizes,
                                    SizeArray*         outputMinSizes,
                                    unsigned           shapeTensorRank,
                                    TensorPtr&         shape,
                                    NodeList*          retNodes,
                                    const std::string& nodeName);

    void AddEinsumExpandShapeNode(const TensorVector&    inputs,
                                  std::vector<unsigned>* freeDims,
                                  SizeArray*             outputSizes,
                                  SizeArray*             outputMinSizes,
                                  unsigned               shapeTensorRank,
                                  TensorPtr&             shape,
                                  NodeList*              retNodes,
                                  const std::string&     nodeName);
};
