#include "einsum_node.h"

#include "dynamic_reshape_shape_node.h"
#include "graph_traits.h"
#include "habana_graph.h"
#include "node_factory.h"
#include "perf_lib_layer_params.h"
#include "sif/shape_inference_metadata.h"
#include "tpc_kernel_names.h"
#include "transpose_utils.h"
#include "types_exception.h"

#include <string>

// Returns the DimensionType given whether the corresponding label is present
// in exactly one input subscript (is_unique) and whether it is absent from
// the output subscripts (is_removed). Does not handle broadcasting
// dimensions.
DimensionType EinsumNode::GetDimensionType(bool is_removed, bool is_unique)
{
    if (!is_removed && !is_unique)
    {
        return kBatch;
    }
    else if (!is_removed && is_unique)
    {
        return kFree;
    }
    else if (is_removed && !is_unique)
    {
        return kContract;
    }
    else  // is_removed && is_unique
    {
        return kReduce;
    }
}

// Maps the character labels to consecutive integers.
void EinsumNode::MapToLabels(const std::string& subscript, Labels* labels, std::unordered_map<char, int>* label_mapping)
{
    for (int i = 0; i < subscript.size(); ++i)
    {
        const char label_char = subscript[i];
        if (label_char == '.')
        {
            labels->push_back(kEllipsisLabel);
            if (subscript[i + 1] != '.' || subscript[i + 2] != '.')
            {
                LOG_ERR(HABANA_NODE, "Expecting ellipsis (...) in equation {}", subscript);
            }
            i += 2;  // Skip next 2 characters as well.

            continue;
        }
        if (label_mapping->find(label_char) == label_mapping->end())
        {
            const int next_label         = label_mapping->size();
            (*label_mapping)[label_char] = next_label;
        }
        const int mapped_label = (*label_mapping)[label_char];
        labels->push_back(mapped_label);
    }
}

bool EinsumNode::ParseEinsumEquationHelper(const std::string&        equation,
                                           std::vector<std::string>* input_subscripts,
                                           std::string*              output_subscript) const
{
    std::vector<std::string> inputs_and_output_subscripts = splitString(equation, '-');
    if (inputs_and_output_subscripts.size() != 2 || inputs_and_output_subscripts[1][0] != '>')
    {
        LOG_ERR(HABANA_NODE, "Expecting exactly one '->' in einsum equation: \"{}\"", equation);
        return false;
    }
    *output_subscript = std::move(inputs_and_output_subscripts[1].substr(1));
    *input_subscripts = splitString(std::move(inputs_and_output_subscripts[0]), ',');
    if (input_subscripts->size() != 1 && input_subscripts->size() != 2)
    {
        LOG_ERR(HABANA_NODE,
                "Expecting 1 or 2 input subscripts in equation \"{}\" but got: {}",
                equation,
                input_subscripts->size());
        return false;
    }
    // Reverse equation for Synapse FCD
    for (int i = 0; i < input_subscripts->size(); i++)
    {
        std::reverse((*input_subscripts)[i].begin(), (*input_subscripts)[i].end());
    }
    std::reverse(output_subscript->begin(), output_subscript->end());
    return true;
}

// Parses and validates the equation and the input shapes. Single character
// labels are integerized and we populate input and output label subscripts
// and corresponding counts. Also create the mapping from (named) labels to
// their DimensionType.
// for more details on einsum - https://www.tensorflow.org/api_docs/python/tf/einsum
bool EinsumNode::ParseEquation()
{
    std::vector<std::string> input_str;
    std::string              output_str;
    ParseEinsumEquationHelper(m_params.equation, &input_str, &output_str);

    // Temporary map from single character labels to (consecutive) integer
    // labels.
    std::unordered_map<char, int> label_mapping;
    int                           num_inputs = input_str.size();
    m_input_labels.resize(num_inputs);

    // Map from single characters to integer labels.
    for (int i = 0; i < num_inputs; ++i)
    {
        MapToLabels(input_str[i], &m_input_labels.at(i), &label_mapping);
    }
    MapToLabels(output_str, &m_output_labels, &label_mapping);

    // Compute counts for input and output labels.
    int num_labels = label_mapping.size();
    m_input_label_counts.resize(num_inputs);
    m_input_has_ellipsis.resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i)
    {
        m_input_label_counts.at(i).resize(num_labels);
        for (const int label : m_input_labels.at(i))
        {
            if (label != kEllipsisLabel)
            {
                m_input_label_counts.at(i)[label] += 1;
            }
            else
            {
                m_input_has_ellipsis.at(i) = true;
            }
        }
    }
    m_output_label_counts.resize(num_labels);
    for (const int label : m_output_labels)
    {
        if (label != kEllipsisLabel)
        {
            m_output_label_counts.at(label) += 1;
        }
        else
        {
            m_output_has_ellipsis = true;
        }
    }

    // Map each label to a unique DimensionType.
    m_label_types.resize(num_labels);
    for (int label = 0; label < num_labels; ++label)
    {
        if (label == kEllipsisLabel) continue;
        bool removed = m_output_label_counts[label] == 0;
        bool unique  = num_inputs == 1 || m_input_label_counts[0][label] == 0 || m_input_label_counts[1][label] == 0;
        m_label_types[label] = GetDimensionType(removed, unique);
    }
    return true;
}

// Insert new (unnamed) broadcasting labels at the location of ellipsis.
void EinsumNode::InsertBroadcastLabels(int          num_bcast_dims,
                                       int          num_named_labels,
                                       int          ellipsis_axis,
                                       Labels*      labels,
                                       LabelCounts* label_counts)
{
    labels->erase(labels->begin() + ellipsis_axis);
    labels->insert(labels->begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels->begin() + ellipsis_axis, labels->begin() + ellipsis_axis + num_bcast_dims, num_named_labels);
    // Increment label counts. Since these are new labels, the count is set to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
    m_label_types.resize(num_named_labels + num_bcast_dims, kBroadcasting);
}

// Record and validate the label to dimension mapping. Must be a named
// (non-broadcasting) label as broadcasting labels don't have a fixed
// dimension.
bool EinsumNode::RecordLabelToDimension(const int label, const int axis, const TensorPtr& input)
{
    const unsigned input_dim_size = input->getSizeInElements(axis);

    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label < m_label_to_dim_sizes.size() && m_label_to_dim_sizes.at(label) != 0)
    {
        if (m_label_types[label] != kBroadcasting && m_label_to_dim_sizes.at(label) != input_dim_size)
        {
            LOG_ERR(HABANA_NODE,
                    "Expected dimension size {} at axis {} of the input shaped {} but got dimension size {}",
                    m_label_to_dim_sizes.at(label),
                    axis,
                    input->getDimSizesStr(),
                    input_dim_size);
            return false;
        }
        // label_to_dim_sizes[label] should be equal to either 1 or input_dim_size
        if (m_label_types[label] == kBroadcasting && m_label_to_dim_sizes[label] != 1 &&
            m_label_to_dim_sizes[label] != input_dim_size && input_dim_size != 1)
        {
            LOG_ERR(HABANA_NODE,
                    "Expected dimension size {} or 1 at axis {} of the input shaped {} but got dimension size {}",
                    m_label_to_dim_sizes.at(label),
                    axis,
                    input->getDimSizesStr(),
                    input_dim_size);
            return false;
        }
    }

    m_label_to_dim_sizes[label] = (m_label_types[label] == kBroadcasting)
                                      ? std::max(m_label_to_dim_sizes[label], input_dim_size)
                                      : input_dim_size;
    return true;
}

bool EinsumNode::ProcessDimensions()
{
    auto FindLabelIndex = [](Labels& array, int label) -> int {
        auto it = find(array.begin(), array.end(), label);
        return (it == array.end()) ? -1 : it - array.begin();
    };

    if (m_inputs.size() != m_input_labels.size())
    {
        LOG_ERR(HABANA_NODE, "Expected {} inputs but got: {}", m_input_labels.size(), m_inputs.size());
        return false;
    }

    const int num_inputs = m_inputs.size();
    const int num_named_labels = m_label_types.size();

    // We infer the number of broadcasting dimensions by taking the maximum rank
    // among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;

    for (int i = 0; i < num_inputs; i++)
    {
        Labels&   labels               = m_input_labels[i];
        const int ellipsis_label_index = FindLabelIndex(labels, kEllipsisLabel);
        const int num_bcast_dims       = m_inputs[i]->getDim() - labels.size() + 1;

        if (ellipsis_label_index != -1)
        {
            // Input has an ellipsis
            if (m_inputs[i]->getDim() + 1 < labels.size())
            {
                LOG_ERR(HABANA_NODE,
                        "Expected input {} to have rank at least {} but got: {}",
                        i,
                        labels.size() - 1,
                        m_inputs[i]->getDim());
                return false;
            }

            InsertBroadcastLabels(num_bcast_dims,
                                  num_named_labels,
                                  ellipsis_label_index,
                                  &labels,
                                  &m_input_label_counts.at(i));
            max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
        }
        else
        {
            // Input has no ellipsis
            if (m_inputs[i]->getDim() != labels.size())
            {
                LOG_ERR(HABANA_NODE,
                        "Expected input {} to have rank {} but got: {}",
                        i,
                        labels.size(),
                        m_inputs[i]->getDim());
                return false;
            }
        }

        m_label_to_dim_sizes.resize(num_named_labels + max_bcast_dims);

        for (int axis = 0; axis < labels.size(); ++axis)
        {
            const int label = labels[axis];
            if (!RecordLabelToDimension(label, axis, m_inputs[i])) return false;
        }
    }

    // If no broadcast labels anywhere, we're done.
    if (std::count(m_input_has_ellipsis.begin(), m_input_has_ellipsis.end(), true) == 0 && !m_output_has_ellipsis)
    {
        return true;
    }

    // Insert broadcasting dimensions in the output labels.
    const int ellipsis_label_index = FindLabelIndex(m_output_labels, kEllipsisLabel);
    if (ellipsis_label_index != -1)
    {
        InsertBroadcastLabels(max_bcast_dims,
                              num_named_labels,
                              ellipsis_label_index,
                              &m_output_labels,
                              &m_output_label_counts);
    }
    else if (max_bcast_dims > 0)
    {
        LOG_ERR(HABANA_NODE,
                "Output contains {} broadcasting dimension(s) but no ellipsis (...) "
                "was found in the output subscripts.",
                max_bcast_dims);
        return false;
    }

    return true;
}

// Permutes the labels according to the given permutation.
void EinsumNode::PermuteLabels(const std::vector<int>& permutation, Labels& labels)
{
    Labels permuted_labels(labels.size());
    for (int i = 0; i < labels.size(); ++i)
    {
        permuted_labels[i] = labels[permutation[i]];
    }
    labels.swap(permuted_labels);
}

// Returns whether transposing would be a no-op; whether input has rank < 2 or
// the permutation is the identity permutation.
bool EinsumNode::ShouldTranspose(const TensorPtr& input, const std::vector<int>& permutation)
{
    if (input->getDim() < 2) return false;
    for (int i = 0; i < permutation.size(); ++i)
    {
        if (permutation[i] != i) return true;
    }
    return false;
}

void EinsumNode::AddTransposeNodeIfNeeded(const TensorPtr&  input,
                                          std::vector<int>& permutation,
                                          TensorPtr&        output,
                                          NodeList*         retNodes,
                                          std::string       transposeName)
{
    if (!ShouldTranspose(input, permutation))
    {
        output = input;
        LOG_TRACE(HABANA_NODE, "Einsum - no need to do Transpose for node - {}", transposeName);
        return;
    }

    TransposePermutationArray permutationArray;
    for (int i = 0; i < input->getDim(); ++i)
    {
        permutationArray.push_back(static_cast<TransposePermutationDim>(permutation[i]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input->getTotalElements() == 0)
    {
        // TODO:[SW-39925] do we need reshape?
        output = input;
        return;
    }
    synTransposeParamsNDims params = permutationToParams(permutationArray);
    output                    = getTensorAfterTranspose(*input, permutationArray);
    output->setName(transposeName);

    NodePtr transpose = NodeFactory::createNode({input}, {output}, &params, "transpose", transposeName);

    retNodes->push_back(transpose);
    LOG_TRACE(HABANA_NODE,
              "Einsum - Adding new node- {}, sizes ={}, strides ={}",
              transpose->getNodeName(),
              transpose->getOutput(0)->getDimSizesStr(),
              transpose->getOutput(0)->getStridesStr());
}

void EinsumNode::AddReduceNode(const TensorPtr&      input,
                               std::vector<unsigned> axesToReduce,
                               TensorPtr&            output,
                               NodeList*             retNodes,
                               std::string           baseName)
{
    if (axesToReduce.empty())
    {
        output = input;
        LOG_TRACE(HABANA_NODE, "Einsum - no need to do Reduce for node - {}", baseName);
        return;
    }
    // calculate output shape - remove axis to reduce from input sizes.
    SizeArray       reshapedSizes    = input->getAllSizesInElements();
    SizeArray       reshapedMinSizes = input->getAllMinimalSizesInElements();
    ComponentVector reshape_cv       = DynamicReshapeShapeNode::initComponentVector(input->getDim());
    int             axisToReduce     = axesToReduce[0];
    for (int i = 1; i < axesToReduce.size(); ++i)
    {
        int axis = axesToReduce[i];
        reshapedSizes[axisToReduce] *= reshapedSizes[axis];
        reshapedSizes[axis] = 1;
        reshapedMinSizes[axisToReduce] *= reshapedMinSizes[axis];
        reshapedMinSizes[axis] = 1;
        reshape_cv[axisToReduce] += "*" + reshape_cv[axis];
        reshape_cv[axis] = "";
    }

    TensorPtr   reshapedInput;
    TensorPtr   shape         = nullptr;
    int         newTensorSize = input->getDim() - axesToReduce.size() + 1;

    if (input->isDynamicShape())
    {
        std::string reshapeEq = DynamicReshapeShapeNode::makeReshapeEquation(reshape_cv, input->getDim());
        AddDynamicReshapeShapeNode(input,
                                   &reshapeEq,
                                   &reshapedSizes,
                                   &reshapedMinSizes,
                                   newTensorSize,
                                   shape,
                                   retNodes,
                                   fmt::format("{}_dynamic_reshape_before_reduce", baseName));
    }

    AddReshapeNode(input,
                   shape,
                   &reshapedSizes,
                   &reshapedMinSizes,
                   newTensorSize,
                   reshapedInput,
                   retNodes,
                   fmt::format("{}_reshape_before_reduce", baseName));

    reshapedSizes[axisToReduce] = 1;
    reshapedMinSizes[axisToReduce] = 1;
    std::string reduceName         = fmt::format("{}_reduce_sum", baseName);
    output =
        std::make_shared<Tensor>(newTensorSize, reshapedSizes.data(), input->getElementType(), reshapedMinSizes.data());
    output->setName(reduceName);
    output->setAllQuantizationParams(input->getAllQuantizationParams());
    output->setDynamicRange(input->getDynamicRange());

    ns_Reduction::Params params = {axisToReduce};
    std::string          kernelName = fmt::format("reduce_sum_fwd{}", type2Str(input->getElementType()));
    NodePtr              reduceSumNode =
        NodeFactory::createGenericTPCNode({reshapedInput}, {output}, &params, kernelName, reduceName);
    retNodes->push_back(reduceSumNode);

    LOG_TRACE(HABANA_NODE,
              "Einsum - Adding new node- {}, sizes ={}, strides ={}",
              reduceSumNode->getNodeName(),
              reduceSumNode->getOutput(0)->getDimSizesStr(),
              reduceSumNode->getOutput(0)->getStridesStr());
}

bool EinsumNode::ReshapeNeeded(const TensorPtr& input, SizeArray* outputSizes, int newTensorSize)
{
    bool reshapeNeeded = true;
    if (input->getDim() == newTensorSize)
    {
        reshapeNeeded = false;
        unsigned dim  = input->getDim();
        for (unsigned d = 0; d < dim; ++d)
        {
            if (input->getSizeInElements(d) != (*outputSizes)[d])
            {
                reshapeNeeded = true;
            }
        }
    }
    return reshapeNeeded;
}

void EinsumNode::AddReshapeNode(const TensorPtr&   input,
                                const TensorPtr&   shape,
                                SizeArray*         outputSizes,
                                SizeArray*         outputMinSizes,
                                unsigned           outputTensorRank,
                                TensorPtr&         output,
                                NodeList*          retNodes,
                                const std::string& reshapeName)
{
    outputTensorRank = outputTensorRank == 0 ? 1 : outputTensorRank;
    if (!ReshapeNeeded(input, outputSizes, outputTensorRank))
    {
        output = input;
        LOG_TRACE(HABANA_NODE, "Einsum Node no need to do Reshape for node - {}", reshapeName);
        return;
    }

    output = std::make_shared<Tensor>(outputTensorRank,
                                      outputSizes->data(),
                                      input->getElementType(),
                                      (shape) ? outputMinSizes->data() : nullptr);
    output->setName(reshapeName);
    output->setAllQuantizationParams(input->getAllQuantizationParams());
    output->setDynamicRange(input->getDynamicRange());

    NodePtr      reshape;
    TensorVector inputs = {input};
    if (shape) inputs.push_back(shape);

    reshape = NodeFactory::createNode(inputs, {output}, nullptr, NodeFactory::reshapeNodeTypeName, reshapeName);

    retNodes->push_back(reshape);
    LOG_TRACE(HABANA_NODE,
              "Einsum - Adding new node- {}, sizes ={}, strides ={}",
              reshape->getNodeName(),
              reshape->getOutput(0)->getDimSizesStr(),
              reshape->getOutput(0)->getStridesStr());
}

void EinsumNode::AddBatchGemmNode(const TensorPtr& lhs,
                                  const TensorPtr& rhs,
                                  TensorPtr&       output,
                                  NodeList*        retNodes,
                                  std::string      batchGemmName)
{
    // calculate Output shape
    SizeArray lhsSizes       = lhs->getAllSizesInElements();
    SizeArray rhsSizes       = rhs->getAllSizesInElements();
    SizeArray lhsMinSizes    = lhs->getAllMinimalSizesInElements();
    SizeArray rhsMinSizes    = rhs->getAllMinimalSizesInElements();
    SizeArray outputSizes    = {1, 1, 1, 1, 1};
    SizeArray outputMinSizes = {1, 1, 1, 1, 1};

    outputSizes[0]    = rhsSizes[1];
    outputSizes[1]    = lhsSizes[1];
    outputMinSizes[0] = rhsMinSizes[1];
    outputMinSizes[1] = lhsMinSizes[1];
    auto maxDim       = std::max(lhs->getDim(), rhs->getDim());

    for (int dim = DIM_GEMM_BATCH; dim < maxDim; dim++)
    {
        outputSizes[dim]    = std::max(lhsSizes[dim], rhsSizes[dim]);
        outputMinSizes[dim] = std::max(lhsMinSizes[dim], rhsMinSizes[dim]);
    }
    output = std::make_shared<Tensor>(maxDim, outputSizes.data(), lhs->getElementType(), outputMinSizes.data());
    output->setName(batchGemmName);
    output->setAllQuantizationParams(lhs->getAllQuantizationParams());
    output->setDynamicRange(lhs->getDynamicRange());
    synGEMMParams params        = synGEMMParams(false, true);
    NodePtr       batchGemmNode = NodeFactory::createNode({lhs, rhs},
                                                    {output},
                                                    &params,
                                                    sizeof(params),
                                                    NodeFactory::batchGemmNodeTypeName,
                                                    batchGemmName);
    retNodes->push_back(batchGemmNode);
    LOG_TRACE(HABANA_NODE,
              "Einsum - Adding new node- {}, sizes ={}, strides ={}",
              batchGemmNode->getNodeName(),
              batchGemmNode->getOutput(0)->getDimSizesStr(),
              batchGemmNode->getOutput(0)->getStridesStr());
}

void EinsumNode::ReduceOperand(TensorPtr& input_transposed,
                               unsigned   inputIdx,
                               Labels&    free_labels,
                               TensorPtr& output,
                               NodeList*  retNodes)
{
    std::string inputIdxStr = std::to_string(inputIdx);
    Labels&     labels      = m_input_labels[inputIdx];
    // TODO:[SW-39925] handle repeated labels "i->iii"

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each DimensionType.
    // std::vector<unsigned> reshape(5, 1);
    // std::vector<unsigned> minReshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    SizeArray             sizes = {1, 1, 1, 1, 1};
    SizeArray             minSizes = {1, 1, 1, 1, 1};
    ComponentVector       reshape_cv(5, "1");
    std::vector<unsigned> axisToReduce;
    int                   batchIdx = DIM_GEMM_BATCH;
    for (int label_idx = 0; label_idx < labels.size(); ++label_idx)
    {
        int      label      = labels.at(label_idx);
        auto     labelType  = m_label_types[label];
        unsigned dimSize    = input_transposed->getSizeInElements(label_idx);  // label_idx == dim
        unsigned minDimSize = input_transposed->getMinimalSizeInElements(label_idx);
        if (labelType == kBroadcasting || labelType == kBatch)
        {
            minSizes[batchIdx]   = minDimSize;
            sizes[batchIdx]      = dimSize;
            reshape_cv[batchIdx] = std::string {DynamicReshapeShapeNode::getLabelForDim(label_idx)};
            batchIdx++;
        }
        else if (labelType == kReduce)
        {
            axisToReduce.push_back(label_idx);
        }
        else if (labelType == kFree)
        {
            free_labels.push_back(label);
            minSizes[DIM_W] *= minDimSize;
            sizes[DIM_W] *= dimSize;
            reshape_cv[DIM_W] += "*" + std::string {DynamicReshapeShapeNode::getLabelForDim(label_idx)};
        }
        else if (labelType == kContract)
        {
            minSizes[DIM_C] *= minDimSize;
            sizes[DIM_C] *= dimSize;
            reshape_cv[DIM_C] += "*" + std::string {DynamicReshapeShapeNode::getLabelForDim(label_idx)};
        }
    }

    TensorPtr   input_reduced;
    AddReduceNode(input_transposed, axisToReduce, input_reduced, retNodes, fmt::format("{}/{}", m_name, inputIdxStr));

    TensorPtr shape = nullptr;
    reshape_cv.resize(batchIdx);
    if (input_reduced->isDynamicShape())
    {
        std::string reshapeEq = DynamicReshapeShapeNode::makeReshapeEquation(reshape_cv, input_reduced->getDim());
        AddDynamicReshapeShapeNode(input_reduced,
                                   &reshapeEq,
                                   &sizes,
                                   &minSizes,
                                   batchIdx,
                                   shape,
                                   retNodes,
                                   fmt::format("{}/{}_dynamic_reshape_after_reduce", m_name, inputIdxStr));
    }

    AddReshapeNode(input_reduced,
                   shape,
                   &sizes,
                   &minSizes,
                   batchIdx,
                   output,
                   retNodes,
                   fmt::format("{}/{}_reshape_after_reduce", m_name, inputIdxStr));
}

// Contracts the inputs along the last axis. (or the second last if the
// corresponding value of swap_free_and_contract is true). The batch
// dimensions are broadcast to the output shape.
// TODO(anudhyan): BatchMatMul might devolve into a component-wise
// multiplication when the matrix shape is [1,1]; in this case BatchMatMul
// functor would be very inefficient. The functor should detect if this is the
// case and perform componentwise multiplication functor instead.
void EinsumNode::ContractOperands(const TensorVector& inputs, TensorPtr& output, NodeList* retNodes)
{
    if (inputs.size() == 1)
    {
        output = inputs[0];
        LOG_TRACE(HABANA_NODE, "Einsum Node no need to do contraction");
        return;
    }

    if (inputs[0]->getTotalElements() == 0 || inputs[1]->getTotalElements() == 0)
    {
        // TODO:[SW-39925] return zero Tensor (memset node?) can be handled by the MME node?
        LOG_TRACE(HABANA_NODE, "Einsum Node Total Elements is zero - no need to do contraction");
        return;
    }
    AddBatchGemmNode(inputs[0], inputs[1], output, retNodes, fmt::format("{}/batch_gemm", m_name));
}

void EinsumNode::ExpandOutput(const TensorVector&  inputs,
                              const TensorPtr&     contractionOutput,
                              const OperandLabels& free_labels,
                              TensorPtr&           output,
                              Labels&              result_labels,
                              NodeList*            retNodes)
{
    // Reshape contract output to uncompress the free indices
    int       num_inputs             = inputs.size();
    int       num_labels             = m_label_types.size();
    SizeArray reshapeSizes           = {1, 1, 1, 1, 1};
    SizeArray minReshapeSizes        = {1, 1, 1, 1, 1};
    SizeArray contractOutputShape    = contractionOutput->getAllSizesInElements();
    SizeArray contractOutputMinShape = contractionOutput->getAllMinimalSizesInElements();

    int batchIdx = DIM_GEMM_BATCH;
    int dimIdx   = 0;
    //  free dims were compressed, so in contractionOutput[0] we have a product of all free dims in input[1]
    //                                in contractionOutput[1] we have a product of all free dims in input[0]
    std::vector<unsigned> freeDims[2];

    for (int i = num_inputs - 1; i >= 0; --i)
    {
        for (int label : free_labels[i])
        {
            unsigned dim = std::distance(m_input_labels[i].begin(),
                                         std::find(m_input_labels[i].begin(), m_input_labels[i].end(), label));

            result_labels.push_back(label);
            minReshapeSizes[dimIdx] = inputs[i]->getMinimalSizeInElements(dim);
            reshapeSizes[dimIdx]    = inputs[i]->getSizeInElements(dim);  // m_label_to_dim_sizes[label];
            // get all free dimensions from (transposed) inputs
            freeDims[i].push_back(dim);
            dimIdx++;
        }
    }
    // for (int label = num_labels - 1; label >= 0; --label)
    for (int label = 0; label < num_labels; ++label)
    {
        if (m_label_types[label] == kBatch)
        {
            minReshapeSizes[dimIdx] = contractOutputMinShape[batchIdx];
            reshapeSizes[dimIdx]    = contractOutputShape[batchIdx];
            result_labels.push_back(label);
            dimIdx++;
            batchIdx++;
        }
    }

    // for (int label = num_labels - 1; label >= 0; --label)
    for (int label = 0; label < num_labels; ++label)
    {
        if (m_label_types[label] == kBroadcasting)
        {
            minReshapeSizes[dimIdx] = contractOutputMinShape[batchIdx];
            reshapeSizes[dimIdx]    = contractOutputShape[batchIdx];
            result_labels.push_back(label);
            dimIdx++;
            batchIdx++;
        }
    }

    TensorPtr shape = nullptr;
    if (m_inputs[0]->isDynamicShape() || (num_inputs == 2 && m_inputs[1]->isDynamicShape()))
    {
        TensorVector expandShapeInputs;

        expandShapeInputs.push_back(contractionOutput);
        expandShapeInputs.push_back(inputs[0]);
        if (num_inputs == 2) expandShapeInputs.push_back(inputs[1]);

        AddEinsumExpandShapeNode(expandShapeInputs,
                                 freeDims,
                                 &reshapeSizes,
                                 &minReshapeSizes,
                                 dimIdx,
                                 shape,
                                 retNodes,
                                 fmt::format("{}/expand_before_reshape", m_name));
    }

    AddReshapeNode(contractionOutput,
                   shape,
                   &reshapeSizes,
                   &minReshapeSizes,
                   dimIdx,
                   output,
                   retNodes,
                   fmt::format("{}/reshape_after_contraction", m_name));

    return;
}

EinsumNode::EinsumNode(const TensorVector& inputs,
                       const TensorVector& outputs,
                       UserParams          params,
                       std::string_view    name)
: MultiNode(inputs, outputs, name, Node::TYPE_EINSUM, SIF_EINSUM)
{
    setParams(params, sizeof(synEinsumParams));
    m_parsing_status = ParseEquation() && ProcessDimensions();
}

NodePtr EinsumNode::createNode(const TensorVector& inputs,
                               const TensorVector& outputs,
                               UserParams          userParams,
                               std::string_view    guid,
                               std::string_view    name)
{
    return NodePtr(new EinsumNode(inputs, outputs, userParams, name));
}

void EinsumNode::setParams(UserParams userParams, unsigned int userParamsSize)
{
    if (userParams == nullptr)
    {
        LOG_ERR(HABANA_NODE, "EinsumNode userParams is null");
        throw InvalidNodeParamsException(m_name, "userParams");
    }
    std::string equation = static_cast<synEinsumParams*>(userParams)->equation;
    if (equation == INVALID_EINSUM_EQUATION)
    {
        HB_ASSERT(false,
                  "SynEinsumParams equation max length of {} reached for node {}",
                  MAX_EINSUM_EQUATION_LENGTH,
                  m_name);
    }
    if (userParamsSize != sizeof(synEinsumParams))
    {
        LOG_ERR(HABANA_NODE, "EinsumNode userParams size is incorrect");
        throw InvalidNodeParamsSizeException(m_name, userParamsSize, sizeof(synEinsumParams));
    }
    synEinsumParams params = *(synEinsumParams*)userParams;
    LOG_INFO(HABANA_NODE, "Create EinsumNode node, name - {}, params - equation=\"{}\"", m_name, params.equation);
    m_params = params;
}

bool EinsumNode::IsSupportedEinsumEquation(std::string_view einsumEquation) const
{
    // check if their is a repeated indcies in string, this is option is not covered yet in implemnation
    for (int i = 0; i < einsumEquation.size(); i++)
    {
        if (einsumEquation[i] == '.') continue;
        if (std::count(einsumEquation.begin(), einsumEquation.end(), einsumEquation[i]) > 1)
        {
            LOG_ERR(HABANA_NODE, "EinsumNode is not supported yet, equation=\"{}\"", einsumEquation);
            return false;
        }
    }
    return true;
}

bool EinsumNode::validateNode() const
{
    if ((m_inputs.size() != 1 && m_inputs.size() != 2) || m_outputs.size() != 1)
    {
        LOG_ERR(HABANA_NODE, "Invalid number of operands (expecting max 2 input and 1 output)");
        return false;
    }
    std::vector<std::string> input_str;
    std::string              output_str;
    if (!ParseEinsumEquationHelper(m_params.equation, &input_str, &output_str))
    {
        return false;
    }
    bool isEinsumSupprted = true;
    for (int i = 0; i < input_str.size(); i++)
    {
        isEinsumSupprted &= IsSupportedEinsumEquation(input_str[i]);
    }
    isEinsumSupprted &= IsSupportedEinsumEquation(output_str);
    isEinsumSupprted &= m_parsing_status;

    return isEinsumSupprted && MultiNode::validateNode();
}

NodePtr EinsumNode::clone() const
{
    return NodePtr(new EinsumNode(*this));
}

NodeList EinsumNode::extract()
{
    NodeList retNodes;
    if (!m_parsing_status) return retNodes;

    const int     num_inputs = m_inputs.size();
    OperandLabels free_labels(num_inputs);
    TensorVector  inputs_transposed(num_inputs);
    TensorVector  inputs_reduced(num_inputs);
    TensorPtr     contraction_output;

    for (int i = 0; i < num_inputs; ++i)
    {
        Labels& labels = m_input_labels[i];
        // Find the permutation to transpose the input dimensions in the reversed order of
        // DimensionType; i.e. reduce contract free batch brodcast dimensions.
        std::vector<int> permutation(m_inputs[i]->getDim());

        std::iota(permutation.begin(), permutation.end(), 0);
        std::sort(permutation.begin(), permutation.end(), [&](int i, int j) {
            int label_i = labels[i];
            int label_j = labels[j];
            return std::tie(m_label_types[label_i], label_j) > std::tie(m_label_types[label_j], label_i);
        });
        std::string transposeName = fmt::format("{}/{}_transpose_before_reduce", m_name, i);
        AddTransposeNodeIfNeeded(m_inputs[i], permutation, inputs_transposed[i], &retNodes, transposeName);

        PermuteLabels(permutation, labels);

        ReduceOperand(inputs_transposed[i], i, free_labels[i], inputs_reduced[i], &retNodes);
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    ContractOperands(inputs_reduced, contraction_output, &retNodes);

    // Reshape contract output to uncompress the free indices
    int       num_labels = m_label_types.size();
    Labels    result_labels;
    TensorPtr contraction_output_expanded;

    // Reshape contract output to uncompress the free indices
    ExpandOutput(inputs_transposed,
                 contraction_output,
                 free_labels,
                 contraction_output_expanded,
                 result_labels,
                 &retNodes);

    // TODO:[SW-39925] Inflate the output if necessary. (E.g. for the equation 'i->iii' which

    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, 1, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int> output_permutation(m_output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i)
    {
        // Remember the position of only the leftmost result label.
        if (label_to_position[result_labels[i]] == -1)
        {
            label_to_position[result_labels[i]] = i;
        }
    }
    for (int i = 0; i < m_output_labels.size(); ++i)
    {
        output_permutation[i] = label_to_position[m_output_labels[i]];
        // We have found the leftmost occurrence. The next one would be adjacent.
        label_to_position[m_output_labels[i]] += 1;
    }
    TensorPtr   outputTensor;
    AddTransposeNodeIfNeeded(contraction_output_expanded,
                             output_permutation,
                             outputTensor,
                             &retNodes,
                             fmt::format("{}/transpose_to_output_labels", m_name));

    retNodes.back()->replaceOutput(TENSOR_OFM, getOutput(TENSOR_OFM));
    return retNodes;
}

bool EinsumNode::validateNodeForGraph(const HabanaGraph& g) const
{
    return (g.getTraits().trainingGraph());
}

void EinsumNode::printParamsRawData() const
{
    BaseClass::printParamsRawData((void*)&m_params, sizeof(m_params));
}

template<typename T>
static size_t getContainerSize(T& c)
{
    return c.size() * sizeof(*c.data());
}

SifNodeParams EinsumNode::getShapeInferenceFunctionUserParams()
{
    if (!m_parsing_status) return nullptr;

    if (m_sifMetadataBuffer.empty())
    {
        m_sifMetadataBuffer.resize(getShapeInferenceFunctionUserParamsSize());

        SifEinsumMetadata* metadata = reinterpret_cast<SifEinsumMetadata*>(m_sifMetadataBuffer.data());

        metadata->output_dims = m_output_labels.size();
        for (size_t i = 0; i < m_inputs.size(); i++)
        {
            if (m_input_labels[i].data())
            {
                memcpy(metadata->input_dims_to_labels[i],
                       m_input_labels[i].data(),
                       getContainerSize(m_input_labels[i]));
            }
        }
        if (m_output_labels.data())
        {
            memcpy(metadata->output_dims_to_labels, m_output_labels.data(), getContainerSize(m_output_labels));
        }
        if (m_label_types.data())
        {
            memcpy(metadata->labels_to_types, m_label_types.data(), getContainerSize(m_label_types));
        }
    }

    return reinterpret_cast<SifNodeParams>(m_sifMetadataBuffer.data());
}

size_t EinsumNode::getShapeInferenceFunctionUserParamsSize() const
{
    return sizeof(SifEinsumMetadata);
}

void EinsumNode::AddDynamicReshapeShapeNode(const TensorPtr&   input,
                                            std::string*       reshapeEq,
                                            SizeArray*         outputSizes,
                                            SizeArray*         outputMinSizes,
                                            unsigned           shapeTensorRank,
                                            TensorPtr&         shape,
                                            NodeList*          retNodes,
                                            const std::string& nodeName)
{
    shapeTensorRank = shapeTensorRank == 0 ? 1 : shapeTensorRank;
    if (!ReshapeNeeded(input, outputSizes, shapeTensorRank))
    {
        LOG_TRACE(HABANA_NODE, "Einsum node - no need to create shape tensor");
        return;
    }

    dynamicReshapeParams* params = reinterpret_cast<dynamicReshapeParams*>(reshapeEq);

    shape = input->cloneGeometry();
    shape->setName(fmt::format("{}_shape_tensor", nodeName));
    shape->setShapeTensor(SHAPE_TENSOR);
    shape->reshape(shapeTensorRank, outputSizes->data(), nullptr, outputMinSizes->data());

    NodePtr dynamicReshape =
        NodeFactory::createNode({input}, {shape}, params, NodeFactory::dynamicReshapeNodeTypeName, nodeName);
    retNodes->push_back(dynamicReshape);

    return;
}

void EinsumNode::AddEinsumExpandShapeNode(const TensorVector&    inputs,
                                          std::vector<unsigned>* freeDims,
                                          SizeArray*             outputSizes,
                                          SizeArray*             outputMinSizes,
                                          unsigned               shapeTensorRank,
                                          TensorPtr&             shape,
                                          NodeList*              retNodes,
                                          const std::string&     nodeName)
{
    if (!ReshapeNeeded(inputs[0], outputSizes, shapeTensorRank))
    {
        LOG_TRACE(HABANA_NODE, "Einsum node - no need to do expand shape");
        return;
    }

    einsumExtractParams* params = reinterpret_cast<einsumExtractParams*>(freeDims);

    shape = inputs[0]->cloneGeometry();
    shape->setName(fmt::format("{}_shape_tensor", nodeName));
    shape->setShapeTensor(SHAPE_TENSOR);
    shape->reshape(shapeTensorRank, outputSizes->data(), nullptr, outputMinSizes->data());

    NodePtr einsumExpand =
        NodeFactory::createNode(inputs, {shape}, params, NodeFactory::einsumExpandShapeNodeTypeName, nodeName);
    retNodes->push_back(einsumExpand);

    return;
}