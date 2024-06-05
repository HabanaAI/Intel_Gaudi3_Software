#include "handle_memory_reuse.h"
#include "defs.h"
#include "habana_nodes.h"
#include "graph_editor.h"
#include "register_memory_coherence.h"
#include "tpc_node.h"
#include "types.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace MemoryReuseHandler
{
/*
    Define a "Positive Bounded Diphantine Equation" as:
        s[0]*x[0] + s[1]*x[1] + s[2]*x[2] + ... + s[dim]*x[dim] = b
    where s[i] and x[i] are positive integers. and x[i] are bounded:  0 <= x[i] < UpperBound[i]

    We call this kind of equation "Simple" if it complies with the special case where:
    s[i] = s[i-1] * a[i],  for some integers { a[i] }

    This function finds if a solution exists to a Simple Positive Bounded Diphantine Equation.
    --------------------------------------------------------------------------------------------------------------------
    How does it work? Example:  s[0]*x[0] + s[1]*x[1] + s[2]*x[2] = b

    - convert {s[i]} to its factors:   a[0]*x[0] + a[1]*a[0]*x[1] + a[2]*a[1]*a[0]*x[2] = b
    - take out the common term:        a[0]*(x[0] + a[1]*x[1] + a[2]*a[1]*x[2]) = b
    - for a solution to exits, b must be divisible by a[0]:  x[0] + a[1]*x[1] + a[1]*a[2]*x[2] = b_0      (b_0 = b/a[0])

    Now for a solution to exist, we must have: x[0] = (b_0 % a[1]) + x_remain    (where x_remain % a[1] == 0)
    if (b_0 % a[1]) >= UpperBound[0] we have no solution and we are done. Otherwise:
    b_1 <- floor(b / a[1])

    Now, since we don't want to miss a potential solution we will take x[0] to be (b_0 % a[1]):
    a[1]*(x[1] + x_remain/a[1]) + a[1]*a[2]*x[2] = (b_0 - (b_0 % a[1])) / a[1]
    and "save" x_remain for later (by merging it into x[1]):
    UpperBound[1] <-- UpperBound[1] + (x_remain / a[1]).

    Finish this step with: x[1] + a[2]*x[2] = b_1.
    Then continue the procedure until failing to find a suitable x[i], or finishing
    --------------------------------------------------------------------------------------------------------------------
    For more information on checking overlapping memory using Diphantine Equations lookup:
    https://fossies.org/linux/numpy/numpy/core/src/common/mem_overlap.c
*/
bool simpleDiphantine(const std::vector<uint64_t>& s, const std::vector<uint64_t>& shape, int64_t b)
{
    HB_ASSERT(s.size() == shape.size(), "strides and shape mismatch");
    HB_ASSERT(b >= 0, "expected base >= 0");

    uint64_t upperBound     = 1;  // upper bound for first iteration - 'b' must be divisible
    uint64_t previousStride = 1;
    for (int i = 0; i < s.size(); i++)
    {
        uint64_t a = s[i] / previousStride;  // calculate a[i]
        HB_ASSERT(a * previousStride == s[i], "not a Simple Diphantine Equation");

        uint64_t xSolution = b % a;
        if (xSolution >= upperBound) return false;                                    // no solution
        b /= a;                                                                       // update b
        upperBound     = shape[i] + (/* x_remain */ upperBound - 1 - xSolution) / a;  // update UpperBound[i+1]
        previousStride = s[i];
    }
    return b < upperBound;  // last remainder: x_dim = b_dim
}

/*
    in order to have a Simple Diphantine Equation, we need to strides to be sorted.
    for optimization, we "squash" together equal strides and fix the upper bounds (shape) accordingly.
*/
std::tuple<std::vector<uint64_t>, std::vector<uint64_t>> mergeStridesAndShape(const std::vector<uint64_t>& shape1,
                                                                              const std::vector<uint64_t>& shape2,
                                                                              const std::vector<uint64_t>& strides1,
                                                                              const std::vector<uint64_t>& strides2)
{
    HB_ASSERT(shape1.size() == strides1.size(), "shape and strides should have same size");
    HB_ASSERT(shape2.size() == strides2.size(), "shape and strides should have same size");
    // zip together strides and sizes for easier sorting
    std::vector<std::pair<uint64_t, uint64_t>> stridesAndShapes(shape1.size() + shape2.size());
    for (int i = 0; i < shape1.size(); i++)
    {
        stridesAndShapes[i] = {strides1[i], shape1[i]};
    }
    for (int i = 0; i < shape2.size(); i++)
    {
        stridesAndShapes[shape1.size() + i] = {strides2[i], shape2[i]};
    }

    // sort zipped container by strides
    std::sort(stridesAndShapes.begin(),
              stridesAndShapes.end(),
              [](const std::pair<uint64_t, uint64_t>& p1, const std::pair<uint64_t, uint64_t>& p2) {
                  return p1.first < p2.first;
              });

    // squash identical strides and create new coefficients
    std::vector<uint64_t> strides;
    std::vector<uint64_t> shapes;
    for (const auto& p : stridesAndShapes)
    {
        if (p.first == 0) continue;  // ignore 0 strides
        if (!strides.empty() && strides.back() == p.first)
        {
            shapes.back() += p.second - 1;  // if (0 <= x < ub_x) and (0 <= y < ub_x) then (0 <= x+y < ub_x + ub_y - 1)
        }
        else
        {
            strides.push_back(p.first);
            shapes.push_back(p.second);
        }
    }
    return std::tie(strides, shapes);
}

static std::pair<std::vector<uint64_t>, std::vector<uint64_t>> getShapeAndStrides(const TensorPtr& t)
{
    std::vector<uint64_t> shape(t->getDim()), strides(t->getDim());
    for (unsigned dim = 0; dim < t->getDim(); dim++)
    {
        shape[dim]   = t->getSizeInElements(dim);
        strides[dim] = t->getStrideInBytes(dim);
    }
    return std::make_pair(shape, strides);
}

bool isPartOfNonPersistentSection(const TensorPtr& t)
{
    return t->isPartOfWorkspaceSection() || t->isPartOfRMWSection();
}

// calculate byte offset of t from its real tensor
uint64_t getRealTensorOffset(const TensorPtr& t)
{
    uint64_t  offset     = 0;
    TensorPtr realTensor = t;
    while (realTensor->isAliasedTensor())
    {
        offset += realTensor->getAliasedByteOffset();
        realTensor = realTensor->getAliasTensor();
    }

    if (realTensor->isPersistent())
    {
        offset += realTensor->getMemorySectionOffset();
    }
    else if (isPartOfNonPersistentSection(realTensor))
    {
        offset += realTensor->getTensorAnnotation().nonPersistentSectionInfo.offsetFromBase.value();
    }

    return offset;
}

uint64_t getLastElementOffset(const TensorPtr& t)
{
    uint64_t ret = 0;
    for (unsigned i = 0; i < t->getDim(); i++)
    {
        ret += (t->getSizeInElements(i) - 1) * t->getStrideInBytes(i);
    }
    return ret;
}

uint64_t getLastElementOffset(const std::vector<uint64_t>& sizes, const std::vector<uint64_t>& strides)
{
    HB_ASSERT(sizes.size() == strides.size(), "shape and strides should have same size");
    uint64_t ret = 0;
    for (unsigned i = 0; i < sizes.size(); i++)
    {
        ret += (sizes[i] - 1) * strides[i];
    }
    return ret;
}

bool hasLowerOffset(const NodePtr& n1, const NodePtr& n2, bool isOutput)
{
    if (!n1 || !n2 || n1 == n2) return false;
    // if a node has more than 1 operand, "who writes/reads first" may be ambiguous
    unsigned numOperands1 = isOutput ? n1->getNumOutputsDataTensors() : n1->getNumInputsDataTensors();
    unsigned numOperands2 = isOutput ? n2->getNumOutputsDataTensors() : n2->getNumInputsDataTensors();
    if (numOperands1 != 1 || numOperands2 != 1) return false;

    const TensorPtr& t1 = isOutput ? n1->getOutput(0) : n1->getInput(0);
    const TensorPtr& t2 = isOutput ? n2->getOutput(0) : n2->getInput(0);
    return (t1 && t2 && (Tensor::getRealTensor(t1) == Tensor::getRealTensor(t2)) &&
            getRealTensorOffset(t1) < getRealTensorOffset(t2));
}

bool hasLowerWritingOffset(const NodePtr& n1, const NodePtr& n2)
{
    return hasLowerOffset(n1, n2, true);
}

bool hasLowerReadingOffset(const NodePtr& n1, const NodePtr& n2)
{
    return hasLowerOffset(n1, n2, false);
}

bool sameRealTensor(const TensorPtr& t1, const TensorPtr t2)
{
    return Tensor::getRealTensor(t1) == Tensor::getRealTensor(t2);
}

bool sameMemorySection(const TensorPtr& t1, const TensorPtr& t2)
{
    const TensorPtr& t1Real = Tensor::getRealTensor(t1);
    const TensorPtr& t2Real = Tensor::getRealTensor(t2);
    return (t1Real->isPersistent() && t2Real->isPersistent() &&
            t1Real->getMemorySectionID() == t2Real->getMemorySectionID()) ||
           (isPartOfNonPersistentSection(t1Real) && isPartOfNonPersistentSection(t2Real) &&
            t1Real->getTensorAnnotation().nonPersistentSectionInfo.sectionId ==
                t2Real->getTensorAnnotation().nonPersistentSectionInfo.sectionId);
}

bool isStridedOverlap(const std::vector<uint64_t>& sizes1,
                      const std::vector<uint64_t>& sizes2,
                      const std::vector<uint64_t>& strides1,
                      const std::vector<uint64_t>& strides2,
                      uint64_t                     offset1,
                      uint64_t                     offset2)
{
    /*
        modified baseB to compensate negative strides:
        the original equation is:
        x_a[0]*s_a[0] + x_a[1]*s_a[1] + ... + baseA = x_b[0]*s_b[0] + x_b[1]*s_b[1] + ... + baseB
        when we move the strides from the right hand side to the left hand side we get negative strides.
        Since we want all-positive coefficients we need to convert the original strides to negative strides -
        i.e., start counting from the end of the tensor  ==> "negative" baseB' = last address of B
        converting the equation to:
        x_a[0]*s_a[0] + x_b[0]*s_b[0] + x_a[1]*s_a[1] + x_b[1]*s_b[1] + ... = baseB' - baseA
    */
    uint64_t base = offset2 + getLastElementOffset(sizes2, strides2) - offset1;
    HB_ASSERT(offset2 + getLastElementOffset(sizes2, strides2) >= offset1, "no linear memory overlap!");
    if (base == 0) return true;  // trivial solution: x_[i] = 0 for all i.

    auto [strides, shapes] = mergeStridesAndShape(sizes1, sizes2, strides1, strides2);

    // if it's not a simple Diphantine Equation consider as overlapping
    for (int i = 1; i < strides.size(); i++)
    {
        if (strides[i] % strides[i - 1]) return true;
    }
    return simpleDiphantine(strides, shapes, base);
}

bool isStridedOverlap(const TensorPtr& t1, const TensorPtr t2)
{
    if (!isDenseOverlap(t1, t2)) return false;
    if (!sameRealTensor(t1, t2) && !sameMemorySection(t1, t2)) return false;
    uint64_t baseA = getRealTensorOffset(t1);
    uint64_t baseB = getRealTensorOffset(t2);

    auto [sizes1, strides1] = getShapeAndStrides(t1);
    auto [sizes2, strides2] = getShapeAndStrides(t2);

    return isStridedOverlap(sizes1, sizes2, strides1, strides2, baseA, baseB);
}

// check for overlapping linear memory bounds
bool isDenseOverlap(const TensorPtr& t1, const TensorPtr& t2)
{
    HB_ASSERT_PTR(t1);
    HB_ASSERT_PTR(t2);
    if (!sameRealTensor(t1, t2) && !sameMemorySection(t1, t2)) return false;
    uint64_t start1 = getRealTensorOffset(t1);
    uint64_t end1   = start1 + t1->getTotalSizeInBytes();
    uint64_t start2 = getRealTensorOffset(t2);
    uint64_t end2   = start2 + t2->getTotalSizeInBytes();
    return start1 < end2 && end1 > start2;
}

// check if t1 and t2 have the same strides, shape, and base offset
bool isExactInplace(const TensorPtr& t1, const TensorPtr& t2)
{
    if (getRealTensorOffset(t1) != getRealTensorOffset(t2)) return false;
    if (!t1->compareGeometry(*t2)) return false;
    for (unsigned i = 0; i < t1->getDim(); i++)
    {
        if (t1->getStrideInBytes(i) != t2->getStrideInBytes(i)) return false;
    }
    return true;
}

bool isExactOverlap(const TensorPtr& t1, const TensorPtr& t2)
{
    if (!sameRealTensor(t1, t2) && !sameMemorySection(t1, t2)) return false;
    return isExactInplace(t1, t2);
}

bool isReuseAllowed(const NodePtr& n, const TensorPtr& out, const TensorPtr& t2)
{
    if (n->getNodeType() == Node::TYPE_DMA && std::static_pointer_cast<DMANode>(n)->isBroadcast()) return true;

    // exact inplace case
    if (isExactInplace(out, t2))
    {
        if (isMemcpy(*n)) return true;

        const auto& reuseMap = n->getReusableInputs();  // allow inplace op if it supports it
        auto        it       = reuseMap.find(out);
        return (it != reuseMap.end()) && std::find(it->second.begin(), it->second.end(), t2) != it->second.end();
    }

    // potential dangerous overlap
    return !isStridedOverlap(out, t2);
}

void handleMemoryReuse(HabanaGraph& graph)
{
    using ReuseCollision = std::tuple<NodePtr, TensorPtr, TensorPtr>;
    std::vector<ReuseCollision> reuseCollisions;

    // find potential memory reuse collisions - node operands that share the same memory
    for (const NodePtr& n : graph.getNodes())
    {
        if (n->isLogicalOperation()) continue;
        for (const TensorPtr& out : n->getOutputs())
        {
            if (!out) continue;
            for (const TensorPtr& t2 : n->getOperands())  // search for collision in read+write or write+write
            {
                if (!t2 || out == t2) continue;
                if (isDenseOverlap(out, t2))
                {
                    reuseCollisions.push_back({n, out, t2});  // output reusing a different operands memory
                }
            }
        }
    }

    // sort collisions in reverse coherency order
    if (graph.getGraphAnnotation().memoryCoherence)
    {
        TensorCoherenceMapping::CoherencyComparator comparator(*graph.getGraphAnnotation().memoryCoherence);
        std::sort(reuseCollisions.begin(),
                  reuseCollisions.end(),
                  [&](const ReuseCollision& a, const ReuseCollision& b) {
                        return comparator(std::get<1>(b), std::get<1>(a));
                  });
    }

    TensorSet handledOutputs;
    for (const auto& [node, output, otherTensor] : reuseCollisions)
    {
        if (handledOutputs.find(output) != handledOutputs.end()) continue;
        if (handledOutputs.find(otherTensor) != handledOutputs.end()) continue;
        // check if this is indeed a reuse of memory, and if so - check if it's allowed
        if (!isReuseAllowed(node, output, otherTensor))
        {
            LOG_INFO(GC,
                     "A possible illegal memory reuse was detected in node {}, between tensor {} and {}. A memcopy "
                     "will be planted",
                     node->getNodeName(),
                     output->getName(),
                     otherTensor->getName());
            GraphEditor::insertMemcpyForOutput(graph, node,
                                               output);  // not allowed - plant a memcopy
            handledOutputs.insert(output);               // mark as handled
        }
    }
}
};  // namespace MemoryReuseHandler

bool handleMemoryReuse(HabanaGraph& graph)
{
    MemoryReuseHandler::handleMemoryReuse(graph);
    return true;
}
