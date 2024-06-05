#include "two_dim_transpose_reshaper.h"
#include "transpose_strategies.h"
#include "node_factory.h"
#include "transpose_utils.h"

using TPD = TransposePermutationDim;

const TransposePermutationArray TwoDimTransposeReshaper::splitFcdPermutation = {TPD(2), TPD(0), TPD(1)};
const TransposePermutationArray TwoDimTransposeReshaper::splitScdPermutation = {TPD(1), TPD(2), TPD(0)};

TwoDimTransposeReshaper::TwoDimTransposeReshaper(const TransposeNode& originalTranspose)
: m_2DimTranspose(originalTranspose)
{
    const auto& input = originalTranspose.getInput(0);
    HB_ASSERT(input->getDim() == 2,
              "try to create TwoDimTransposeReshaper class for {} dim transpose",
              input->getDim());
    const auto& hal    = CompilationHalReader::getHalReader();
    m_clSizeInElements = hal->getCacheLineSizeInBytes() / input->getElementSizeInBytes();

    tryToReshape2DimTranspose();
}

const NodeVector& TwoDimTransposeReshaper::getWrappingNodes() const
{
    HB_ASSERT(isValid(), "try to get wrapping node, but there isn't valid reshape strategy");
    return m_wrappingNodes;
}

const TransposeNode& TwoDimTransposeReshaper::get3DimTranspose() const
{
    HB_ASSERT(isValid() && m_3DimTranspose.has_value(),
              "try to get 3 dim transpose, but there isn't valid reshape strategy");
    return m_3DimTranspose.value();
}

void TwoDimTransposeReshaper::createWrappingNodes()
{
    HB_ASSERT(m_3DimTranspose.has_value(), "3Dim transpose not created until this point");

    TensorPtr firstReshapeInput   = m_2DimTranspose.getInput(0);
    TensorPtr secondReshapeOutput = m_2DimTranspose.getOutput(0);

    // If the original 2 dim transpose is dynamic we need to extract the max size, since shape manipulation is
    // forbidden in case of dynamic dim.
    if (m_2DimTranspose.isDynamicShape())
    {
        const auto& aux =
            TransposeWithStaticShape().createAuxilaryNodes(TransposeNodeParams::fromNode(m_2DimTranspose));
        m_wrappingNodes = {aux.inferMaxNode, aux.transposeShapeNode, aux.identityNode};

        firstReshapeInput   = aux.newTransposeInput;
        secondReshapeOutput = aux.newTransposeOutput;
    }

    // Since in case of dynamic node we extract the max sizes, the reshape below is only for static shape,
    // therefore, shape tensors are not needed.
    m_wrappingNodes.push_back(NodeFactory::createNode({firstReshapeInput},
                                                      {m_3DimTranspose->getInput(0)},
                                                      nullptr,
                                                      NodeFactory::reshapeNodeTypeName,
                                                      m_2DimTranspose.getNodeName() + "/first_reshape"));
    m_wrappingNodes.push_back(NodeFactory::createNode({m_3DimTranspose->getOutput(0)},
                                                      {secondReshapeOutput},
                                                      nullptr,
                                                      NodeFactory::reshapeNodeTypeName,
                                                      m_2DimTranspose.getNodeName() + "/second_reshape"));
}

void TwoDimTransposeReshaper::create3DimTranspose(const Dim dimToSplit, const TSize divisor)
{
    const auto& originalInput = m_2DimTranspose.getInput(0);
    HB_ASSERT(originalInput->getSizeInElements(dimToSplit) % divisor == 0, "divisor must divide the split dim size");

    TSize sizes[3];
    sizes[dimToSplit == FCD ? 0 : 2] = originalInput->getSizeInElements(dimToSplit) / divisor;
    sizes[1]                         = divisor;
    sizes[dimToSplit == FCD ? 2 : 0] = originalInput->getSizeInElements(oppositeDim(dimToSplit));

    const TransposePermutationArray& permutation = dimToSplit == FCD ? splitFcdPermutation : splitScdPermutation;

    // Create the new tensors
    const auto& input3D = originalInput->clone(false, false, false);
    // Because in our case shape manipulation is forbidden in case of dynamic shape we create the tensors with
    // static shape, and if it needed we later wrap it with nodes that handle "dynamic shape to max shape".
    input3D->reshape(3, sizes, nullptr, sizes);
    const auto& output3D = getTensorAfterTranspose(*input3D, permutation);

    // create the new transpose node with the new tensors
    m_3DimTranspose = m_2DimTranspose;
    m_3DimTranspose->setName(m_2DimTranspose.getNodeName() + "/reshaped_to_3d");
    m_3DimTranspose->replaceInput(0, input3D);
    m_3DimTranspose->replaceOutput(0, output3D);
    m_3DimTranspose->setPermutation(permutation);
}

void TwoDimTransposeReshaper::tryToReshape2DimTranspose()
{
    const auto& originalSizes = m_2DimTranspose.getInput(0)->getAllSizesInElements();

    // prefer the dim that is bigger.
    Dim dimToSplit = originalSizes[FCD] > originalSizes[SCD] ? FCD : SCD;

    const auto& divisor = std::gcd(m_clSizeInElements, originalSizes[dimToSplit]);
    if (divisor == 1 || divisor == originalSizes[dimToSplit]) return;  // there is nothing to do
    create3DimTranspose(dimToSplit, divisor);
    createWrappingNodes();
}
