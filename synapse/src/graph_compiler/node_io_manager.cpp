#include "defs.h"
#include "layout.h"
#include "synapse_common_types.h"
#include "synapse_types.h"
#include "node_io_manager.h"
#include "tpc_node.h"
#include "transpose_permutation.h"
#include "kernel_db.h"
#include "habana_graph.h"
#include "graph_traits.h"
#include "transpose_node.h"
#include "tpc_kernel_loader.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

using namespace gc;

static const LayoutVector conv3DInputLayouts =
    {Layout("CWHDN"), Layout("KCSRQ"), Layout(), Layout("CWHDN"), Layout(), Layout()};
static const LayoutVector conv2DInputLayouts =
    {Layout("CWHN"), Layout("KCSR"), Layout(), Layout("CWHN"), Layout(), Layout()};

static const LayoutVector conv3DOutputLayouts = {Layout("CWHDN"), Layout("CWHDN")};
static const LayoutVector conv2DOutputLayouts = {Layout("CWHN"), Layout("CWHN")};

static const LayoutVector dedw3DInputLayouts = {Layout("CWHDN"), Layout("CWHDN")};
static const LayoutVector dedw2DInputLayouts = {Layout("CWHN"), Layout("CWHN")};

static const LayoutVector dedw3DOutputLayouts = {Layout("KCSRQ"), Layout("KCSRQ")};
static const LayoutVector dedw2DOutputLayouts = {Layout("KCSR"), Layout("KCSR")};

static const LayoutVector dedx3DInputLayouts = {Layout("CWHDN"), Layout("KCSRQ"), Layout("CWHDN")};
static const LayoutVector dedx2DInputLayouts = {Layout("CWHN"), Layout("KCSR"), Layout("CWHN")};

static const LayoutVector dedx3DOutputLayouts = {Layout("CWHDN"), Layout("CWHDN")};
static const LayoutVector dedx2DOutputLayouts = {Layout("CWHN"), Layout("CWHN")};

static const LayoutVector cudBnFwdOutputLayouts = {Layout("CWHN"), Layout(), Layout(), Layout(), Layout()};

static const LayoutVector cudBnBwdInputLayouts =
    {Layout("CWHN"), Layout("CWHN"), Layout(), Layout(), Layout(), Layout()};
static const LayoutVector cudBnBwdOutputLayouts = {Layout("CWHN"), Layout(), Layout(), Layout("CWHN")};

// Override TPC layouts

static const std::vector<Node::eNodeType> restrictedNodes = {Node::TYPE_STRIDED_VIEW,
                                                             Node::TYPE_STRIDED_INSERT,
                                                             Node::TYPE_INTERNAL_BROADCAST,
                                                             Node::TYPE_GEMM,
                                                             Node::TYPE_GEMM_DEDW,
                                                             Node::TYPE_GEMM_DEDX,
                                                             Node::TYPE_BATCH_GEMM,
                                                             Node::TYPE_BATCH_GEMM_DEDX,
                                                             Node::TYPE_BATCH_GEMM_DEDW,
                                                             Node::TYPE_MASKED_BATCH_GEMM};

static bool _isAllLayoutsEqual(const LayoutVector& layouts)
{
    return std::all_of(layouts.begin() + 1, layouts.end(), [&](const auto& layout) { return layouts[0] == layout; });
}

static bool _isAllDontCare(const LayoutVector& layouts)
{
    return std::all_of(layouts.begin(), layouts.end(), [&](const auto& layout) { return layout.isDontCare(); });
}

static bool _isAllNotAvailable(const LayoutVector& layouts)
{
    static const Layout NA_LAYOUT("NotAvailable");
    return std::all_of(layouts.begin(), layouts.end(), [&](const auto& layout) { return NA_LAYOUT == layout; });
}

NodeIOManager::NodeIOManager(Node* node)
: m_node(node),
  m_supportedInputLayouts(LayoutVector(node->getNumInputs())),   // default layouts - don't care for all inputs
  m_supportedOutputLayouts(LayoutVector(node->getNumOutputs()))  // default layouts - don't care for all outputs
{
}

NodeIOManager::~NodeIOManager()
{
}

const LayoutVector& NodeIOManager::getInputSupportedLayouts() const
{
    return m_supportedInputLayouts;
}

const LayoutVector& NodeIOManager::getOutputSupportedLayouts() const
{
    return m_supportedOutputLayouts;
}

void NodeIOManager::setSupportedLayoutsHelper(LayoutVector&       inputLayouts,
                                              LayoutVector&       outputLayouts,
                                              const LayoutVector& supportedInputLayouts,
                                              const LayoutVector& supportedOutputLayouts) const
{
    HB_ASSERT(supportedInputLayouts.size() >= m_node->getNumInputs(),
              "supportedInputLayouts doesn't have enough entries.");
    inputLayouts = LayoutVector(supportedInputLayouts.begin(), supportedInputLayouts.begin() + m_node->getNumInputs());

    HB_ASSERT(supportedOutputLayouts.size() >= m_node->getNumOutputs(),
              "supportedOutputLayouts doesn't have enough entries.");
    outputLayouts =
        LayoutVector(supportedOutputLayouts.begin(), supportedOutputLayouts.begin() + m_node->getNumOutputs());
}

void NodeIOManager::setSupportedLayoutsConv(LayoutVector&       inputLayouts,
                                            LayoutVector&       outputLayouts,
                                            const LayoutVector& supportedInputLayouts,
                                            const LayoutVector& supportedOutputLayouts,
                                            const LayoutVector& supported3DInputLayouts,
                                            const LayoutVector& supported3DOutputLayouts) const
{
    const auto& inTensor = m_node->getInput(0);
    if (inTensor->getDim() == CONV_3D_TENSOR_DIM)
    {
        setSupportedLayoutsHelper(inputLayouts, outputLayouts, supported3DInputLayouts, supported3DOutputLayouts);
    }
    else
    {
        setSupportedLayoutsHelper(inputLayouts, outputLayouts, supportedInputLayouts, supportedOutputLayouts);
    }
}

// This utility used for both Supported and Actual IO layouts.
void NodeIOManager::selectLayoutsConvNode(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const
{
    setSupportedLayoutsConv(inputLayouts,
                            outputLayouts,
                            conv2DInputLayouts,
                            conv2DOutputLayouts,
                            conv3DInputLayouts,
                            conv3DOutputLayouts);
}

void NodeIOManager::selectLayoutsDedwNode(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const
{
    setSupportedLayoutsConv(inputLayouts,
                            outputLayouts,
                            dedw2DInputLayouts,
                            dedw2DOutputLayouts,
                            dedw3DInputLayouts,
                            dedw3DOutputLayouts);
}

void NodeIOManager::selectLayoutsDedxNode(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const
{
    setSupportedLayoutsConv(inputLayouts,
                            outputLayouts,
                            dedx2DInputLayouts,
                            dedx2DOutputLayouts,
                            dedx3DInputLayouts,
                            dedx3DOutputLayouts);
}

void NodeIOManager::selectLayoutsByType(LayoutVector& inputLayouts, LayoutVector& outputLayouts) const
{
    switch (m_node->getNodeType())
    {
        case Node::TYPE_CONVOLUTION:
            selectLayoutsConvNode(inputLayouts, outputLayouts);
            break;
        case Node::TYPE_DEDW:
            selectLayoutsDedwNode(inputLayouts, outputLayouts);
            break;
        case Node::TYPE_DEDX:
            selectLayoutsDedxNode(inputLayouts, outputLayouts);
            break;
        case Node::TYPE_FROBENIUS_NORM_NODE:
            inputLayouts  = {Layout(std::string("CWHN", m_node->getInput(0)->getDim()))};
            outputLayouts = {Layout()};
            break;
        case Node::TYPE_NMS:
            inputLayouts  = {Layout("BXN"), Layout("BCN")};
            outputLayouts = {Layout("LX")};
            break;
        case Node::TYPE_ROTATE:
            inputLayouts  = {Layout("NCHW")};
            outputLayouts = {Layout("NCHW")};
            break;
        default:
            // don't care nodes
            inputLayouts  = LayoutVector(m_node->getNumInputs());
            outputLayouts = LayoutVector(m_node->getNumOutputs());
            break;
    }
}

bool NodeIOManager::isDontCareNode() const
{
    switch (m_node->getNodeType())
    {
        case Node::TYPE_CONVOLUTION:
        case Node::TYPE_DEDW:
        case Node::TYPE_DEDX:
        case Node::TYPE_FROBENIUS_NORM_NODE:
        case Node::TYPE_NMS:
        case Node::TYPE_ROTATE:
            return false;
        default:
            return true;
    }
}

void NodeIOManager::setSupportedLayouts(const LayoutVector& supportedInputLayouts,
                                        const LayoutVector& supportedOutputLayouts)
{
    m_supportedInputLayouts  = supportedInputLayouts;
    m_supportedOutputLayouts = supportedOutputLayouts;
    m_allDontCare            = _isAllDontCare(m_supportedInputLayouts) && _isAllDontCare(m_supportedOutputLayouts);
}

void NodeIOManager::setDefaultIOLayouts()
{
    Layout defaultLayout("CWHN");
    m_supportedInputLayouts.resize(m_node->getNumInputs(), defaultLayout);
    m_supportedOutputLayouts.resize(m_node->getNumOutputs(), defaultLayout);
    m_allDontCare = false;
}

bool NodeIOManager::setSupportedIOLayouts(synDeviceType deviceType)
{
    // don't reset the supported data layout if it is still all don't care
    if (!isDontCareNode() || !isAllDontCare() || m_node->getNumInputs() != m_supportedInputLayouts.size() ||
        m_node->getNumOutputs() != m_supportedOutputLayouts.size())
    {
        LayoutVector inputLayouts, outputLayouts;
        selectLayoutsByType(inputLayouts, outputLayouts);
        setSupportedLayouts(inputLayouts, outputLayouts);

        LOG_DEBUG(DATA_LAYOUT,
              "Setting these supported layouts for node {} (guid {}): inputs:{}, outputs:{}",
              m_node->getNodeName(),
              m_node->getGUID(),
              Layout::toString(inputLayouts),
              Layout::toString(outputLayouts));
    }
    return true;
}

void NodeIOManager::permuteInternal(const LayoutVector& fromLayouts,
                                    const LayoutVector& toLayouts,
                                    PermutationVector&  permutations) const
{
    HB_ASSERT(fromLayouts.size() == toLayouts.size(), "fromLayouts and toLayouts sizes do not match");
    HB_ASSERT(permutations.empty(), "permutations is not empty");
    permutations.resize(fromLayouts.size());
    for (unsigned i = 0; i < permutations.size(); ++i)
    {
        if (fromLayouts[i].isDontCare()) continue;
        fromLayouts[i].getPermutation(toLayouts[i], permutations[i]);
    }
}

bool NodeIOManager::validateAllRequiredLayoutsExist(const LayoutVector& supportedLayouts,
                                                    const LayoutVector& userLayouts) const
{
    if (_isAllDontCare(userLayouts)) return true;  // no layouts passed from the user
    for (unsigned i = 0; i < supportedLayouts.size(); ++i)
    {
        const Layout& supported = supportedLayouts[i];
        if (!supported.isDontCare())
        {
            if (supported.dims() < 4) continue;  // TODO: remove this exception when we pass user layouts for all dims

            if (userLayouts[i].isDontCare())
            {
                LOG_ERR(DATA_LAYOUT,
                        "Got don't care user layout for a node tensor index [{}] that has a required supported layout {}, node name: {}",
                        i, supported.toString(), m_node->getNodeName());
                return false;
            }
        }
    }
    return true;
}

bool NodeIOManager::permutationRequired(const LayoutVector& supportedLayouts, const LayoutVector& userLayouts) const
{
    for (unsigned i = 0; i < supportedLayouts.size(); ++i)
    {
        if (!supportedLayouts[i].isDontCare() && supportedLayouts[i] != userLayouts[i])
        {
            return true;
        }
    }
    return false;
}

bool NodeIOManager::permutationsRequired() const
{
    if (isAllDontCare()) return false;
    return permutationRequired(m_supportedInputLayouts, m_node->getInputLayouts()) ||
           permutationRequired(m_supportedOutputLayouts, m_node->getOutputLayouts());
}

bool NodeIOManager::validateLayouts() const
{
    const auto& inputLayouts  = m_node->getInputLayouts();
    const auto& outputLayouts = m_node->getOutputLayouts();

    HB_ASSERT(m_supportedInputLayouts.size() == inputLayouts.size(),
              "Each node input IO layout should have a matching supported layout");
    HB_ASSERT(m_supportedOutputLayouts.size() == outputLayouts.size(),
              "Each node output IO layout should have a matching supported layout");

    bool result = true;
    if (!validateAllRequiredLayoutsExist(m_supportedInputLayouts, inputLayouts))
    {
        result = false;
        LOG_ERR(DATA_LAYOUT, "Input layout validation failed for node: {}", m_node->getNodeName());
    }
    if (!validateAllRequiredLayoutsExist(m_supportedOutputLayouts, outputLayouts))
    {
        result = false;
        LOG_ERR(DATA_LAYOUT, "Output layout validation failed for node: {}", m_node->getNodeName());
    }

    return result;
}

void NodeIOManager::permute(PermutationVector& inputPermutations, PermutationVector& outputPermutations) const
{
    LOG_TRACE(DATA_LAYOUT,
              "Generating input permutations from {} to {}",
              Layout::toString(m_node->getInputLayouts()),
              Layout::toString(m_supportedInputLayouts));
    permuteInternal(m_node->getInputLayouts(), m_supportedInputLayouts, inputPermutations);
    LOG_TRACE(DATA_LAYOUT,
              "Generating output permutations from {} to {}",
              Layout::toString(m_supportedOutputLayouts),
              Layout::toString(m_node->getOutputLayouts()));
    permuteInternal(m_supportedOutputLayouts, m_node->getOutputLayouts(), outputPermutations);
    LOG_TRACE(DATA_LAYOUT, "Input permutations: {}", Permutation::toString(inputPermutations));
    LOG_TRACE(DATA_LAYOUT, "Output permutations: {}", Permutation::toString(outputPermutations));
}

bool NodeIOManager::nodeisAllDontCare() const
{
    return _isAllDontCare(m_node->getInputLayouts()) && _isAllDontCare(m_node->getOutputLayouts());
}

bool NodeIOManager::validateAndSetActualIOLayouts()
{
    LayoutVector inputLayouts, outputLayouts;
    // If user has manually set any node layout, don't override it
    if (!_isAllDontCare(m_node->getInputLayouts()) || !_isAllDontCare(m_node->getOutputLayouts()))
    {
        LOG_TRACE(DATA_LAYOUT,
                    "Got IO layout from user. Node: {}, Type: {}",
                    m_node->getNodeName(),
                    m_node->getNodeTypeStr());
        return true;
    }
    // we assume layouts are matching our definitions in case node are given, so set the node actual layouts as
    // the node's supported layouts - must run after the supported layouts pass is run
    m_node->setInputLayouts(m_supportedInputLayouts);
    m_node->setOutputLayouts(m_supportedOutputLayouts);
    return true;
}

LayoutVector NodeIOManager::getInputInferLayouts(const LayoutVector& outputLayouts, synDeviceType deviceType)
{
    HB_ASSERT(outputLayouts.size() == m_node->getNumOutputs(), "outputLayouts size shall be equal to outputs number");

    if (!isAllDontCare())
    {
        // If supported layouts are defined (at least one tensor is not DontCare) then return them
        return m_supportedInputLayouts;
    }

    unsigned numInputs = m_node->getNumInputs();
    LayoutVector inputInferLayouts(numInputs, gc::Layout("NotAvailable"));

    bool isOutLayoutsDontCare = _isAllDontCare(outputLayouts);
    if (isOutLayoutsDontCare)
    {
        // if outputLaytouts is all set as dont care then return dont care for input layouts
        std::fill(inputInferLayouts.begin(), inputInferLayouts.end(), gc::Layout());
        return inputInferLayouts;
    }

    // If transpose node and has output layout then inverse the output layout and return as input layout
    if (m_node->getNodeType() == Node::TYPE_INTERNAL_TRANSPOSE && !_isAllNotAvailable(outputLayouts))
    {
        TransposeNode* transposeNode = dynamic_cast<TransposeNode*>(m_node);
        HB_ASSERT_PTR(transposeNode);

        gc::Permutation perm(transposeNode->permutation());
        gc::Permutation invPerm(perm.getInversePermutation());
        inputInferLayouts[0] = outputLayouts[0].permute(invPerm);
        return inputInferLayouts;
    }

    return inputInferLayouts;
}


TPCNodeIOManager::TPCNodeIOManager(TPCNode* node) : NodeIOManager(node)
{
}

template<typename DataLayoutStruct>
static gc::Layout getLayoutFromGlueCodeLayout(const DataLayoutStruct& glueCodeLayout)
{
    constexpr auto N = std::size(decltype(DataLayoutStruct::layout) {});

    // The glue code layout char array isn't null-terminated - copy it to a null-terminated array
    char layout[N + 1] = {0};

    // Remove asterisks from the layout:
    // Glue code sets the layout as 5 asterisks ("*****" is don't care) before setting the real layout on top of it,
    // so we need to change "CWHN*" to "CWHN" (for example), and "*****" should be an empty string (our definition of don't care).
    // TODO - remove the support for the deprecated 'x' symbol when perflib update the new convention
    std::copy_if(std::begin(glueCodeLayout.layout), std::end(glueCodeLayout.layout), std::begin(layout), [](char c) {
        return c != '*' && c != 'x';
    });

    // The layout is composed of meanning-less symbols representing a permutation, for ex. AB can be either AB or BA,
    // Therfore a sub-two symbol layout is meaningless for permuting and we can replace it with a don't-care.
    if (layout[0] == 0) return Layout();
    if (layout[1] == 0)
    {
        LOG_DEBUG(DATA_LAYOUT, "Single dim layout \"{}\" replaced with \"Don'tCare\"", layout);
        return Layout();
    }
    // A sequence of #'s (in length of the tensor’s rank) represents a restricted layout
    if (layout[0] == '#')
    {
        LOG_TRACE(DATA_LAYOUT, "Layout \"{}\" is marked as restricted", layout);
        Layout restrictedLayout = Layout();
        restrictedLayout.setAsRestricted();
        return restrictedLayout;
    }
    return Layout(std::string(layout));
}

bool TPCNodeIOManager::setSupportedIOLayouts(synDeviceType deviceType)
{
    TPCNode* tpcNode = static_cast<TPCNode*>(m_node);
    return setGuidSupportedLayouts(*tpcNode, deviceType);
}

bool TPCNodeIOManager::setGuidSupportedLayouts(const TPCNode& tpcNode, synDeviceType deviceType)
{
    if (deviceTypeToDeviceID(deviceType) == tpc_lib_api::DEVICE_ID_MAX)
    {
        LOG_ERR(DATA_LAYOUT, "setGuidSupportedLayouts: Invalid device type: {}", deviceType);
        return false;
    }
    unsigned                             layoutCount = 0;
    const auto&                          guid        = tpcNode.getGUID();
    tpc_lib_api::GlueCodeReturn          ret         = tpc_lib_api::GLUE_FAILED;
    tpc_lib_api::HabanaKernelParams      params      = {};
    tpc_lib_api::NodeDataLayouts         layouts     = {};
    TPCLibTensorOperandsVector           tensorOperands;
    TPCLibTensorOperandsDataLayoutVector operandTensorLayouts;
    TPCLibTensorDataLayoutVector         shapeTensorLayouts;

    KernelInstantiationWrapper::updateGlueCodeParamsAndLayoutsPointers(tpcNode,
                                                                       params,
                                                                       tensorOperands,
                                                                       layouts,
                                                                       operandTensorLayouts,
                                                                       shapeTensorLayouts);
    KernelInstantiationWrapper::initParams(params, tpcNode, deviceTypeToDeviceID(deviceType));

    try
    {
        ret = KernelDB::instance().GetKernelSupportedDataLayouts(&params,
                                                                 nullptr,
                                                                 &layoutCount,
                                                                 tpcNode.getGUIDAndHash());
    }
    catch (std::exception& e)
    {
        LOG_DEBUG(DATA_LAYOUT,
                  "perf_lib GetSupportedDataLayouts for guid {} isn't implemented, "
                  "setting all input/output layouts for node {} as don't-care",
                  guid,
                  m_node->getNodeName());
        setSupportedLayouts(LayoutVector(m_node->getNumInputs()), LayoutVector(m_node->getNumOutputs()));
        return true;
    }
    if (ret != tpc_lib_api::GLUE_SUCCESS || layoutCount == 0)
    {
        LOG_ERR(DATA_LAYOUT, "Glue Code GetSupportedLayouts failed for guid {}", guid);
        setSupportedLayouts(LayoutVector(m_node->getNumInputs()), LayoutVector(m_node->getNumOutputs()));
        return false;
    }

    ret = KernelDB::instance().GetKernelSupportedDataLayouts(&params, &layouts, &layoutCount, tpcNode.getGUIDAndHash());
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_ERR(DATA_LAYOUT,
                "Glue Code GetSupportedLayouts for guid {}, and node {} returns error {}",
                guid,
                m_node->getNodeName(),
                KernelDB::parseReturnValue(ret));
        return false;
    }

    // get supported input and output layouts
    LayoutVector inputLayouts;
    LayoutVector outputLayouts;
    {
        const unsigned inputNum {m_node->getNumInputs()};
        const unsigned outputNum {m_node->getNumOutputs()};
        inputLayouts.reserve(inputNum);
        outputLayouts.reserve(outputNum);

        // PerfLib only supports a single layout
        HB_DEBUG_VALIDATE(layoutCount == 1);

        std::size_t shapeIdx  = 0;
        std::size_t inputIdx  = 0;
        std::size_t outputIdx = 0;

        // get supported input layouts
        const auto& inputTensors = m_node->getInputs();
        for (unsigned i = 0; i < inputNum; ++i)
        {
            if (inputTensors[i] != nullptr)
            {
                const auto& layout =
                    inputTensors[i]->isShapeTensor() ? layouts.shapeTensors[shapeIdx++] : layouts.inputs[inputIdx++];
                inputLayouts.push_back(getLayoutFromGlueCodeLayout(layout));
            }
            else
            {
                inputLayouts.emplace_back();  // The default layout is "don't care"
            }
        }

        // get supported output layouts
        const auto& outputTensors = m_node->getOutputs();
        for (unsigned i = 0; i < outputNum; ++i)
        {
            if (outputTensors[i] != nullptr)
            {
                const auto& layout =
                    outputTensors[i]->isShapeTensor() ? layouts.shapeTensors[shapeIdx++] : layouts.outputs[outputIdx++];
                outputLayouts.push_back(getLayoutFromGlueCodeLayout(layout));
            }
            else
            {
                outputLayouts.emplace_back();  // The default layout is "don't care"
            }
        }
    }

    LOG_DEBUG(DATA_LAYOUT,
              "Setting these supported layouts for node {} (guid {}): inputs:{}, outputs:{}",
              m_node->getNodeName(),
              guid,
              Layout::toString(inputLayouts),
              Layout::toString(outputLayouts));
    setSupportedLayouts(inputLayouts, outputLayouts);
    return true;
}

LayoutVector TPCNodeIOManager::getInputInferLayouts(const LayoutVector& outputLayouts, synDeviceType deviceType)
{
    LayoutVector inputInferLayouts = NodeIOManager::getInputInferLayouts(outputLayouts, deviceType);
    if (!_isAllNotAvailable(inputInferLayouts))
    {
        return inputInferLayouts;
    }

    TPCNode* tpcNode = static_cast<TPCNode*>(m_node);

    // Check if the tpc node is an elementwise node
    bool isNodeElementwise = tpcNode->isSeparable(deviceTypeToDeviceID(deviceType));
    if (isNodeElementwise)
    {
        // if all output layouts are equal set the layout value for inputs
        if (_isAllLayoutsEqual(outputLayouts))
        {
            std::fill(inputInferLayouts.begin(), inputInferLayouts.end(), outputLayouts[0]);
        }
    }

    return inputInferLayouts;
}

LogicalNodeIOManager::LogicalNodeIOManager(LogicalOpNode* node) : NodeIOManager(node)
{
}

void LogicalNodeIOManager::permute(PermutationVector& inputPermutations, PermutationVector& outputPermutations) const
{
    // TODO handling logical nodes parameters should be implemented in all logical nodes, when accessing them.
    // right now they access them in their constructor, that also modify other parameters based on them.
    // this should be changed, so they do it only after the data layout was adjusted (in runLogicalOperation?).
    // Separate ticket was opened. Until then, not permuting (with addition to restricted layout) will promise that
    // the user layout will be preserved.
}