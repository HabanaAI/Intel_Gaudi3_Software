#pragma once

// synapse top-level
#include "access_pattern.h"
#include "tpc_kernel_lib_interface.h"
#include "tpc_kernel_lib_interface_private.h"
#include "generic_parameters_node.h"       // for GenericParametersNode
#include "kernel_instantiation_wrapper.h"  // for KernelInstantiationWrapper
// synapse graph_compiler
#include "types.h"                         // for TensorPtr
#include "graph_annotation.h"              // for AuxiliaryTensors
#include "data_type_utils.h"               // for extractDtypeFromGUID
// tpc_kernels top-level
#include "tpc_elf_api.hpp"                 // for TpcElfTools

// for miltinode sif info
#include "multi_sif.h"
#include <bitset>

class HalReader;
class CodeGenerator;

struct KernelInfo
{
    char*    kernelBinary = nullptr;
    kernelID kernelId     = 0;
    unsigned kernelSize   = 0;
    // cached kernel if exists or Elf Buffer if we should cache the Elf binary
    std::shared_ptr<char> cachedBinary;
};

class TPCNode : public GenericParametersNode,
                public MultiSifNodeInfoHelper
{
public:
    using BaseClass = GenericParametersNode;

    TPCNode(const TensorVector& inputs,
            const TensorVector& outputs,
            std::string_view    name,
            UserParams          params     = nullptr,
            unsigned            paramsSize = 0);

    ~TPCNode();
    TPCNode& operator=(const TPCNode& other);

    static NodePtr createNode(const TensorVector& inputs,
                              const TensorVector& outputs,
                              UserParams          userParams,
                              unsigned            userParamsSize,
                              std::string_view    guid,
                              std::string_view    name);

    std::string_view                          getGUIDWithoutDtype() const;
    std::string_view                          getDtypeFromGUID() const { return extractDtypeFromGUID(getGUID()); }
    bool isGuidPrefix(std::string_view prefix) const { return startsWith(getGUID(), prefix); }
    const tpc_lib_api::HabanaKernelInstantiation& getInstance() const
    {
        HB_ASSERT(m_instanceWrapper.isInstantiated(), "Kernel not initialized");
        return m_instanceWrapper.getInstance();
    }
    void             setKernelOffsetInSection(deviceAddrOffset offset) { m_kernelOffsetInSection = offset; }
    deviceAddrOffset getKernelOffsetInSection() const { return m_kernelOffsetInSection.value(); }
    unsigned         getKernelSize() const;
    kernelID         getUniqueID() const { return m_uniqueID; }
    KernelInfo       getKernelInfo() const;
    std::string_view getEngineTypeStr() const override;
    unsigned         getNumParams() const
    {
        HB_ASSERT(m_instanceWrapper.isInstantiated(), "Kernel not initialized");
        return getInstance().kernel.paramsNr;
    }
    unsigned                                  getNumInputsToKernel() const;
    const TensorPtr&                          getPrintfTensor() const { return m_printfTensor; }
    unsigned                                  getPrintfPosition(unsigned int descTensorCount) const;
    void registerKernelToCodeGen(const std::unique_ptr<CodeGenerator>& codeGen, const deviceAddrOffset& addr) const;
    NodePtr                                   clone() const override;
    NodePtr                                   getSlice() const override;
    bool isLoweringKernel() const { return getGUID().find("lowering_pack_2_w77_s22") != std::string::npos; }
    bool                                      isCast() const override;
    TensorSemanticType                        getParamSemanticType(const TensorPtr& param) const override;
    bool                                      validateNodeLayout() const override;
    bool                                      validateNodeForGraph(const HabanaGraph& g) const override;
    virtual tpc_lib_api::GlueCodeReturn       init(tpc_lib_api::DeviceId   deviceId,
                                                   AuxiliaryTensors*       cachedAuxiliaryTensors,
                                                   std::optional<uint32_t> kernelUniqueId);
    bool                                      isBroadcastableOperation() const override;
    virtual bool                              isSuggestedOptimizationDone() const;
    void setSuggestedOptimizationDone(bool isOptimizationDone) { m_optimized = isOptimizationDone; }
    bool isInstantiated() const { return m_instanceWrapper.isInstantiated(); }
    void                                      upgradeNodePrecisionIfMissingKernel(bool forceUpgrade = false);
    void                                      setNodePrecision(synDataType precision) override;
    HabanaDeviceType                          getNodeDeviceType() const override;
    const tpc_lib_api::HabanaKernelParams&    getSucceededGlueParams() const;
    bool                                      hasMandatorySplitDim() const;
    unsigned                                  getMandatorySplitDim() const;

    void updateCache() override;

    struct CostModelResult
    {
        CostModelResult(const TpcElfTools::CostModelResult& result)
        : tpcCyclesFinalDecision(result.finalDecision), asicCycles(result.asicCycles), asicTimeInUsec(result.asicTime)
        {
        }
        uint64_t tpcCyclesFinalDecision;
        uint64_t asicCycles;
        double   asicTimeInUsec;
    };

    std::optional<CostModelResult> getCostModelResult() const;

    uint64_t getShapeInferenceFunctionVersion() const override;
    bool     isSmallVLMRequired() const;
    bool     isPrintfUsed() const;
    bool     is44bitMode() const;
    bool     isAccessingSharedMem() const;
    uint16_t getRmwOutputMask(tpc_lib_api::DeviceId deviceId) const;
    bool     isOutputTensorRmw(unsigned tensorOutputIndex, tpc_lib_api::DeviceId deviceId) const;
    bool     isOutputTensorMemset(unsigned tensorOutputIndex, tpc_lib_api::DeviceId deviceId) const;
    bool     isOutputTensorAllRequired(unsigned tensorOutputIndex) const;
    bool     isOutputTensorFullyWritten(unsigned tensorOutputIndex) const;
    bool     isOutputTensorPartialWrites(unsigned tensorOutputIndex) const;
    bool     isFusedKernel() const;
    // static functions
    static std::map<unsigned, unsigned> getDuplicateTensorsFromElf(void*& elf, unsigned int elfSize);
    static unsigned                     getLlvmTensorIdFromMap(unsigned tensorIdx, uint64_t dupTensorsMap);

    // Asks the KernelDB for a suggestion of a tensor manipulation that would improve performance
    virtual tpc_lib_api::GlueCodeReturn getSuggestedTensorManipulation(tpc_lib_api::TensorManipulationSuggestion* s);

    bool hasTransposeOptimization(tpc_lib_api::DeviceId deviceId) const;
    void setDoubleStore(void* kernelElf, uint32_t elfSize, unsigned tensorIdx, bool isInput);

    static constexpr unsigned         BUFFER_SIZE = 16384U;  // 16kb
    static constexpr std::string_view RUN_ON_TPC  = "_runOnTpc";

    NodeROI generateRoi() const override;
    Settable<NodeROI> getInputROI(const NodeROI& roi, uint32_t tensorIdx)        const override;
    Settable<NodeROI> getOutputROI(const NodeROI& roi, uint32_t tensorIdx)       const override;

    bool runShapeInferenceFunction(synDeviceType deviceType,
                                   SifParams*    params,
                                   SifOutputs*   outputs,
                                   bool          inferMax,
                                   bool          skipStatic) override;

    std::vector<Node::NodeDynamicShapeProjection> getDynamicShapeProjectionsTensors() const override;

    synDataType getRequiredInputType(uint32_t tensorIdx)                         const override;
    synDataType getRequiredOutputType(uint32_t tensorIdx)                        const override;

    std::string getNodeTypeStr()                                                 const override;

    void accept(NodeVisitor* visitor) override;

    std::map<TensorPtr, TensorVector, TensorComparator> getReusableInputs()       const override;
    std::map<TensorPtr, TensorVector, TensorComparator> getReusableInputBinding() const override;

    // Return vector of output tensors that should be initialized before using
    virtual TensorVector                    getMemsetBeforeExecTensors()                 const;

    // Heuristic check if the kernel behaves like element-wise, and can be sliced to any size
    virtual bool isSeparable(tpc_lib_api::DeviceId deviceId) const;
    virtual bool isSeparable(tpc_lib_api::DeviceId deviceId, unsigned dimension) const;

    virtual std::vector<bool> getInputsScalarPipeStatus(tpc_lib_api::DeviceId deviceId) const;

    virtual uint64_t getScalarPipeInputsSize(tpc_lib_api::DeviceId deviceId) const;

    bool getInfoInstance(KernelInstantiationWrapper& out,
                         tpc_lib_api::DeviceId       deviceId,
                         bool                        extractElf,
                         bool                        setReducible = false) const;

    virtual void resetInstantiated();

    bool isSeparableException() const;

    bool isRestrictedShapeRandomNode() const;

    void setAllowedForStitching(bool allowedForStitching);
    bool isAllowedForStitching(const HabanaGraph& graph) const;
    bool isNode64BitCompatible() const override;

    static unsigned numTensorDescriptors(const Tensor& tensor)
    {
        return div_round_up(tensor.getDim(), SYN_MAX_TENSOR_DIM);
    }

    static unsigned numTensorGlueCodeAccessPatternEntries(const TensorPtr& tensor);
    unsigned getTotalNumDescriptors();

    static constexpr std::string_view NOP_KERNEL_NAME = "nop";

    static unsigned getMaxAvailableTpc(tpc_lib_api::DeviceId deviceId);
    static unsigned getMaxAvailableTpc(const HalReader* reader);

    bool requiresOutputMaxDimInfer() const override;
    void permuteParams(const PermutationVector& inputPermutations) override;

    bool isTranspose() const override;

    virtual bool canHandleStridedInput(synDeviceType device = synDeviceTypeInvalid) const override;

protected:
    TPCNode(const TPCNode& other);

    void                            initPrintfTensor();

    // Instantiate the TPC kernel and return it without initializing the node with the instance
    tpc_lib_api::GlueCodeReturn instantiate(KernelInstantiationWrapper& instance) const
    {
        return instance.instantiate(getGUIDAndHash());
    }

    bool                            validateNodeWithoutPrecision() const;
    TensorPtr                       getAuxTensor(const tpc_lib_api::AuxTensor* auxiliaryTensor,
                                                 const std::shared_ptr<char>&  data,
                                                 AuxiliaryTensors*             cachedAuxiliaryTensors);
    static bool                     isGuidBlocked(const std::string& guid);

    // Need to be overrideable for TPCSlice
    virtual NodeROI getWorkROI(const tpc_lib_api::TensorAccessPattern& accessPattern,
                               const NodeROI&                          roi,
                               const TensorPtr&                        tensor,
                               unsigned                                startDim) const;
    static NodeROI  generateFullWorkROI(const TensorPtr& tensorPtr);
    static NodeROI  generateInitialWorkROIForTensor(const TensorPtr& tensorPtr, const NodeROI& baseROI);
    virtual TOffset getSliceOffset(const TensorPtr&                         tensor,
                                   unsigned                                 dim,
                                   const tpc_lib_api::DimIndexSpaceMapping& dimAccessPattern) const;
    bool            hasNodeIOManagerSpecialization() const override { return true; }

    gc::access_pattern::NodeAccessPatternPtr generateNodeAccessPattern() const override;
    bool     isSpecialFunctionsUsed() const;

    KernelInstantiationWrapper m_instanceWrapper;

    bool                                         m_optimized;
    kernelID                                     m_uniqueID;
    TensorPtr                                    m_printfTensor;
    unsigned                                     m_printfPosition;
    static const std::set<std::string_view>      m_seperableBlockList;
    static const std::set<std::string_view>      m_lfsrRandomGuidList;
    static const std::set<std::string_view>      m_stridedBlockList;

private:
    std::map<TensorPtr, TensorVector, TensorComparator>
    getReusableInputs(const KernelInstantiationWrapper& instanceWrapper, bool isSuggestedBinding) const;
    std::map<TensorPtr, TensorVector, TensorComparator> getReusableInputs(bool isSuggestedBinding) const;

    void     createAuxTensors(AuxiliaryTensors* cachedAuxiliaryTensors);
    void     addDoubleStoreAccessPattern(unsigned tensorIdx, bool isInput);
    void     setKernelElf(void* kernelElf, uint32_t elfSize);
    void     isAccessPatternExceedsTensor(const TensorPtr&                        inputTensor,
                                          unsigned                                baseDim,
                                          const uint64_t*                         indexSpaceGeometry,
                                          const tpc_lib_api::TensorAccessPattern& accessPattern) const;
    bool     shouldSkipAccessPatternValidation() const;

    void validateAccessPattern() const;     // Verifies access pattern is not totally out of the inputs tensor bounds
    void validatePreferredSplitDim() const; // Verifies preferredSplitDim is also allRequired

    void validateTensorsIndexSpace(const TensorVector&                     tensors,
                                   const tpc_lib_api::TensorAccessPattern* accessPattern) const;

    bool
    isSeparableOnAllOrSingleDim(tpc_lib_api::DeviceId deviceId, bool checkAllDimensions = true, unsigned dim = 0) const;

    void generateDynamicShapeProjectionsTensors();

    bool isTensorCoveringIndexSpaceDim(unsigned                                indexSpaceDim,
                                       TSize*                                  tensorSize,
                                       unsigned int                            dimNum,
                                       const uint64_t*                         geometry,
                                       const tpc_lib_api::TensorAccessPattern& accessPattern,
                                       unsigned&                               tensorDim,
                                       unsigned&                               minMissingElements) const;

    bool findProjectionBetterCoveringIndexSpaceDimension(unsigned                          indexSpaceDim,
                                                         TensorVector                      tensors,
                                                         bool                              isOutput,
                                                         unsigned&                         minMissingElements,
                                                         unsigned&                         maxRank,
                                                         bool&                             foundMappedDim,
                                                         Node::NodeDynamicShapeProjection& projection) const;

    uint64_t                      m_shapeInferenceFunctionVersion;
    std::vector<Node::NodeDynamicShapeProjection> m_dynamicShapeProjectionTensors;

    std::string getUpgradedGUID(const std::string& guid);
    void   setSparseAccessTensorsAnnotation();
    void        setTensorsPrefetchStride(const HalReader& reader);

    thread_local static unsigned s_printfAllocatedSize;

    mutable std::optional<bool> m_isAllowedForStitching;
    std::optional<deviceAddrOffset> m_kernelOffsetInSection;
};
