#pragma once
#include <memory>
#include <vector>
#include "tpc_kernel_lib_interface.h"
#include "tpc_kernel_lib_interface_private.h"
#include "tpc_elf_api.hpp"
#include "types.h"
#include <string>

class TPCNode;

template<typename T>
class TensorOperandsVector
{
public:
    void clear()
    {
        m_numInputs = 0;
        m_data.clear();
    }
    void resize(size_t numInputs, size_t numOutputs)
    {
        m_numInputs = numInputs;
        m_data.resize(numInputs + numOutputs);
    }
    T* getInputs() { return m_data.data(); }
    T* getOutputs() { return m_data.data() + m_numInputs; }

private:
    size_t                        m_numInputs = 0;
    SmallVector<T, MAX_TENSOR_NR> m_data;
};

using TPCLibTensorOperandsVector              = TensorOperandsVector<tpc_lib_api::Tensor>;
using TPCLibTensorOperandsAccessPatternVector = TensorOperandsVector<tpc_lib_api::TensorAccessPattern>;
using TPCLibTensorOperandsDataLayoutVector    = TensorOperandsVector<tpc_lib_api::TensorDataLayout>;
using TPCLibTensorOperandsSuggestionsVector   = TensorOperandsVector<tpc_lib_api::TensorOperation>;
using TPCLibTensorDataLayoutVector            = SmallVector<tpc_lib_api::TensorDataLayout, MAX_TENSOR_NR>;

class KernelInstantiationWrapper
{
public:
    KernelInstantiationWrapper(const KernelInstantiationWrapper& other);
    KernelInstantiationWrapper();
    ~KernelInstantiationWrapper() {};

    // getters:
    const std::shared_ptr<char>&                  getAuxTensor(size_t i);
    void                                          deleteAuxTensorBuffer(size_t i);
    const tpc_lib_api::HabanaKernelInstantiation& getInstance() const { return m_instance; };
    tpc_lib_api::HabanaKernelInstantiation&       getInstance() { return m_instance; }
    const std::shared_ptr<char>&                  getCacheKernel() const { return m_kernelBinary; }
    void*                                         getKernelBinary() const { return m_kernelBinary.get(); }
    unsigned                                      getKernelSize() const { return m_kernelSize; }
    bool                                          isValidElfProgramHeader() const { return m_validTpcHeader; }
    bool                                          isInstantiated() const { return m_instantiated; }
    const TpcElfTools::TPCProgramHeader&          getElfProgramHeader() const { return m_programHeader; }
    tpc_lib_api::HabanaKernelParams&              getGlueParams() { return m_glueParams; }
    const tpc_lib_api::HabanaKernelParams&        getGlueParams() const { return m_glueParams; }

    void        updateGlueCodeParamsAndTensorAccessPatternPointers(const TPCNode& tpcNode);
    static void updateGlueCodeParamsAndLayoutsPointers(const TPCNode&                        tpcNode,
                                                       tpc_lib_api::HabanaKernelParams&      glueParams,
                                                       TPCLibTensorOperandsVector&           tensorOperands,
                                                       tpc_lib_api::NodeDataLayouts&         nodeLayouts,
                                                       TPCLibTensorOperandsDataLayoutVector& operandTensorLayouts,
                                                       TPCLibTensorDataLayoutVector&         shapeTensorLayouts);

    void        initParams(const TPCNode& tpcNode, tpc_lib_api::DeviceId deviceId, bool setReducible = false);
    static void initParams(tpc_lib_api::HabanaKernelParams& outParams,
                           const TPCNode&                   tpcNode,
                           tpc_lib_api::DeviceId            deviceId,
                           bool                             setReducible = false);
    void        resetParams() { m_glueParamsInitialized = false; }

    tpc_lib_api::GlueCodeReturn instantiate(const StringWithHash& guidAndHash);
    // setters:
    void setInstantiated(bool val) { m_instantiated = val; }
    void setKernelElf(void* kernelElf, uint32_t elfSize);
    void setProgrammingHeader(TpcElfTools::TPCProgramHeader& programHeader)
    {
        m_programHeader  = programHeader;
        m_validTpcHeader = true;
    }  // for unit test

    // We need to cache the elf in case there's multiple TPC nodes use the same unique ID
    // And the first one is removed from the graph (we don't want the binary will be lost)
    bool shouldCacheElf() const
    {
        return m_kernelElf.get() == m_instance.kernel.kernelElf && m_kernelBinary == nullptr;
    }
    const std::shared_ptr<char> getElfBuffer() const { return m_kernelElf; }
    TpcElfTools::TpcElfStatus   extractElf(const std::string& name);

private:
    void setKernelBinary(void* kernelBinary, uint32_t binarySize);
    bool checkAndIncreaseAuxBuffer();
    void deleteUnneededAuxBuffers();
    void validateAuxTensorsSize();
    void prepareElfBuffer(uint64_t bufferSize);
    void resetKernelInstantiationParams();

    struct TensorOperandCounts
    {
        uint8_t inputTensorCount  = 0;
        uint8_t outputTensorCount = 0;
        uint8_t shapeTensorCount  = 0;
    };

    static TensorOperandCounts updateGlueCodeParamsTensorPointers(const TPCNode&                   tpcNode,
                                                                  tpc_lib_api::HabanaKernelParams& glueParams,
                                                                  TPCLibTensorOperandsVector&      tensorOperands);

    struct BufferDescriptor
    {
        BufferDescriptor() = default;
        BufferDescriptor(size_t bufferSize, std::shared_ptr<char>&& bufferData)
        : size(bufferSize), buffer(std::move(bufferData))
        {
        }
        size_t                size;
        std::shared_ptr<char> buffer;
    };

    using AuxBufferVector      = SmallVector<BufferDescriptor, 1>;
    using TPCLibAuxTensorArray = std::array<tpc_lib_api::AuxTensor, MAX_TENSOR_NR>;
    std::shared_ptr<char>                   m_kernelBinary;
    std::shared_ptr<char>                   m_kernelElf;
    unsigned                                m_kernelSize            = 0;
    bool                                    m_instantiated          = false;
    bool                                    m_validTpcHeader        = false;
    bool                                    m_glueParamsInitialized = false;
    tpc_lib_api::HabanaKernelInstantiation  m_instance              = {};
    tpc_lib_api::HabanaKernelParams         m_glueParams            = {};
    TpcElfTools::TPCProgramHeader           m_programHeader         = {};
    TPCLibAuxTensorArray                    m_auxTensors            = {};
    AuxBufferVector                         m_auxBuffers;
    TPCLibTensorOperandsVector              m_tensorOperands;
    TPCLibTensorOperandsAccessPatternVector m_tensorOperandsAccessPattern;
};
