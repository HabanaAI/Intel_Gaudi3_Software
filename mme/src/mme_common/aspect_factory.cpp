#include "mme_aspects.h"

#include "mme_access_pattern.h"
#include "index_space_dimensions.h"

namespace MmeCommon
{
using Dims = Aspect::DimVector;

class AspectFactoryImplBase
{
public:
    AspectFactoryImplBase(const MmeLayerParams* params) : m_accessPattern(AccessPatternFactory::createFrom(params)) {}

    virtual Dims getIndexSpaceDims(AspectName aspectName) const = 0;

    virtual Dims getOperandDims(AspectName aspectName, EMmeOperand operand) const
    {
        Dims dims;
        OperandRole role = operandRole(operand);
        const auto& operandAP = m_accessPattern.operandAccessPatterns.at(role);
        for (auto aspectIdxSpcDim : getIndexSpaceDims(aspectName))
        {
            for (size_t dim = 0; dim < operandAP.dimsAccessPattern.size(); dim++)
            {
                if (operandAP.dimsAccessPattern.at(dim).indexSpaceDim == aspectIdxSpcDim)
                {
                    dims.push_back(dim);
                }
            }
        }
        return dims;
    }

private:
    const AccessPattern m_accessPattern;

    static OperandRole operandRole(EMmeOperand operand)
    {
        OperandRole role {};
        switch (operand)
        {
            case EMmeOperand::e_mme_op_x:
                role = OperandRole::X;
                break;
            case EMmeOperand::e_mme_op_w:
                role = OperandRole::W;
                break;
            case EMmeOperand::e_mme_op_y:
                role = OperandRole::Y;
                break;
            case EMmeOperand::e_mme_op_o:
                role = OperandRole::OUTPUT_COPY;
                break;
        }
        return role;
    }
};

namespace PhysicalAspects
{
using Dims = Aspect::DimVector;
using PhysAspect = PhysicalAspects::Name;

namespace Implementation
{
struct AspectGeneratorBase
{
    virtual Dims getIndexSpaceDims(AspectName aspectName) const = 0;
};

namespace Gemm
{
class AspectGenerator : public AspectGeneratorBase
{
public:
    AspectGenerator(const MmeLayerParams* params) : m_params(params) {}

    Dims getIndexSpaceDims(AspectName aspectName) const override
    {
        using ISDim = AccessPatternDetails::Gemm::IndexSpaceDim;
        Dims dims;
        switch (PhysAspect(aspectName))
        {
            case PhysAspect::OUTPUT_WIDTH:
                dims.push_back(ISDim::DIM_OUT_FCD);
                break;
            case PhysAspect::INPUTS_COMMON:
                dims.push_back(ISDim::DIM_OPERANDS_COMMON);
                break;
            case PhysAspect::OUTPUT_HEIGHT:
                dims.push_back(ISDim::DIM_OUT_HEIGHT);
                if (m_params->canFlatten())
                {
                    dims.push_back(ISDim::DIM_BATCH_0);
                }
                break;
            case PhysAspect::GROUPS:
                if (!m_params->canFlatten())
                {
                    dims.push_back(ISDim::DIM_BATCH_0);
                }
                dims.insert(dims.end(), {ISDim::DIM_BATCH_1, ISDim::DIM_BATCH_2});
                break;
        }
        return dims;
    }

private:
    const MmeLayerParams* m_params;
};
}  // namespace Gemm
namespace Conv
{
class AspectGenerator : public AspectGeneratorBase
{
public:
    AspectGenerator(const MmeLayerParams* params) : m_params(params) {}

    Dims getIndexSpaceDims(AspectName aspectName) const override
    {
        Dims dims;
        switch (PhysAspect(aspectName))
        {
            case PhysAspect::OUTPUT_WIDTH:
                if (m_params->isDedxOperation())
                {
                    pushInChannels(dims);
                }
                else
                {
                    // Fwd or dedw
                    pushOutChannels(dims);
                }
                break;
            case PhysAspect::INPUTS_COMMON:
                if (m_params->isDedwOperation())
                {
                    pushSpatialDims(dims);
                }
                else
                {
                    if (m_params->isDedxOperation())
                    {
                        // dedx
                        pushOutChannels(dims);
                    }
                    else
                    {
                        // fwd
                        pushInChannels(dims);
                    }
                }
                break;
            case PhysAspect::OUTPUT_HEIGHT:
                if (m_params->isDedwOperation())
                {
                    pushInChannels(dims);
                }
                else
                {
                    pushSpatialDims(dims);
                }
            case PhysAspect::GROUPS:
                break;
        }
        return dims;
    }

private:
    using ISDim = AccessPatternDetails::Conv::IndexSpaceDim;

    const MmeLayerParams* m_params;

    static void pushInChannels(Dims& vec)
    {
        vec.push_back(ISDim::DIM_IN_CHANNELS);  // C
    }
    static void pushOutChannels(Dims& vec)
    {
        vec.push_back(ISDim::DIM_OUT_CHANNELS);  // K
    }
    static void pushSpatialDims(Dims& vec)  // BDHW
    {
        vec.insert(vec.end(), {ISDim::DIM_WIDTH, ISDim::DIM_HEIGHT, ISDim::DIM_DEPTH, ISDim::DIM_BATCH});
    }
};
}  // namespace Conv
}  // namespace Implementation

class Factory::Impl : public AspectFactoryImplBase
{
public:
    Impl(const MmeLayerParams* params) : AspectFactoryImplBase(params)
    {
        if (params->isGemmOperation())
        {
            m_aspectGenerator = std::make_unique<Implementation::Gemm::AspectGenerator>(params);
        }
        if (params->isConvOperation())
        {
            m_aspectGenerator = std::make_unique<Implementation::Conv::AspectGenerator>(params);
        }
    }

    Dims getIndexSpaceDims(AspectName aspectName) const override { return generator()->getIndexSpaceDims(aspectName); }

private:
    std::unique_ptr<Implementation::AspectGeneratorBase> m_aspectGenerator;
    const Implementation::AspectGeneratorBase* generator() const
    {
        MME_ASSERT(m_aspectGenerator != nullptr, "Aspect generator not initialized");
        return m_aspectGenerator.get();
    }
};

Factory::Factory(const MmeLayerParams* params) : m_impl(new Factory::Impl(params)) {}
Factory::~Factory() = default;

IndexSpaceAspect Factory::create(AspectName name) const
{
    return createIndexSpaceAspect(m_impl->getIndexSpaceDims(name));
}

OperandAspect Factory::create(AspectName name, EMmeOperand operand) const
{
    return createOperandAspect(m_impl->getOperandDims(name, operand));
}
}  // namespace PhysicalAspects

namespace ConvolutionAspects
{
using ConvAspect = ConvolutionAspects::Name;

class Factory::Impl : public AspectFactoryImplBase
{
public:
    Impl(const MmeLayerParams* params) : AspectFactoryImplBase(params)
    {
        MME_ASSERT(params->isConvOperation(),
                   "Convolution aspect factory can only be instantiated for convolution operations");
    }
    Dims getIndexSpaceDims(AspectName aspectName) const override
    {
        using ISDim = AccessPatternDetails::Conv::IndexSpaceDim;

        Dims dims;
        switch (ConvAspect(aspectName))
        {
            case Name::IN_CHANNELS:
                dims.push_back(ISDim::DIM_IN_CHANNELS);
                break;
            case Name::OUT_CHANNELS:
                dims.push_back(ISDim::DIM_OUT_CHANNELS);
                break;
            case Name::SPATIAL:
                dims.insert(dims.begin(), {ISDim::DIM_WIDTH, ISDim::DIM_HEIGHT, ISDim::DIM_DEPTH, ISDim::DIM_BATCH});
                break;
            case Name::FILTER:
                dims.insert(dims.begin(), {ISDim::DIM_FILTER_S, ISDim::DIM_FILTER_R, ISDim::DIM_FILTER_Q});
                break;
        }
        return dims;
    }
};

Factory::Factory(const MmeLayerParams* params) : m_impl(new Factory::Impl(params)) {}
Factory::~Factory() = default;

IndexSpaceAspect Factory::create(AspectName name) const
{
    return createIndexSpaceAspect(m_impl->getIndexSpaceDims(name));
}

OperandAspect Factory::create(AspectName name, EMmeOperand operand) const
{
    return createOperandAspect(m_impl->getOperandDims(name, operand));
}

}  // namespace ConvolutionAspects
}  // namespace MmeCommon