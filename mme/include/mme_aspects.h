#ifndef MME__ASPECTS_H
#define MME__ASPECTS_H

#include "llvm/small_vector.h"
#include "mme_common/mme_common_enum.h"

namespace MmeCommon
{
// An aspect is a list of dimensions that should be treated as a single one, since they are traeted like that in HW.
// Some examples: Convolutions B,D,H,W and BGemms batch dimensions.
class Aspect
{
public:
    static constexpr size_t EXP_MAX_DIMS_PER_ASPECT = 4;
    using DimVector = llvm_vecsmall::SmallVector<size_t, EXP_MAX_DIMS_PER_ASPECT>;

    const DimVector& dims() const { return m_dims; }

    template<typename Container>
    void pushDims(Container& cont) const
    {
        std::copy(m_dims.begin(), m_dims.end(), std::back_inserter(cont));
    }

    // Some standard container-like methods
    DimVector::size_type size() const { return m_dims.size(); }
    DimVector::const_iterator begin() const { return m_dims.begin(); }
    DimVector::const_iterator end() const { return m_dims.end(); }
    bool empty() const { return m_dims.empty(); }

protected:
    Aspect(const DimVector& dims) : m_dims(dims) {}

    const DimVector m_dims;
};

// An index space aspect is an aspect that lists index space dimensions. It is defined to allow compile time distinction
// between it and aspects listing tensor dimensions.
class IndexSpaceAspect : public Aspect
{
private:
    friend struct AspectFactory;
    IndexSpaceAspect(const DimVector& dims) : Aspect(dims) {}
};

// An operand aspect is and aspect that lists tensor dimensions. It is defined to allow compile time distinction between
// it and aspects listing index space dimensions.
class OperandAspect : public Aspect
{
private:
    friend struct AspectFactory;
    OperandAspect(const DimVector& dims) : Aspect(dims) {}
};

// Base for all aspect names enumerations
using AspectName = uint8_t;

// Base for all aspect factories
struct AspectFactory
{
    virtual ~AspectFactory() = default;
    virtual IndexSpaceAspect create(AspectName name) const = 0;
    virtual OperandAspect create(AspectName name, EMmeOperand operand) const = 0;

protected:
    static IndexSpaceAspect createIndexSpaceAspect(const Aspect::DimVector& dims) { return {dims}; }
    static OperandAspect createOperandAspect(const Aspect::DimVector& dims) { return {dims}; }
};
using AspectFactoryPtr = std::unique_ptr<AspectFactory>;

// Physical aspects represent the dimensions that compose the MME physical engine traversal and tiling directions
namespace PhysicalAspects
{
enum Name : AspectName
{
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    INPUTS_COMMON,
    GROUPS,  // The dimensions on which several problems are merged into a single operation (batch-gemm, group-conv...)
};

struct Factory : public AspectFactory
{
    Factory(const MmeLayerParams* params);
    ~Factory() override;

    IndexSpaceAspect create(AspectName name) const override;
    OperandAspect create(AspectName name, EMmeOperand operand) const override;

private:
    // Using pImpl for a more stable API
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace PhysicalAspects

// Convolution aspects are aspects that are used to describe the dimensions of a convolution operation
namespace ConvolutionAspects
{
enum Name : AspectName
{
    IN_CHANNELS,
    OUT_CHANNELS,
    SPATIAL,
    FILTER,
};

struct Factory : public AspectFactory
{
    Factory(const MmeLayerParams* params);
    ~Factory() override;

    IndexSpaceAspect create(AspectName name) const override;
    OperandAspect create(AspectName name, EMmeOperand operand) const override;

private:
    // Using pImpl for a more stable API
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
}  // namespace ConvolutionAspects
}  // namespace MmeCommon

#endif //MME__ASPECTS_H
