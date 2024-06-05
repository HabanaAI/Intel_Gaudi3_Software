#pragma once

#include "types.h"
#include <optional>

/**
 * Control which dimensions in the tensor represent each section
 * in the matrix multiplication
 */
class MmeDimController
{
public:
    explicit MmeDimController(const NodePtr& mmeNode);
    const DimVector& getNonCommonAxis(unsigned operandIndex) const;
    const DimVector& commonDimOperandA() const;
    const DimVector& nonCommonDimOperandA() const;

    const DimVector& commonDimOperandB() const;
    const DimVector& nonCommonDimOperandB() const;

    const DimVector& nonCommonDimsForOperand(unsigned inputIndex) const;
    const DimVector& commonDimsForOperand(unsigned inputIndex) const;

    const DimVector& widthOutput() const;
    const DimVector& heightOutput() const;

    const DimVector& qrsSizes() const;
    const DimVector& batchDim() const;

    bool isCommonDimOperandA(uint32_t dim) const;
    bool isCommonDimOperandB(uint32_t dim) const;

    bool isCommonDimForOperand(uint32_t dim, unsigned inputIndex) const;

private:
    void initGemmDims(const NodePtr& mmeNode);

    DimVector m_commonDimOperandA;
    DimVector m_nonCommonDimOperandA;

    DimVector m_commonDimOperandB;
    DimVector m_nonCommonDimOperandB;

    DimVector m_widthOutput;
    DimVector m_heightOutput;

    DimVector m_QRS;
    DimVector m_batchDim;
};
