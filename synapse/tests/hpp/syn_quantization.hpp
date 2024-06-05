#pragma once

#include "syn_object.hpp"

namespace syn
{
template<class T>
class Quantization
{
public:
    Quantization(synQuantizationProperty property, T quantizationData)
    : m_property(property), m_quantizationData(quantizationData)
    {
    }

    T&                      get() { return m_quantizationData; }
    uint64_t                size() const { return sizeof(m_quantizationData); }
    synQuantizationProperty getProperty() const { return m_property; }

private:
    synQuantizationProperty m_property;
    T                       m_quantizationData;
};
}  // namespace syn