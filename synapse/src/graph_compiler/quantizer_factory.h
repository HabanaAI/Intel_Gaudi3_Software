#pragma once

#include "types.h"
#include "quantizer.h"

class QuantizerFactory
{
public:
    static QuantizerFactory&   getInstance();
    static const QuantizerPtr& getDefaultNodeQuantizer();
    static const QuantizerPtr& getNodeQuantizer(const StringViewWithHash& guidWithoutDType);
    virtual ~QuantizerFactory() {}

private:
    QuantizerFactory();
    QuantizerFactory(QuantizerFactory const&)   = delete;
    void operator=(QuantizerFactory const&)     = delete;
    QuantizersMap            m_quantizersMap;
    QuantizerPtr             m_defaultQunatizer;
};
