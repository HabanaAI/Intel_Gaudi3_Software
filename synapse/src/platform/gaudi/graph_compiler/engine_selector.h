#pragma once

#include "graph_compiler/engine_selector.h"

namespace gaudi
{

class EngineSelector : public gc::EngineSelector
{
private:
    virtual void visit(TransposeNode* node) override;
};

} /* namespace gaudi */
