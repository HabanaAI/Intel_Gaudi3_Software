#pragma once

#include "types.h"
#include "gaudi3_graph.h"
namespace gaudi3
{
class TensorInHbmSetter
{
public:
    explicit TensorInHbmSetter(Gaudi3Graph* g);

    virtual ~TensorInHbmSetter() = default;

    bool setInHbm();

protected:
    Gaudi3Graph* m_graph;
};
}  // namespace gaudi3
