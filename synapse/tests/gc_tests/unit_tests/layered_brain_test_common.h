#pragma once

#include "compilation_hal_reader.h"
#include "graph_editor.h"
#include "layered_brain.h"
#include "synapse_common_types.h"
#include "graph_optimizer_test.h"
#include "brain_data.h"
#include "utils.h"
#include "types.h"

template<class GraphType>
class LayeredBrainCommonTest : public GraphOptimizerTest
{
public:
    LayeredBrainCommonTest() : m_graph(), m_halSetter(&m_graph), m_ctx(m_graph)
    {
        m_graph.setLayeredBrainData(std::make_unique<gc::layered_brain::LayeredBrainData>());
    }

protected:
    class BPGraphContext
    {
    public:
        explicit BPGraphContext(HabanaGraph& graph) : m_graph(graph) { m_graph.constructBPGraph(); }
        virtual ~BPGraphContext() { m_graph.discardBPGraph(); }
        HabanaGraph& m_graph;
    };

    template<typename SizesContainer>
    TensorPtr createTensor(SizesContainer shape, synDataType dtype = syn_type_single, bool persistent = true) const
    {
        auto                t = std::make_shared<Tensor>(shape.size(), shape.data(), dtype);
        synMemoryDescriptor memDesc(persistent);
        t->setMemoryDescriptor(memDesc);
        if (persistent)
        {
            t->setMemorySectionID(m_memorySectionId++);
        }
        t->map();
        return t;
    }

    template<typename SizesContainer>
    TensorPtr createTensor(SizesContainer shape,
                           SizesContainer minShape,
                           synDataType    dtype      = syn_type_single,
                           bool           persistent = true) const
    {
        HB_ASSERT(shape.size() == minShape.size(), "");
        TensorPtr t = createTensor(shape, dtype, persistent);
        t->setMinSize(minShape.data());
        return t;
    }

    template<typename SizesContainer>
    TensorPtr createShapeTensor(SizesContainer shape,
                                SizesContainer minShape,
                                synDataType    dtype      = syn_type_single,
                                bool           persistent = true) const
    {
        return std::make_shared<Tensor>(shape.size(),
                                        shape.data(),
                                        dtype,
                                        nullptr,
                                        nullptr,
                                        false,
                                        false,
                                        INVALID_BATCH_POS,
                                        minShape.data(),
                                        SHAPE_TENSOR);
    }

    GraphType                  m_graph;
    CompilationHalReaderSetter m_halSetter;
    BPGraphContext             m_ctx;
    mutable unsigned           m_memorySectionId = MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + 1;
};