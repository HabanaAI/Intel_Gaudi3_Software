#pragma once

#include "graph_compiler/habana_graph.h"
#include "hal_reader/hal_reader.h"

class CompilationHalReader
{
    friend class CompilationHalReaderSetter;

public:
    static const std::shared_ptr<HalReader>& getHalReader(bool allowFailure = false);
    static void                              setHalReader(const std::shared_ptr<HalReader>& halReader);
    static bool                              isHalReaderSet();

private:
    static std::shared_ptr<HalReader>& instance();
    static void                        unsetHalReader();
};

class CompilationHalReaderSetter
{
public:
    CompilationHalReaderSetter(const HabanaGraph* g);
    ~CompilationHalReaderSetter();
};
