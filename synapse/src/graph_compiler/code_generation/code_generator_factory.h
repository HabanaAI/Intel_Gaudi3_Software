#pragma once
#include "code_generator.h"
#include "compiler_types.h"

class CodeGeneratorFactory
{
public:
    static CodeGeneratorPtr createCodeGenerator(synDeviceType deviceType, HabanaGraph* graph);

private:
    static CodeGeneratorPtr instantiateGaudiCodeGenerator(HabanaGraph* graph);
    static CodeGeneratorPtr instantiateGaudi2CodeGenerator(HabanaGraph* graph);
    static CodeGeneratorPtr instantiateGaudi3CodeGenerator(HabanaGraph* graph);
    static CodeGeneratorPtr instantiateSimCodeGenerator(HabanaGraph* graph);
};
