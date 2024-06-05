#include "code_generator_factory.h"

using namespace CodeGeneration;

CodeGeneratorPtr CodeGeneratorFactory::createCodeGenerator(synDeviceType deviceType, HabanaGraph* graph)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return instantiateGaudiCodeGenerator(graph);

        case synDeviceGaudi2:
            return instantiateGaudi2CodeGenerator(graph);

        case synDeviceGaudi3:
            return instantiateGaudi3CodeGenerator(graph);

        default:
            return instantiateSimCodeGenerator(graph);
    }
}