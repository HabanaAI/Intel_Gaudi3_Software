#include "hal_reader/hal_reader.h"

#include <stdexcept>

bool HalReader::isSupportedDataType(synDataType type) const
{
    return type != syn_type_na && ((getSupportedTypes() & type) == type);
}

bool HalReader::isSupportedMmeDataType(synDataType type) const
{
    return type != syn_type_na && ((getSupportedMmeTypes() & type) == type);
}

/*
 * default implementation match training (gaudi) logic.
 * inference HalReader should override
 */
bool HalReader::isSupportedMmeInputDataType(synDataType type) const
{
    return isSupportedMmeDataType(type);
}

unsigned HalReader::getNumMmeEnginesWithSlaves() const
{
    return getNumMmeEngines() * 2;
}