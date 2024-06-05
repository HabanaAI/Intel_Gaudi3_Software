#pragma once

#include "synapse_common_types.h"

class QmanDefinitionInterface;

namespace generic
{
class CommandBufferPktGenerator;
}

QmanDefinitionInterface* getDeviceQmansDefinition(synDeviceType devType);

generic::CommandBufferPktGenerator* getDeviceCommandBufferPktGenerator(synDeviceType devType);
