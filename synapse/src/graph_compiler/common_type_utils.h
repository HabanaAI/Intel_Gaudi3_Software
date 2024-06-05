#pragma once

#include "synapse_common_types.h"

std::string   tensorTypeToString(const synTensorType tensorType);
synTensorType tensorTypeFromString(const std::string& str);