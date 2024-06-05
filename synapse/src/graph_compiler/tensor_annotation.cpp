#include "tensor_annotation.h"

#include "compilation_hal_reader.h"
#include "hal_reader/hal_reader.h"

TensorMemoryAlignment::TensorMemoryAlignment()
{
    const auto& halReader = CompilationHalReader::getHalReader(true);
    if (halReader != nullptr)
    {
        alignment = halReader->getCacheLineSizeInBytes();
    }
}
