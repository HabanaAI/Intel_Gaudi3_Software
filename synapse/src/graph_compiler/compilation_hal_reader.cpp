#include "compilation_hal_reader.h"
#include "infra/defs.h"
#include "log_manager.h"

std::shared_ptr<HalReader>& CompilationHalReader::instance()
{
    thread_local static std::shared_ptr<HalReader> instance;
    return instance;
}

// should only by used by tests that don't have graph
void CompilationHalReader::setHalReader(const std::shared_ptr<HalReader>& halReader)
{
    std::shared_ptr<HalReader>& currentHalReader = instance();
    if (currentHalReader != nullptr)
    {
        LOG_WARN(GC, "CompilationHalReader is already set!");
    }
    currentHalReader = halReader;
}

void CompilationHalReader::unsetHalReader()
{
    instance() = nullptr;
}

bool CompilationHalReader::isHalReaderSet()
{
    if (instance() == nullptr)
    {
        return false;
    }
    return true;
}

const std::shared_ptr<HalReader>& CompilationHalReader::getHalReader(bool allowFailure)
{
    const std::shared_ptr<HalReader>& halReader = instance();
    if (!allowFailure)
    {
        HB_ASSERT_PTR(halReader);
    }
    return halReader;
}

CompilationHalReaderSetter::CompilationHalReaderSetter(const HabanaGraph* g)
{
    CompilationHalReader::setHalReader(g->getHALReader());
    GlobalConfManager::instance().setDeviceType(g->getDeviceType());
}

CompilationHalReaderSetter::~CompilationHalReaderSetter()
{
    CompilationHalReader::unsetHalReader();
}