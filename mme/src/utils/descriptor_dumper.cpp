#include "include/utils/descriptor_dumper.h"
#include "mme_assert.h"
#include "spdlog/common.h"

#include <cstdlib>  // gentenv
#include <fstream>

template<typename Desc>
void MmeDescriptorDumper<Desc>::setFileNamePrefix(const std::string& prefix)
{
    if (!prefix.empty())
    {
        m_filenamePrefix = prefix;
    }
}

template<typename Desc>
std::string MmeDescriptorDumper<Desc>::dump(const Desc& desc, unsigned descIdx, bool asBinary, bool fullDump)
{
    std::string suffix;
    m_dumpPath = getHabanaLogsPath();
    m_dumpPath.append("mme_debug");
    fs::create_directory(m_dumpPath);
    if (asBinary)
    {
        m_dumpPath.append(m_filenamePrefix + "mme_descriptor_dump_" + std::to_string(descIdx) + ".bin");
        dumpAsBinary(desc);
    }
    else
    {
        m_dumpPath.append(m_filenamePrefix + "mme_descriptor_dump_" + std::to_string(descIdx) + ".txt");
        dumpAsText(desc, fullDump);
    }
    return m_stream.str();
}

template<typename Desc>
void MmeDescriptorDumper<Desc>::dumpAddress(uint64_t address, const std::string& context, bool addEndL)
{
    m_stream << fmt::format("{}: {:#x}  ", context, address);
    if (addEndL)
    {
        m_stream << std::endl;
    }
}
template<typename Desc>
void MmeDescriptorDumper<Desc>::dumpUnsignedHex(unsigned val, const std::string& context, bool addEndL)
{
    m_stream << fmt::format("{} : {:#x}  ", context, val);
    if (addEndL)
    {
        m_stream << std::endl;
    }
}
template<typename Desc>
void MmeDescriptorDumper<Desc>::dumpAsBinary(const Desc& desc)
{
    std::ofstream dumpFile;
    dumpFile.open(m_dumpPath.c_str(), std::ios::binary);
    MME_ASSERT(dumpFile.is_open(), fmt::format("could not open file {}", m_dumpPath.c_str()).c_str());
    dumpFile.write((char*) &desc, sizeof(Desc));
    dumpFile.close();
}
template<typename Desc>
void MmeDescriptorDumper<Desc>::dumpAsText(const Desc& desc, bool fullDump)
{
    dumpDescriptor(desc, fullDump);
    std::ofstream dumpFile;
    dumpFile.open(m_dumpPath.c_str(), std::ios::binary);
    MME_ASSERT(dumpFile.is_open(), fmt::format("could not open file {}", m_dumpPath.c_str()).c_str());
    dumpFile << m_stream.str();
    dumpFile.close();
}
template<typename Desc>
fs::path MmeDescriptorDumper<Desc>::getHabanaLogsPath() const
{
    if (const char* e = getenv("HABANA_LOGS"); e != nullptr) return e;
    if (const char* e = getenv("HOME"); e != nullptr) return fmt::format("{}/.habana_logs", e);
    return {};
}

#include "gaudi/mme.h"
template class MmeDescriptorDumper<Mme::Desc>;
#include "gaudi2/mme.h"
template class MmeDescriptorDumper<Gaudi2::Mme::Desc>;
#include "gaudi3/mme.h"
template class MmeDescriptorDumper<gaudi3::Mme::Desc>;