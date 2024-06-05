#ifndef MME__DESCRIPTOR_DUMPER_H
#define MME__DESCRIPTOR_DUMPER_H

#include "include/general_utils.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

template<typename Desc>
class MmeDescriptorDumper
{
public:
    void setFileNamePrefix(const std::string& prefix = "");
    std::string dump(const Desc& desc, unsigned descIdx, bool asBinary = false, bool fullDump = false);
    template<class T>
    inline MmeDescriptorDumper& operator<<(T data)
    {
        m_stream << data;
        return *this;
    }
    inline void clear() { m_stream.str(""); }

protected:
    MmeDescriptorDumper() = default;
    virtual ~MmeDescriptorDumper() = default;
    virtual void dumpDescriptor(const Desc& desc, bool fullDump) = 0;
    template<typename T>
    void dumpArray(const T& array, const std::string& context, bool addEndL = false)
    {
        m_stream << context << ": [" << arrayToStr(std::begin(array), std::end(array)) << "]";
        if (addEndL)
        {
            m_stream << std::endl;
        }
    }
    void dumpAddress(uint64_t address, const std::string& context, bool addEndL = false);
    void dumpUnsignedHex(unsigned val, const std::string& context, bool addEndL = false);
    std::stringstream m_stream;

private:
    void dumpAsBinary(const Desc& desc);
    void dumpAsText(const Desc& desc, bool fullDump);
    inline fs::path getHabanaLogsPath() const;
    fs::path m_dumpPath;
    std::string m_filenamePrefix;
};

#endif //MME__DESCRIPTOR_DUMPER_H
