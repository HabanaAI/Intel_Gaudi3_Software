#pragma once
#include <string>

class FS_Mme;
namespace Gaudi2
{
namespace Mme
{
class Unit
{
   public:
    Unit(FS_Mme* mme, const std::string& name) : m_mme(mme), m_name(name) {}

    virtual ~Unit() {}

    void setName(const std::string& name) { m_name = name; }

    const std::string& getInstanceName() const { return m_name; }

    FS_Mme* getMme() const { return m_mme; }

   protected:
    FS_Mme*     m_mme;
    std::string m_name;
};
} // namespace Mme
} // namespace Gaudi2
