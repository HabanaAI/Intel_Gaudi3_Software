#pragma once
#include <iosfwd>
#include <memory>
#include <vector>
#include "defs.h"
#include "segment.h"
#include "settable.h"


// This class assumes all offsets are relative to loadDescReg ()
class DescriptorShadow
{
public:
    enum class RegisterDataHandling
    {
        Ignore,
        Data,
        AlwaysWrite,
        Patching,
        DynamicShapePatching,
        AlwaysWritePatching,
        OptOutPatching,
        Banned,
        Undefined
    };

    enum class WriteType
    {
        NoWrite,
        WriteExecute,
        WritePatching
    };

    struct RegisterProperties
    {
        bool ignore              : 1;
        bool patching            : 1;
        bool alwaysWrite         : 1;
        bool dynamicPatching     : 1;
        bool banned              : 1;
        bool alwaysWritePatching : 1;
        bool optOutPatching      : 1;
        // Here we can add more properties
        // If none of the property bits is turned on, the handling is Data
        // If no property object is even exist for a register, the handling would be Undefined

        static RegisterProperties createFromHandling(DescriptorShadow::RegisterDataHandling handling);
        static RegisterProperties getGeneralData()         { return {false, false, false, false, false, false, false}; }
        static RegisterProperties getIgnore()              { return {true,  false, false, false, false, false, false}; }
        static RegisterProperties getPatching()            { return {false, true,  false, false, false, false, false}; }
        static RegisterProperties getDynamicPatching()     { return {false, true,  false, true,  false, false, false}; }
        static RegisterProperties getAlwaysWritePatching() { return {false, true,  true,  false, false, true , false}; }
        static RegisterProperties getAlwaysWrite()         { return {false, false, true,  false, false, false, false}; }
        static RegisterProperties getBanned()              { return {false, false, false, false, true,  false, false}; }
        static RegisterProperties getOptOutPatching()      { return {false, false, false, false, false, false, true};  }

        RegisterDataHandling getHandling() const;
        void addHandling(RegisterDataHandling handling);

        friend std::ostream& operator<<(std::ostream& stream, const RegisterProperties& rhs);
        bool operator==(const RegisterProperties& rhs) const;
    };

    using AllRegistersProperties  = std::shared_ptr<std::vector<RegisterProperties>>;
    using StartEndAndHandling     = std::tuple<uint32_t, uint32_t, RegisterDataHandling>;
    using EndAndHandling          = std::pair<uint32_t, RegisterDataHandling>;

    DescriptorShadow() = default;
    DescriptorShadow(AllRegistersProperties allRegProperties) : m_allRegProperties(std::move(allRegProperties)) {}

    static void                   setRegisterPropertyOnSegment(std::vector<RegisterProperties>& props, const Segment& seg, const RegisterProperties& prop);
    static AllRegistersProperties createPropertiesToAllRegs(uint32_t totalsize, std::initializer_list<StartEndAndHandling> listHandling);
    static AllRegistersProperties createPropertiesToAllRegs(uint32_t totalsize, std::initializer_list<std::tuple<Segment, RegisterDataHandling>> listHandling);
    static AllRegistersProperties createPropertiesToAllRegsByEnds(uint32_t totalsize, std::initializer_list<EndAndHandling> listHandling);
    WriteType                     getWriteType(uint32_t val, uint32_t regIndex) const;
    RegisterDataHandling          getRegHandling(uint32_t index) const;
    RegisterDataHandling          getPastRegHandling(uint32_t index) const;
    void                          addRegHandlingAt(uint32_t index, RegisterDataHandling handling) const;
    void                          updateLoadedReg(uint32_t index, uint32_t data);
    void                          invalidateRegs(std::vector<uint32_t>& indicies);
    void                          updateLoadedSegment(uint32_t start, uint32_t end, const uint32_t* data); // end is non-inclusive
    bool                          canJoin(const Segment& seg1, const Segment& seg2, unsigned offset) const;
    void                          flush();
    Settable<uint32_t>            getDataAt(uint32_t index) const;
    void                          ensureVectorBigEnough(uint32_t index);
    RegisterProperties            propertiesAt(uint32_t index) const;
    void                          setPropertiesAt(uint32_t index, const RegisterProperties& prop);
    void                          printAllHandling(std::ostream& stream);
    bool                          isPatching(RegisterDataHandling handling) const;
    bool                          isSkippable(RegisterDataHandling handling) const;

    void setAllRegProperties(AllRegistersProperties props) { m_allRegProperties = std::move(props); }
    void movePresentPropertiesToPastProperties()           { m_loadedRegProperties = std::move(m_allRegProperties); }
    bool hasData() const                                   { return !m_data.empty(); }
    const std::vector<uint32_t>& data() const              { return m_data; }

private:
    AllRegistersProperties  m_allRegProperties = nullptr;    // properties of present descriptor
    AllRegistersProperties  m_loadedRegProperties = nullptr; // properties of past descriptor
    std::vector<uint32_t>   m_data;                          // registers loaded value (i.e. the last history value)
    std::vector<bool>       m_loadedRegsValidityMask;        // a flag for each m_data entry to indicate its validity
};
