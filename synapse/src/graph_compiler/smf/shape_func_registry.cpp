#include <string>
#include "smf/shape_func_registry.h"
#include "defs.h"
#include "kernel_db.h"

#include "sif/static_common_sif.h"

#include "smf/static_common_smf.h"
#include "platform/gaudi/graph_compiler/smf/static_gaudi_smf.h"
#include "platform/gaudi2/graph_compiler/smf/static_gaudi2_smf.h"
#include "platform/gaudi3/graph_compiler/smf/static_gaudi3_smf.h"

// This function is used to init the ShaperFuncRegistry
void initShapeFuncRegistry(synDeviceType deviceType)
{
    LOG_TRACE_T(KERNEL_DB, "{}", HLLOG_FUNC);
    ShapeFuncRegistry::instance().init(deviceType);

    // SIFs are device independent
    KernelDB::instance().registerSif();

#ifdef SFR_STATS
    ShapeFuncRegistry::instance().initStats();
#endif
}

static constexpr char unknownName[] = "???";

ShapeFuncRegistry& ShapeFuncRegistry::instance()
{
    static ShapeFuncRegistry instance;
    return instance;
}

ShapeFuncRegistry::ShapeFuncRegistry() {}

void ShapeFuncRegistry::destroy()
{
#ifdef SFR_STATS
    delete m_pSifStats;
    delete m_pSmfStats;
#endif

    m_initDone = false;
    m_smfBank.clear();
    m_sifBank.clear();

#ifdef SFR_STATS
    m_sifMap.clear();
    m_sifStatIdx.clear();

    m_smfMap.clear();
    m_smfStatIdx.clear();
#endif
}

void ShapeFuncRegistry::init(synDeviceType deviceType)
{
    HB_ASSERT(m_initDone == false, "ShapeFuncRegistry already init");

    LOG_DEBUG_T(KERNEL_DB, "ShapeFuncRegistry initDsdTable SMF & SIF");

    // always register SIFs, they are all device-independent
    registerSIFTable(staticSIFs, sizeof(staticSIFs) / sizeof(staticSIFs[0]));

    // always register commom device independent SMFs
    registerSMFTable(commonStaticSMFs, sizeof(commonStaticSMFs) / sizeof(commonStaticSMFs[0]));

    // register

    if (deviceType == synDeviceType::synDeviceGaudi || deviceType == synDeviceType::synDeviceTypeInvalid)
    {
        registerSMFTable(gaudi::staticSMFs, sizeof(gaudi::staticSMFs) / sizeof(gaudi::staticSMFs[0]));
    }

    if (deviceType == synDeviceType::synDeviceGaudi2 || deviceType == synDeviceType::synDeviceGaudi3 || deviceType == synDeviceType::synDeviceTypeInvalid)
    {
        registerSMFTable(gaudi2::staticSMFs, sizeof(gaudi2::staticSMFs) / sizeof(gaudi2::staticSMFs[0]));
    }

    if (deviceType == synDeviceType::synDeviceGaudi2 || deviceType == synDeviceType::synDeviceTypeInvalid)
    {
        registerSMFTable(gaudi3::staticSMFs, sizeof(gaudi3::staticSMFs) / sizeof(gaudi3::staticSMFs[0]));
    }

    m_initDone = true;
}

void ShapeFuncRegistry::registerSMFTable(const StaticSmfEntry* smfTable, unsigned tableSize)
{
    for (unsigned i = 0; i < tableSize; i++)
    {
        registerSMF(smfTable[i].id, smfTable[i].pFunc, std::string {smfTable[i].name});
    }
}

void ShapeFuncRegistry::registerSIFTable(const StaticSifEntry* sifTable, unsigned tableSize)
{
    for (unsigned i = 0; i < tableSize; i++)
    {
        registerSIF(sifTable[i].id, sifTable[i].pFunc, std::string {sifTable[i].name}, GC_SIF_VERSION);
    }
}

void ShapeFuncRegistry::registerSMF(ShapeFuncID id, smf_t pFunc, const std::string& name)
{
    sm_function_id_t key;
    key.sm_tableid = LIB_ID_RESERVED_FOR_GC_SMF;
    key.sm_funcid  = id;

    LOG_TRACE_T(KERNEL_DB, "{}: id 0x{:x} pFunc 0x{:x} {}", HLLOG_FUNC, key.sm_func_index, TO64(pFunc), name);

    auto ret = m_smfBank.emplace(key.sm_func_index, SmfInfo {pFunc, name});

    if (ret.second == false)
    {
        if (ret.first->second.func != pFunc)
        {
            LOG_ERR_T(KERNEL_DB,
                      "SMF, a different function is already registered, replacing. Id 0x{:x} pFunc 0x{:x} current "
                      "0x{:x} name {}",
                      key.sm_func_index,
                      TO64(pFunc),
                      TO64(ret.first->second.func),
                      ret.first->second.name);
            ret.first->second = {pFunc, name};
        }
    }
}

void ShapeFuncRegistry::registerSIF(ShapeFuncID        id,
                                    sif_t              pFunc,
                                    const std::string& name,
                                    uint64_t           version,
                                    ShapeFuncOrigin    originator /*= LIB_ID_RESERVED_FOR_GC_SIF*/)
{
    sm_function_id_t key;
    key.sm_tableid = originator;
    key.sm_funcid  = id;
    registerSIF(key, pFunc, name, version);
}

void ShapeFuncRegistry::registerSIF(sm_function_id_t id, sif_t pFunc, const std::string& name, uint64_t version)
{
    tpc_lib_api::GuidInfo guid = {0};

    guid.nameHash.hashValue      = id.sm_funcid;
    guid.nameHash.sharedObjectId = id.sm_tableid;
    strncpy(guid.name, name.c_str(), tpc_lib_api::MAX_NODE_NAME - 1);

    registerSIF(id, pFunc, version, guid);
}

void ShapeFuncRegistry::registerSIF(sm_function_id_t             id,
                                    sif_t                        pFunc,
                                    uint64_t                     version,
                                    const tpc_lib_api::GuidInfo& guid)
{
    LOG_TRACE_T(KERNEL_DB,
                "{}: id 0x{:x} pFunc 0x{:x} name {} version {}",
                HLLOG_FUNC,
                id.sm_func_index,
                TO64(pFunc),
                guid.name,
                version);

    auto ret = m_sifBank.emplace(id.sm_func_index, SifInfo {pFunc, version, guid.name, guid});

    if (ret.second == false)
    {
        if (ret.first->second.func != pFunc)
        {
            LOG_WARN_T(KERNEL_DB,
                       "SIF, a different function is already registered, overwriting. Id 0x{:x} pFunc 0x{:x} current "
                       "0x{:x} version 0x{:x} name {}",
                       id.sm_func_index,
                       TO64(pFunc),
                       TO64(ret.first->second.func),
                       ret.first->second.version,
                       ret.first->second.name);
            ret.first->second = {pFunc, version, guid.name};
        }
    }
}

smf_t ShapeFuncRegistry::getSMF(ShapeFuncID id)
{
    sm_function_id_t key;
    key.sm_tableid = LIB_ID_RESERVED_FOR_GC_SMF;
    key.sm_funcid  = id;
    return getSMF(key);
}

sif_t ShapeFuncRegistry::getSIF(ShapeFuncID id, ShapeFuncOrigin originator /*= LIB_ID_RESERVED_FOR_GC_SIF*/)
{
    sm_function_id_t key;
    key.sm_tableid = originator;
    key.sm_funcid  = id;
    return getSIF(key);
}

smf_t ShapeFuncRegistry::getSMF(sm_function_id_t id)
{
    HB_ASSERT(id.sm_tableid == LIB_ID_RESERVED_FOR_GC_SMF, "SMF originator must be GC");
    auto it = m_smfBank.find(id.sm_func_index);
    return (it != m_smfBank.end()) ? it->second.func : nullptr;
}

const char* ShapeFuncRegistry::getSmfName(sm_function_id_t id)
{
    auto it = m_smfBank.find(id.sm_func_index);
    return (it != m_smfBank.end()) ? it->second.name.c_str() : unknownName;
}

sif_t ShapeFuncRegistry::getSIF(sm_function_id_t id)
{
    auto it = m_sifBank.find(id.sm_func_index);
    return (it != m_sifBank.end()) ? it->second.func : nullptr;
}
std::pair<sif_t, tpc_lib_api::GuidInfo*> ShapeFuncRegistry::getSIFandGuidInfo(sm_function_id_t id)
{
    auto it = m_sifBank.find(id.sm_func_index);
    return (it != m_sifBank.end()) ? std::make_pair(it->second.func, &it->second.guid)
                                   : std::make_pair(nullptr, nullptr);
}

const char* ShapeFuncRegistry::getSifName(sm_function_id_t id)
{
    auto it = m_sifBank.find(id.sm_func_index);
    return (it != m_sifBank.end()) ? it->second.name.c_str() : unknownName;
}

uint64_t ShapeFuncRegistry::getSifVersion(sm_function_id_t id)
{
    auto it = m_sifBank.find(id.sm_func_index);
    return (it != m_sifBank.end()) ? it->second.version : (uint64_t)(-1);
}

#ifdef SFR_STATS

#include <sstream>

// init ShapeFuncRegistry statistics. It builds the needed information to construct
// the statistics and create them
void ShapeFuncRegistry::initStats()
{
    buildSifStatIdx();
    m_pSifStats = new StatisticsVec("SIF time", m_sifStatIdx, 1, true);

    buildSmfStatIdx();
    m_pSmfStats = new StatisticsVec("SMF time", m_smfStatIdx, 1, true);
}

template<class func, typename info>
void ShapeFuncRegistry::buildStatIdx(std::unordered_map<uint32_t, info>& bank,
                                     std::map<func, IdxAndName>&         map,
                                     std::vector<StatEnumMsg<int>>&      stat,
                                     const std::string&                  msg)
{
    for (auto& x : bank)
    {
        std::stringstream stream;
        info&             funcInfo = x.second;
        stream << std::hex << x.first << ":" << funcInfo.name;
        std::string name(stream.str());

        map.emplace(funcInfo.func, std::make_pair(-1, name));  //-1 is a placeholder, will be updated later
    }
    // Copy to a vector (used by stats class)
    uint32_t seqNum = 1;  // point 0 is for start indication
    stat.push_back({0, msg.c_str()});
    for (auto& x : map)
    {
        IdxAndName& idxAndName = x.second;
        idxAndName.first       = seqNum++;
        stat.push_back({idxAndName.first, idxAndName.second.c_str()});
    }
    stat.push_back({seqNum, "unknown"});  // This is needed for test
}

// This function builds the needed vector for the constructor of the Sif statistics
// It first creates a map: key is function pointer, value is a pair of sequence number and description text
// Then it copies from the set to the vector needed for the statistics constructor
void ShapeFuncRegistry::buildSifStatIdx()
{
    // build a map: sifFunc -> pair(seqNum, string(table idx + description)) - need a unique sif function
    // Lets say the bank is {{2,{2000, "S2"}}, {5,{1000,"S5"}}, {4,{3000,"S4"}}} - index/funcPointer
    // sifMap will be: {1000,{1,"5 S5"}}, {2000,{2,"2 S2"}},{3000,{3,"4 S4"}} - order by funcPointer
    // sifStatIdx will be vector of {{0,"SIF End}, {1,"5 S5"}, {2,"2 S2"},{3,"4 S4"}}
    // When logging, the user gives the func (lets say 3000). From the map we get the 3, which is used as the point to
    // collect
    buildStatIdx(m_sifBank, m_sifMap, m_sifStatIdx, "SIF End");
}

// This function builds the needed vector for the constructor of the Sif statistics
// It first creates a map: key is function pointer, value is a pair of sequence number and description text
// Then it copies from the set to the vector needed for the statistics constructor
void ShapeFuncRegistry::buildSmfStatIdx()
{
    // See buildSifStatIdx for explanation
    buildStatIdx(m_smfBank, m_smfMap, m_smfStatIdx, "SMF End");
}

// This function should be called after a sif function was ran with the function pointer and time it took to run the sif
// The function finds the sequence number of the sif function and call collect to add it to the statistics

void ShapeFuncRegistry::sifStatCollect(sif_t pFunc, uint64_t sum)
{
    if (pFunc == nullptr)
    {
        m_pSifStats->collect(0, sum);
    }
    else
    {
        auto x = m_sifMap.find(pFunc);
        if (x == m_sifMap.end())
        {
            m_pSifStats->collect(m_sifStatIdx.size() - 1, sum);
            //            dumpSifSet();
            return;
        }
        m_pSifStats->collect(x->second.first, sum);
    }
}

// This function should be called after a smf function was ran with the function pointer and time it took to run the smf
// The function finds the sequence number of the sif function and call collect to add it to the statistics
void ShapeFuncRegistry::smfStatCollect(smf_t pFunc, uint64_t sum)
{
    if (pFunc == nullptr)
    {
        m_pSmfStats->collect(0, sum);
    }
    else
    {
        auto x = m_smfMap.find(pFunc);
        if (x == m_smfMap.end())
        {
            m_pSmfStats->collect(m_smfStatIdx.size() - 1, sum);
            //            dumpSmfSet();
            return;
        }
        m_pSmfStats->collect(x->second.first, sum);
    }
}

// This function is for debug only, never called. Used to dump the sif map
void ShapeFuncRegistry::dumpSifMap()
{
    int i = 0;
    for (auto y : m_sifMap)
    {
        printf("%2X) sif %lX idx %x name %s\n", i++, TO64(y.first), y.second.first, y.second.second.c_str());
    }
}

// This function is for debug only, never called. Used to dump the smf map
void ShapeFuncRegistry::dumpSmfMap()
{
    int i = 0;
    for (auto y : m_smfMap)
    {
        printf("%2X) sif %lX idx %x name %s\n", i++, TO64(y.first), y.second.first, y.second.second.c_str());
    }
}

#endif

smf_callbacks_t* SmfCallbacks::m_smfCallbacks = nullptr;
