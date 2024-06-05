#include "kernel_db.h"

#include "complex_guid_extractor.h"
#include "infra/defs.h"
#include "types_exception.h"
#include "utils.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <stdlib.h>  // getenv

std::string_view KernelDB::parseReturnValue(tpc_lib_api::GlueCodeReturn ret)
{
#if MAGIC_ENUM_SUPPORTED
    return magic_enum::enum_name(ret);
#else
    switch (ret)
    {
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_SUCCESS)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_NODE_NOT_FOUND)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INSUFFICIENT_ISA_BUFFER)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_COUNT)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INCOMPATIBLE_INPUT_SIZE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_COUNT)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INCOMPATIBLE_OUTPUT_SIZE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INCOMPATIBLE_DATA_TYPE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_LAYER_CONFIGURATION)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INSUFFICIENT_AUX_BUFFER_SIZE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_QUANT_PARAMS)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_BROADCAST_MODE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_API_VERSION)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_NON_STATIC_INPUT_TENSOR)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_KERNEL_REQUIRE_REDUCIBLE_TENSOR)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INVALID_SHAPE_INFERENCE_ID)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_KERNEL_INVALID_SCALAR_ARGUMENT)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_LOW_FCD_INPUT)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_INSUFFICIENT_ELF_BUFFER)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_MISSING_PRIVATE_STRUCTURE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_5D_TENSORS)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_CGUID_GRAPH_UNCHANGED)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_SIF_NULL_PTR)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_DYNAMIC_SHAPE)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_UNSUPPORTED_HUGE_TENSORS)
        TRANSLATE_ENUM_TO_STRING(tpc_lib_api::GLUE_FAILED)

        default:
            return "UNKNOWN RETURN VALUE";
    }
#endif
}

std::string_view KernelDB::deviceIdToString(tpc_lib_api::DeviceId deviceId)
{
    static const std::string S_DEVICE_STRINGS[] = {"Gaudi", "Greco", "Gaudi2", "Gaudi3"};
    // Device array is not zero based anymore, hence the -1.
    static_assert(ARRAY_SIZE(S_DEVICE_STRINGS) == tpc_lib_api::DEVICE_ID_MAX - 1, "Missing name type");

    return S_DEVICE_STRINGS[deviceId - 1];
}

const char* KernelDB::KERNEL_LIB_ENV_VAR = "GC_KERNEL_PATH";

KernelDB& KernelDB::instance()
{
    static auto instance = std::unique_ptr<KernelDB>(new KernelDB());
    return *instance;
}

void KernelDB::init(tpc_lib_api::DeviceId deviceId)
{
    HB_ASSERT(deviceId <= tpc_lib_api::DEVICE_ID_MAX && deviceId >= tpc_lib_api::DEVICE_ID_GAUDI,
              "device Id must be between {} to {}",
              tpc_lib_api::DEVICE_ID_GAUDI,
              tpc_lib_api::DEVICE_ID_MAX);

    if (deviceId == tpc_lib_api::DEVICE_ID_MAX)
    {
        for (int id = tpc_lib_api::DEVICE_ID_GAUDI; id < tpc_lib_api::DEVICE_ID_MAX; ++id)
        {
            if (id == tpc_lib_api::DEVICE_ID_GRECO) continue;

            _init(tpc_lib_api::DeviceId(id), true);
        }
    }
    else
    {
        _init(deviceId, false);
    }
}

void KernelDB::_init(tpc_lib_api::DeviceId deviceId, bool useDefaultLibName)
{
    // Don't initialize twice
    if (!m_initializedIds.insert(deviceId).second) return;

    loadKernelsFromEnv(KERNEL_LIB_ENV_VAR, deviceId);
    // If TPC_FUSER enable load share object (and other conditions that we check in TPC Fuser pass)
    if (GCFG_RUN_TPC_FUSER.value())
    {
        TPCFuserSharedObject& tpcFuserSO = TPCFuserSharedObject::instance();

        std::string_view tpcFuserLibName;
        if (useDefaultLibName)
        {
            tpcFuserLibName = GCFG_TPC_FUSER_LIB_NAME.getDefaultValue(deviceIDToDeviceType(deviceId));
        }
        else
        {
            tpcFuserLibName = GCFG_TPC_FUSER_LIB_NAME.value();
        }
        tpcFuserSO.init(tpcFuserLibName);

        if (tpcFuserSO.isInitialized())
        {
            loadFuserKernelsFromPath(tpcFuserSO.getTPCFuserSharedObjectName().c_str());
        }
    }
    initComplexGuidLib(deviceId);

    // SmallVlm feature is supported from gaudi2
    if (deviceId > tpc_lib_api::DEVICE_ID_GAUDI)
    {
        m_featureSupport.addSupportedFeature(deviceId, Feature::SMALL_VLM);
    }
}

KernelDB::KernelDB() {}

void KernelDB::clear()
{
    for (auto h : m_libHandles)
    {
        if (h != nullptr)
        {
            UnloadSharedObject(h);
        }
    }
    m_libHandles.clear();
    std::for_each(m_kernelDB.begin(), m_kernelDB.end(), [](KernelByName& kernelMap) { kernelMap.clear(); });
    std::for_each(m_functionalComplexGuids.begin(), m_functionalComplexGuids.end(), [](ComplexGuidNamesSet& cguidMap) {
        cguidMap.clear();
    });
    std::for_each(m_performanceComplexGuids.begin(),
                  m_performanceComplexGuids.end(),
                  [](ComplexGuidNamesSet& cguidMap) { cguidMap.clear(); });
    std::for_each(m_complexGuidsDB.begin(), m_complexGuidsDB.end(), [](ComplexGuidByName& cguidDSMap) {
        cguidDSMap.clear();
    });
    m_initializedIds.clear();
    // clearing Complex GUID extractor shared object
    ComplexGuidExtractorSharedObject::instance().destroy();
    TPCFuserSharedObject::instance().destroy();
}

void KernelDB::loadKernelsFromEnv(const char* env, tpc_lib_api::DeviceId deviceId)
{
    // attempt to load kernels
    char* pEnv = std::getenv(env);
    if (pEnv == nullptr)
    {
        LOG_ERR(KERNEL_DB, "Environment variable {} is undefined", env);
        return;
    }
    std::string              kernelPathsStr(pEnv);
    std::vector<std::string> kernelPaths = splitString(kernelPathsStr, ':');
    uint8_t                  libIndex    = 0;
    for (auto it = kernelPaths.begin(); it != kernelPaths.end(); ++it)
    {
        const char* kernelPath = (*it).c_str();

        if (kernelPath != nullptr)
        {
            HB_ASSERT(libIndex <= 15, "Maximum number of supported TPC libs is 15");
            LOG_INFO(KERNEL_DB,
                     "Attempting to load kernels from {} for {} device",
                     kernelPath,
                     deviceIdToString(deviceId));
            loadKernelsFromPath(kernelPath, deviceId, LIB_ID_FIRST_GLUE_CODE_SIF, libIndex);
            libIndex++;
        }
        else
        {
            LOG_ERR(KERNEL_DB, "Kernels path is not defined {}", env);
        }
    }
}

void KernelDB::loadKernelsFromPath(const char*           path,
                                   tpc_lib_api::DeviceId deviceId,
                                   ShapeFuncOrigin       shapeFuncLib,
                                   uint8_t               libIndex)
{
    libHandle h = LoadSharedObject(path);
    if (h == nullptr)
    {
        LOG_ERR(KERNEL_DB, "Attempt to load kernels from {} failed, could not open file", path);
        return;
    }

    uint64_t version = 0;
    if (!getLibraryVersion(h, version))
    {
        LOG_WARN(KERNEL_DB, "Failed loading version number from {}", path);
    }

    if (!loadKernels(h, deviceId, version, shapeFuncLib, libIndex))
    {
        LOG_ERR(KERNEL_DB, "Failed loading kernels from {} for {} device", path, deviceIdToString(deviceId));
        UnloadSharedObject(h);
        return;
    }

    if (m_kernelDB[deviceId].empty())
    {
        LOG_WARN(KERNEL_DB, "No Kernels registered for {} device", deviceIdToString(deviceId));
    }
    m_libHandles.push_back(h);
    m_libsVersions[path] = version;
}

void KernelDB::loadFuserKernelsFromPath(const char* path)
{
    libHandle h = LoadSharedObject(path);
    if (h == nullptr)
    {
        LOG_ERR(KERNEL_DB, "Attempt to load fused kernels from {} failed, could not open file", path);
        return;
    }

    uint64_t version = 0;
    if (!getLibraryVersion(h, version))
    {
        LOG_WARN(KERNEL_DB, "Failed loading version number from {}", path);
    }

    tpc_lib_api::pfnGetKernelGuids getGuids;

    m_fuserMeta.libraryVersion                   = GC_SIF_VERSION;
    m_fuserMeta.guidInfo.nameHash.hashValue      = SIF_SPLIT_FUSED;  // Fuser SIF is implemented on GC side
    m_fuserMeta.guidInfo.nameHash.sharedObjectId = ShapeFuncOrigin::LIB_ID_RESERVED_FOR_GC_SIF;
    m_fuserMeta.guidInfo.supportsDynamicShapes   = true;
    if (!loadFunctions(h,
                       m_fuserMeta.getInstance,
                       m_fuserMeta.getLayouts,
                       m_fuserMeta.getSuggestedManipulation,
                       getGuids))
    {
        LOG_ERR(KERNEL_DB, "Error loading fuser functions");
        UnloadSharedObject(h);
    }
    else
    {
        m_libHandles.push_back(h);
        m_libsVersions[path] = version;
    }
}

void KernelDB::initComplexGuidLib(tpc_lib_api::DeviceId deviceId)
{
    if (GCFG_COMPLEX_GUID_EXTRACTOR_MODE.value() != ComplexGUIDExtractorModeDisabled)
    {
        const std::string libPath = GCFG_COMPLEX_GUID_LIB_NAME.value();
        libHandle         h       = LoadSharedObject(libPath.c_str());
        if (h == nullptr)
        {
            LOG_ERR(KERNEL_DB,
                    "Attempt to load supported complex guids from {} failed, could not open file. dlerror : {}",
                    libPath,
                    dlerror());
            return;
        }
        // load library version
        uint64_t version = 0;
        if (!getLibraryVersion(h, version))
        {
            LOG_ERR(KERNEL_DB, "Failed loading complex guid lib version number from {}", libPath);
            UnloadSharedObject(h);
            return;
        }
        // load supported complex guids to kernel DB from entry point in TPC fuser lib
        if (!loadSupportedComplexGuids(h, deviceId, version))
        {
            LOG_ERR(KERNEL_DB, "Failed loading supported complex guids from {} , device id {}", libPath, deviceId);
            UnloadSharedObject(h);
            return;
        }
        m_libHandles.push_back(h);
        m_libsVersions[libPath] = version;
        // init the shared lib object used in the complex guid extractor pass
        ComplexGuidExtractorSharedObject::instance().init();
    }
}

bool KernelDB::loadSupportedComplexGuids(libHandle h, tpc_lib_api::DeviceId deviceId, uint64_t complexGuidLibVersion)
{
    LOG_TRACE(KERNEL_DB, "loading complex guids data into kernel DB, library version {}", complexGuidLibVersion);
    tpc_lib_api::pfnGetSupportedDataLayout     getCguidLayout;
    tpc_lib_api::pfnGetFunctionalComplexGuids  getFunctionalCguids;
    tpc_lib_api::pfnGetPerformanceComplexGuids getPerformanceCguids;
    tpc_lib_api::pfnGetShapeInference          getCguidSifResult;
    //  load the interface functions - getters for names, SIFs and data layouts
    if (!loadComplexGuidFunctions(h, getCguidLayout, getFunctionalCguids, getPerformanceCguids, getCguidSifResult))
    {
        LOG_ERR(KERNEL_DB, "Failed to load complex guid functions for {} device.", deviceIdToString(deviceId));
        return false;
    }

    // load the supported complex guid names
    std::vector<tpc_lib_api::GuidInfo> functionalCguids;
    std::vector<tpc_lib_api::GuidInfo> performanceCguids;
    // Load supported functional CGUID.
    if (!loadKernelGuids<tpc_lib_api::pfnGetFunctionalComplexGuids>(getFunctionalCguids, deviceId, functionalCguids))
    {
        LOG_ERR(KERNEL_DB,
                "Failed to load supported functional complex guid names for {} device. dlerror {}",
                deviceIdToString(deviceId),
                dlerror());
        return false;
    }
    if (functionalCguids.size() == 0)
    {
        LOG_WARN(KERNEL_DB,
                 "Functional complex guid names array is null, can't load names for {} device.",
                 deviceIdToString(deviceId));
    }
    // Load supported performance CGUID.
    if (!loadKernelGuids<tpc_lib_api::pfnGetPerformanceComplexGuids>(getPerformanceCguids, deviceId, performanceCguids))
    {
        LOG_ERR(KERNEL_DB,
                "Failed to load supported performance complex guid names for {} device. dlerror {}",
                deviceIdToString(deviceId),
                dlerror());
        return false;
    }
    if (performanceCguids.size() == 0)
    {
        LOG_WARN(KERNEL_DB,
                 "Performance complex guid names array is null, can't load names for {} device.",
                 deviceIdToString(deviceId));
    }

    // lambda function to add CGUID to the DB, and update its metadata.
    auto addComplexGuidAndUpdateMetaData = [&](tpc_lib_api::GuidInfo& cguid) {
        ComplexGuidNameAndHash guidAndHash(std::string(cguid.name));
        // emplace returns a pair(first: insertedElement, second: isNewElement)
        auto emplaceRes      = m_complexGuidsDB[deviceId].try_emplace(guidAndHash);
        bool isNewElement    = emplaceRes.second;
        auto insertedElement = emplaceRes.first;
        // if cguid in DB, no need to update (only if it was static).
        // When we had different entry points, we always gave dynamic a higher priority over static (for layouts we
        // returned the dynamic pointer if it existed, and only then static). Now we overwrite it, to mimic the same
        // behavior.
        if (!isNewElement && insertedElement->second.isDynamicShape()) return;
        // Each element in DB is a pair(first: cguidName, second: cguidMetaData)
        KernelMetadata& cguidMetadata                  = insertedElement->second;
        cguidMetadata.libraryVersion                   = complexGuidLibVersion;
        cguidMetadata.guidInfo                         = cguid;
        cguidMetadata.guidInfo.nameHash.hashValue      = guidAndHash.getHash(); // CGUID lib is not setting hash TODO: Remove when implemented
        cguidMetadata.guidInfo.nameHash.sharedObjectId = ShapeFuncOrigin::LIB_ID_COMPLEX_GUID_SIF;
        cguidMetadata.getLayouts                       = getCguidLayout;
        cguidMetadata.getSifResult                     = getCguidSifResult;
    };

    // lambda function to fill CGUID names sets (functional and performance), and add it to the DB.
    auto fillComplexGuidNamesSets = [&](std::vector<tpc_lib_api::GuidInfo>& kernelInfos, bool isFunctional) {
        std::string                   typeStr  = isFunctional ? "functional" : "performance";
        ComplexGuidNamesPerDeviceSet& cguidSet = isFunctional ? m_functionalComplexGuids : m_performanceComplexGuids;
        for (unsigned i = 0; i < kernelInfos.size(); i++)
        {
            ComplexGuidNameAndHash guidAndHash(std::string(kernelInfos[i].name));
            LOG_TRACE(KERNEL_DB, "Adding {} to {} complex guid names set", guidAndHash.getKey(), typeStr);
            cguidSet[deviceId].insert(guidAndHash);
            addComplexGuidAndUpdateMetaData(kernelInfos[i]);
        }
    };
    fillComplexGuidNamesSets(functionalCguids, true);
    fillComplexGuidNamesSets(performanceCguids, false);

    return true;
}

bool KernelDB::getLibraryVersion(libHandle h, uint64_t& version)
{
    fnHandle fn = GetFunction(h, GET_LIB_VERSION_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_DEBUG(KERNEL_DB, "No entry point for {} in the kernel DB", GET_LIB_VERSION_ENTRY_POINT_NAME);
        return false;
    }
    tpc_lib_api::pfnGetLibVersion getLibraryVersion = (tpc_lib_api::pfnGetLibVersion)fn;

    version = getLibraryVersion();

    LOG_INFO(KERNEL_DB, "Library version {}", version);
    return true;
}

const std::unordered_map<std::string, uint32_t>& KernelDB::GetLibraryVersions() const
{
    return m_libsVersions;
}

bool KernelDB::loadKernels(libHandle             h,
                           tpc_lib_api::DeviceId deviceId,
                           uint64_t              libVersion,
                           ShapeFuncOrigin       shapeFuncLib,
                           uint8_t               libIndex)
{
    tpc_lib_api::pfnInstantiateTpcKernel     instantiate;
    tpc_lib_api::pfnGetSupportedDataLayout   getLayout;
    tpc_lib_api::pfnGetSuggestedManipulation getManipulation;
    tpc_lib_api::pfnGetKernelGuids           getGuids;
    tpc_lib_api::pfnGetShapeInference        getSifResult;

    std::vector<tpc_lib_api::GuidInfo> guids;

    if (!loadFunctions(h, instantiate, getLayout, getManipulation, getGuids, getSifResult))
    {
        LOG_ERR(KERNEL_DB, "Error loading functions");
        return false;
    }

    if (!loadKernelGuids<tpc_lib_api::pfnGetKernelGuids>(getGuids, deviceId, guids))
    {
        LOG_ERR(KERNEL_DB, "Failed loading kernel guids");
        return false;
    }

    for (unsigned i = 0; i < guids.size(); ++i)
    {
        const auto&    kernelInfo = guids[i];
        StringWithHash guidAndHash(kernelInfo.name);
        LOG_TRACE(KERNEL_DB, "Registering kernel {}", guidAndHash.getKey());

        // Don't register new entry points for kernels already in the DB
        auto            emplaceRes = m_kernelDB[deviceId].try_emplace(guidAndHash);
        KernelMetadata& kernelMeta = emplaceRes.first->second;
        if (!emplaceRes.second)
        {
            auto allowedDuplicateKernel = GCFG_ALLOW_DUPLICATE_KERNELS.value() || isFusedKernel(guidAndHash.getKey());
            if (kernelMeta.isDynamicShape() && !allowedDuplicateKernel)
            {
                throw SynapseException(
                    fmt::format("Kernel {} was already registered by another perf-lib", guidAndHash.getKey()));
            }
            continue;
        }

        kernelMeta.libraryVersion                   = libVersion;
        kernelMeta.guidInfo                         = kernelInfo;
        kernelMeta.guidInfo.nameHash.sharedObjectId = ShapeFuncOrigin::LIB_ID_FIRST_GLUE_CODE_SIF;
        kernelMeta.getLayouts                       = getLayout;
        kernelMeta.getInstance                      = instantiate;
        kernelMeta.getSuggestedManipulation         = getManipulation;
        kernelMeta.getSifResult                     = getSifResult;
    }

    return true;
}

bool KernelDB::loadFunctions(libHandle                                 h,
                             tpc_lib_api::pfnInstantiateTpcKernel&     instantiate,
                             tpc_lib_api::pfnGetSupportedDataLayout&   getLayout,
                             tpc_lib_api::pfnGetSuggestedManipulation& getManipulation,
                             tpc_lib_api::pfnGetKernelGuids&           getGuids,
                             tpc_lib_api::pfnGetShapeInference&        getSifResult)
{
    fnHandle fn = GetFunction(h, GET_SHAPE_INFERENCE_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", GET_SHAPE_INFERENCE_ENTRY_POINT_NAME);
        return false;
    }
    getSifResult = (tpc_lib_api::pfnGetShapeInference)fn;

    return loadFunctions(h, instantiate, getLayout, getManipulation, getGuids);
}

bool KernelDB::loadFunctions(libHandle                                 h,
                             tpc_lib_api::pfnInstantiateTpcKernel&     instantiate,
                             tpc_lib_api::pfnGetSupportedDataLayout&   getLayout,
                             tpc_lib_api::pfnGetSuggestedManipulation& getManipulation,
                             tpc_lib_api::pfnGetKernelGuids&           getGuids)
{
    fnHandle fn = GetFunction(h, KERNEL_INSTANTIATION_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", KERNEL_INSTANTIATION_ENTRY_POINT_NAME);
        return false;
    }
    instantiate = (tpc_lib_api::pfnInstantiateTpcKernel)fn;

    fn = GetFunction(h, KERNEL_GUIDS_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", KERNEL_GUIDS_ENTRY_POINT_NAME);
        return false;
    }
    getGuids = (tpc_lib_api::pfnGetKernelGuids)fn;

    fn = GetFunction(h, GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_WARN(KERNEL_DB, "No Entry point {}", GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME);
    }
    getLayout = (tpc_lib_api::pfnGetSupportedDataLayout)fn;

    fn = GetFunction(h, GET_SUGGESTED_MANIPULATION_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_WARN(KERNEL_DB, "No Entry point {}", GET_SUGGESTED_MANIPULATION_ENTRY_POINT_NAME);
    }
    getManipulation = (tpc_lib_api::pfnGetSuggestedManipulation)fn;
    return true;
}

bool KernelDB::loadComplexGuidFunctions(libHandle                                   h,
                                        tpc_lib_api::pfnGetSupportedDataLayout&     getLayout,
                                        tpc_lib_api::pfnGetFunctionalComplexGuids&  getFunctionalNames,
                                        tpc_lib_api::pfnGetPerformanceComplexGuids& getPerformanceNames,
                                        tpc_lib_api::pfnGetShapeInference&          getSifResult)
{
    LOG_DEBUG(KERNEL_DB, "Loading complex guid related entry points");

    fnHandle fn = GetFunction(h, FUNCTIONAL_COMPLEX_GUID_NAMES_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", FUNCTIONAL_COMPLEX_GUID_NAMES_ENTRY_POINT_NAME);
        return false;
    }
    getFunctionalNames = (tpc_lib_api::pfnGetFunctionalComplexGuids)fn;

    fn = GetFunction(h, PERFORMANCE_COMPLEX_GUID_NAMES_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", PERFORMANCE_COMPLEX_GUID_NAMES_ENTRY_POINT_NAME);
        return false;
    }
    getPerformanceNames = (tpc_lib_api::pfnGetPerformanceComplexGuids)fn;

    fn = GetFunction(h, GET_SHAPE_INFERENCE_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", GET_SHAPE_INFERENCE_ENTRY_POINT_NAME);
        return false;
    }
    getSifResult = (tpc_lib_api::pfnGetShapeInference)fn;

    fn = GetFunction(h, GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME);
    if (fn == nullptr)
    {
        LOG_ERR(KERNEL_DB, "No Entry point {}", GET_SUPPORTED_DATA_LAYOUTS_ENTRY_POINT_NAME);
        return false;
    }
    getLayout = (tpc_lib_api::pfnGetSupportedDataLayout)fn;

    return true;
}

tpc_lib_api::GlueCodeReturn KernelDB::GetKernelInstance(tpc_lib_api::HabanaKernelParams*        params,
                                                        tpc_lib_api::HabanaKernelInstantiation* instance,
                                                        const StringWithHash&                   guidAndHash) const
{
    HB_ASSERT_PTR(params);
    HB_ASSERT_PTR(instance);

    auto kernelMetaPtr = getKernelMeta(guidAndHash, params->deviceId);
    if (kernelMetaPtr == nullptr)
    {
        LOG_DEBUG(KERNEL_DB, "No instance for kernel {}", guidAndHash.getKey());
        return tpc_lib_api::GLUE_FAILED;
    }
    auto ret = kernelMetaPtr->getInstance(params, instance);
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_DEBUG(KERNEL_DB, "{} Glue code GetKernelInstance returned {}", guidAndHash.getKey(), parseReturnValue(ret));
    }
    return ret;
}

tpc_lib_api::GlueCodeReturn
KernelDB::GetSuggestedTensorManipulation(const tpc_lib_api::HabanaKernelParams*     params,
                                         tpc_lib_api::TensorManipulationSuggestion* suggestion,
                                         const StringWithHash&                      guidAndHash) const
{
    HB_ASSERT_PTR(params);
    HB_ASSERT_PTR(suggestion);

    auto kernelMetaPtr = getKernelMeta(guidAndHash, params->deviceId);
    if (kernelMetaPtr == nullptr)
    {
        LOG_DEBUG(KERNEL_DB, "No instance for kernel {}", guidAndHash.getKey());
        return tpc_lib_api::GLUE_FAILED;
    }
    auto ret = kernelMetaPtr->getSuggestedManipulation(params, suggestion);
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_DEBUG(KERNEL_DB,
                  "{} Glue code GetSuggestedTensorManipulation returned {}",
                  guidAndHash.getKey(),
                  parseReturnValue(ret));
    }
    return ret;
}

tpc_lib_api::GlueCodeReturn KernelDB::GetKernelSupportedDataLayouts(tpc_lib_api::HabanaKernelParams* params,
                                                                    tpc_lib_api::NodeDataLayouts*    supportedLayouts,
                                                                    unsigned*                        layoutCount,
                                                                    const StringWithHash&            guidAndHash) const
{
    HB_ASSERT_PTR(params);
    HB_ASSERT_PTR(layoutCount);

    tpc_lib_api::pfnGetSupportedDataLayout getLayoutPtr = nullptr;
    auto kernelMetaPtr = getComplexGUIDMeta(guidAndHash, params->deviceId);
    if (kernelMetaPtr == nullptr)
    {
        kernelMetaPtr = getKernelMeta(guidAndHash, params->deviceId);
        if (kernelMetaPtr == nullptr)
        {
            // no such kernel
            LOG_WARN(KERNEL_DB, "No GetSupportedLayouts function for kernel {}", guidAndHash.getKey());
            throw NotImplementedException();
        }
    }
    getLayoutPtr = kernelMetaPtr->getLayouts;

    if (getLayoutPtr == nullptr)
    {
        LOG_WARN(KERNEL_DB, "GetSupportedLayouts pointer is null for kernel {}", guidAndHash.getKey());
        throw NotImplementedException();
    }
    auto ret = getLayoutPtr(params, supportedLayouts, layoutCount);
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_WARN(KERNEL_DB, "{} GetSupportedLayouts returned {}", guidAndHash.getKey(), parseReturnValue(ret));
    }
    return ret;
}

const KernelDB::KernelMetadata* KernelDB::getKernelMeta(const StringWithHash& guidAndHash,
                                                        tpc_lib_api::DeviceId deviceId) const
{
    if (auto kernelIter = m_kernelDB[deviceId].find(guidAndHash); kernelIter != m_kernelDB[deviceId].end())
    {
        return &kernelIter->second;
    }
    else if (isFusedKernel(guidAndHash.getKey()))
    {
        return &m_fuserMeta;
    }
    return nullptr;
}

const KernelDB::KernelMetadata* KernelDB::getDynamicShapeKernelMeta(const StringWithHash& guidAndHash,
                                                                    tpc_lib_api::DeviceId deviceId) const
{
    auto kernelMetaPtr = getKernelMeta(guidAndHash, deviceId);
    if (kernelMetaPtr == nullptr || !kernelMetaPtr->isDynamicShape())
    {
        LOG_DEBUG(KERNEL_DB, "kernel {} doesn't exist or does not support dynamic shapes", guidAndHash.getKey());
        return nullptr;
    }
    return kernelMetaPtr;
}

const KernelDB::KernelMetadata* KernelDB::getComplexGUIDMeta(const StringWithHash& guidAndHash,
                                                             tpc_lib_api::DeviceId deviceId) const
{
    if (auto cguidIter = m_complexGuidsDB[deviceId].find(guidAndHash); cguidIter != m_complexGuidsDB[deviceId].end())
    {
        return &cguidIter->second;
    }
    return nullptr;
}

const KernelDB::KernelMetadata* KernelDB::getDynamicShapeCGUIDMeta(const StringWithHash& guidAndHash,
                                                                   tpc_lib_api::DeviceId deviceId) const
{
    auto kernelMetaPtr = getComplexGUIDMeta(guidAndHash, deviceId);
    if (kernelMetaPtr == nullptr || !kernelMetaPtr->isDynamicShape())
    {
        LOG_DEBUG(KERNEL_DB, "complex GUID {} doesn't exist or does not support dynamic shapes", guidAndHash.getKey());
        return nullptr;
    }
    return kernelMetaPtr;
}

uint64_t KernelDB::GetLibraryVersion(tpc_lib_api::DeviceId deviceId, const std::string& kernelName) const
{
    // This function is called during compilation , not during kernel DB init
    uint64_t libVersion = 0;
    StringWithHash guidAndHash(kernelName);

    auto cguidMetaPtr = getComplexGUIDMeta(guidAndHash, deviceId);
    if (cguidMetaPtr && cguidMetaPtr->isDynamicShape())
    {
        libVersion = cguidMetaPtr->libraryVersion;
        if (cguidMetaPtr->getSifResult == nullptr)
        {
            if (auto kernelMetaPtr = getDynamicShapeKernelMeta(guidAndHash, deviceId))
            {
                LOG_DEBUG(KERNEL_DB,
                          "complex guid {} takes its SIF function from regular kernel lib",
                          guidAndHash.getKey());
                libVersion = kernelMetaPtr->libraryVersion;
            }
        }
    }
    else if (auto kernelMetaPtr = getKernelMeta(guidAndHash, deviceId); kernelMetaPtr != nullptr)
    {
        libVersion = kernelMetaPtr->libraryVersion;
    }
    else if (cguidMetaPtr)
    {
        libVersion = cguidMetaPtr->libraryVersion;
    }
    else
    {
        // no such kernel
        LOG_ERR(KERNEL_DB, "kernel {} doesn't exist, can't get its lib version", guidAndHash.getKey());
        throw NotImplementedException();
    }
    return libVersion;
}

bool KernelDB::GetKernelShapeInferenceFunctionID(tpc_lib_api::DeviceId                  deviceId,
                                                 const std::string&                     kernelName,
                                                 tpc_lib_api::UniqueShapeInferenceHash* sifId) const
{
    HB_ASSERT_PTR(sifId);
    StringWithHash guidAndHash(kernelName);

    const KernelDB::KernelMetadata* kernelMetaPtr = getDynamicShapeCGUIDMeta(kernelName, deviceId);
    if (kernelMetaPtr == nullptr || kernelMetaPtr->guidInfo.nameHash.hashValue == 0)
    {
        // complex guid with dynamic shape fallback - try to get the complex guid function id from simple kernel DB
        LOG_DEBUG(KERNEL_DB, "Trying to get SIF id for complex guid {} from simple kernel DB", guidAndHash.getKey());
        kernelMetaPtr = getDynamicShapeKernelMeta(kernelName, deviceId);
        if (kernelMetaPtr == nullptr || kernelMetaPtr->guidInfo.nameHash.hashValue == 0)
        {
            return false;
        }
    }

    *sifId = kernelMetaPtr->guidInfo.nameHash;

    return true;
}

tpc_lib_api::GlueCodeReturn KernelDB::RunShapeInferenceFunction(tpc_lib_api::DeviceId              deviceId,
                                                                const std::string&                 kernelName,
                                                                tpc_lib_api::ShapeInferenceParams* params,
                                                                tpc_lib_api::ShapeInferenceOutput* outputs) const
{
    HB_ASSERT_PTR(params);
    HB_ASSERT_PTR(outputs);

    StringWithHash        guidAndHash(kernelName);
    const KernelMetadata* kernelMetaPtr = getDynamicShapeCGUIDMeta(kernelName, deviceId);
    if (kernelMetaPtr == nullptr || kernelMetaPtr->getSifResult == nullptr)
    {
        // complex guid with dynamic shape fallback - try to get the complex guid SIF from simple kernel DB
        LOG_DEBUG(KERNEL_DB, "Trying to get SIF for complex guid {} from simple kernel DB", guidAndHash.getKey());
        kernelMetaPtr = getDynamicShapeKernelMeta(guidAndHash, deviceId);
        if (kernelMetaPtr == nullptr || kernelMetaPtr->getSifResult == nullptr)
        {
            return tpc_lib_api::GLUE_SIF_NULL_PTR;
        }
    }

    params->guid  = kernelMetaPtr->guidInfo;  // TODO: remove this when [SW-159125] is done
    params->pGuid = &const_cast<KernelMetadata*>(kernelMetaPtr)->guidInfo;

    tpc_lib_api::GlueCodeReturn ret = kernelMetaPtr->getSifResult(deviceId, params, outputs);
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_WARN(KERNEL_DB, "{} RunShapeInferenceFunction returned {}", guidAndHash.getKey(), parseReturnValue(ret));
    }

    return ret;
}

bool KernelDB::isFusedGUID(std::string_view guid)
{
    return guid.find("fused_kernel") != std::string::npos;
}

bool KernelDB::isKernelExist(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const
{
    return isPerfLibKernel(guidAndHash, deviceId) || isSupportedComplexGuid(guidAndHash, deviceId);
}

bool KernelDB::isPerfLibKernel(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const
{
    return getKernelMeta(guidAndHash, deviceId) != nullptr;
}

bool KernelDB::isSupportedComplexGuid(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const
{
    return isSupportedFunctionalComplexGuid(guidAndHash, deviceId) ||
           isSupportedPerformanceComplexGuid(guidAndHash, deviceId);
}

bool KernelDB::isSupportedDynamicShapeComplexGuid(const StringWithHash& guidAndHash,
                                                  tpc_lib_api::DeviceId deviceId) const
{
    return getDynamicShapeCGUIDMeta(guidAndHash, deviceId) != nullptr;
}

bool KernelDB::isSupportedFunctionalComplexGuid(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const
{
    return m_functionalComplexGuids[deviceId].find(guidAndHash) != m_functionalComplexGuids[deviceId].end();
}

bool KernelDB::isSupportedPerformanceComplexGuid(const StringWithHash& guidAndHash,
                                                 tpc_lib_api::DeviceId deviceId) const
{
    return m_performanceComplexGuids[deviceId].find(guidAndHash) != m_performanceComplexGuids[deviceId].end();
}

bool KernelDB::isDynamicShapeKernel(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const
{
    if (!isKernelExist(guidAndHash, deviceId))
    {
        return false;
    }
    const auto* kernelMeta = getKernelMeta(guidAndHash, deviceId);

    return (kernelMeta != nullptr && kernelMeta->isDynamicShape()) ||
           isSupportedDynamicShapeComplexGuid(guidAndHash, deviceId);
}

void KernelDB::registerSifFromDb(const KernelPerDeviceDB& db,
                                 int                      device,
                                 ShapeFuncRegistry&       registry,
                                 ShapeFuncOrigin          origin) const
{
    for (auto& kernel : db[device])
    {
        if (!kernel.second.isDynamicShape()) continue;

        const std::string&                      kernelName = kernel.first.getKey();
        const uint64_t                          funcId     = kernel.second.guidInfo.nameHash.hashValue;
        const tpc_lib_api::pfnGetShapeInference sif        = kernel.second.getSifResult;
        const uint64_t                          version    = kernel.second.libraryVersion;

        if (sif != nullptr)
        {
            sm_function_id_t sifID;
            sifID.sm_funcid  = funcId;
            sifID.sm_tableid = origin;
            registry.registerSIF(sifID, (sif_t)sif, kernelName, version);
        }
    }
}

void KernelDB::registerSif() const
{
    ShapeFuncRegistry& sfr = ShapeFuncRegistry::instance();  // need it here, in case we don't enter the loop
    UNUSED(sfr);

    LOG_DEBUG_T(KERNEL_DB, "Registering CGUIDs and TPC kernels SIF");
    for (int device = tpc_lib_api::DEVICE_ID_GAUDI; device < tpc_lib_api::DEVICE_ID_MAX; ++device)
    {
        registerSifFromDb(m_kernelDB, device, sfr, ShapeFuncOrigin::LIB_ID_FIRST_GLUE_CODE_SIF);
        registerSifFromDb(m_complexGuidsDB, device, sfr, ShapeFuncOrigin::LIB_ID_COMPLEX_GUID_SIF);
    }
}

uint64_t KernelDB::getKernelHashByName(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId)
{
    auto kernelMetaPtr = getKernelMeta(guidAndHash, deviceId);
    if (kernelMetaPtr == nullptr)
    {
        LOG_INFO(KERNEL_DB, "Failed to find hash value for guid: {}", guidAndHash.getKey());
        return 0;
    }
    return kernelMetaPtr->guidInfo.nameHash.Value;
}
