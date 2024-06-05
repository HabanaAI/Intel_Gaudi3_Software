#ifndef _KERNEL_DB_H_
#define _KERNEL_DB_H_

#include "tpc_kernel_lib_interface.h"
#include "tpc_kernel_lib_interface_private.h"

#include "smf/shape_func_registry.h"
#include "utils.h"
#include "types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <list>
#include <map>
#include <set>
#include <string_view>
#include <string>
#include <unordered_map>

class NotImplementedException : public std::exception
{
};

class KernelDB
{
public:
    static const char* KERNEL_LIB_ENV_VAR;
    static constexpr uint32_t MIN_SUPPORTED_VERSION_SMALL_VLM = 16;

    static KernelDB& instance();

    enum class Feature : unsigned int
    {
        SMALL_VLM = 1 << 0, // 1
        // Add more features as needed
    };

private:
    // The only 2 classes allowed to initialize kernelDB
    // Initialization should be on synInitialized and for unittests for all devices
    friend class synSingleton;
    friend class GraphOptimizerTest;

    // Initialize kernels by device type
    // To initialize all, pass DEVICE_ID_MAX
    void init(tpc_lib_api::DeviceId deviceId);

    void loadKernelsFromPath(const char*           path,
                             tpc_lib_api::DeviceId deviceId,
                             ShapeFuncOrigin       shapeFuncLib = LIB_ID_FIRST_GLUE_CODE_SIF,
                             uint8_t               libIndex     = 0);

    void loadFuserKernelsFromPath(const char* path);

    void initComplexGuidLib(tpc_lib_api::DeviceId deviceId);

    void clear();

public:

    class FeatureSupport
    {
    public:
        FeatureSupport() : m_supportedFeatures{0} {}

        void addSupportedFeature(tpc_lib_api::DeviceId deviceId, Feature feature)
        {
            // Set the corresponding bit to 1 to indicate support
            m_supportedFeatures[deviceId] |= static_cast<unsigned int>(feature);
        }

        bool isFeatureSupported(tpc_lib_api::DeviceId deviceId, Feature feature) const
        {
            // Check if the corresponding bit is set to 1
            return (m_supportedFeatures[deviceId] & static_cast<unsigned int>(feature)) != 0;
        }
    private:
        std::array<unsigned int,tpc_lib_api::DEVICE_ID_MAX> m_supportedFeatures;
    };

    bool initialized() const { return !m_initializedIds.empty(); }

    tpc_lib_api::GlueCodeReturn GetKernelInstance(tpc_lib_api::HabanaKernelParams*        params,
                                                  tpc_lib_api::HabanaKernelInstantiation* instance,
                                                  const StringWithHash&                   guidAndHash) const;

    tpc_lib_api::GlueCodeReturn GetKernelSupportedDataLayouts(tpc_lib_api::HabanaKernelParams* params,
                                                              tpc_lib_api::NodeDataLayouts*    supportedLayouts,
                                                              unsigned*                        layoutCount,
                                                              const StringWithHash&            guidAndHash) const;

    tpc_lib_api::GlueCodeReturn GetSuggestedTensorManipulation(const tpc_lib_api::HabanaKernelParams*     params,
                                                               tpc_lib_api::TensorManipulationSuggestion* suggestion,
                                                               const StringWithHash& guidAndHash) const;

    uint64_t GetLibraryVersion(tpc_lib_api::DeviceId deviceId, const std::string& kernelName) const;

    bool GetKernelShapeInferenceFunctionID(tpc_lib_api::DeviceId                  deviceId,
                                           const std::string&                     kernelName,
                                           tpc_lib_api::UniqueShapeInferenceHash* sifId) const;

    tpc_lib_api::GlueCodeReturn RunShapeInferenceFunction(tpc_lib_api::DeviceId              deviceId,
                                                          const std::string&                 kernelName,
                                                          tpc_lib_api::ShapeInferenceParams* params,
                                                          tpc_lib_api::ShapeInferenceOutput* outputs) const;

    static bool isFusedGUID(std::string_view guid);

    bool isKernelExist(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    bool isPerfLibKernel(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    bool isDynamicShapeKernel(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    bool isSupportedComplexGuid(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    bool isSupportedDynamicShapeComplexGuid(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    bool isSupportedFunctionalComplexGuid(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    bool isSupportedPerformanceComplexGuid(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    static std::string_view parseReturnValue(tpc_lib_api::GlueCodeReturn ret);

    void registerSif() const;

    const std::unordered_map<std::string, uint32_t>& GetLibraryVersions() const;

    uint64_t getKernelHashByName(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId);
    const FeatureSupport& getSupportedFeatures() const { return m_featureSupport;};

private:
    bool getLibraryVersion(libHandle h, uint64_t& version);

    bool loadKernels(libHandle             h,
                     tpc_lib_api::DeviceId deviceId,
                     uint64_t              libVersion,
                     ShapeFuncOrigin       shapeFuncLib = LIB_ID_FIRST_GLUE_CODE_SIF,
                     uint8_t               libIndex     = 0);
    bool loadFunctions(libHandle                                 h,
                       tpc_lib_api::pfnInstantiateTpcKernel&     instantiate,
                       tpc_lib_api::pfnGetSupportedDataLayout&   getLayout,
                       tpc_lib_api::pfnGetSuggestedManipulation& getManipulation,
                       tpc_lib_api::pfnGetKernelGuids&           getGuids);
    bool loadFunctions(libHandle                                 h,
                       tpc_lib_api::pfnInstantiateTpcKernel&     instantiate,
                       tpc_lib_api::pfnGetSupportedDataLayout&   getLayout,
                       tpc_lib_api::pfnGetSuggestedManipulation& getManipulation,
                       tpc_lib_api::pfnGetKernelGuids&           getGuids,
                       tpc_lib_api::pfnGetShapeInference&        getShapeInference);

    bool loadComplexGuidFunctions(libHandle                                   h,
                                  tpc_lib_api::pfnGetSupportedDataLayout&     getLayout,
                                  tpc_lib_api::pfnGetFunctionalComplexGuids&  getFunctionalNames,
                                  tpc_lib_api::pfnGetPerformanceComplexGuids& getPerformanceNames,
                                  tpc_lib_api::pfnGetShapeInference&          getShapeInference);

    template<typename GetGuidFunc>
    bool
    loadKernelGuids(GetGuidFunc getNames, tpc_lib_api::DeviceId deviceId, std::vector<tpc_lib_api::GuidInfo>& guids);

    bool loadSupportedComplexGuids(libHandle h, tpc_lib_api::DeviceId deviceId, uint64_t complexGuidLibVersion);

    static std::string_view deviceIdToString(tpc_lib_api::DeviceId deviceId);
    KernelDB();
    void loadKernelsFromEnv(const char* env, tpc_lib_api::DeviceId deviceId);
    void _init(tpc_lib_api::DeviceId deviceId, bool useDefaultLibName = false);

    struct KernelMetadata
    {
        uint64_t                                 libraryVersion = 0;
        tpc_lib_api::GuidInfo                    guidInfo;
        tpc_lib_api::pfnGetSupportedDataLayout   getLayouts               = nullptr;
        tpc_lib_api::pfnGetSuggestedManipulation getSuggestedManipulation = nullptr;
        tpc_lib_api::pfnInstantiateTpcKernel     getInstance              = nullptr;
        tpc_lib_api::pfnGetShapeInference        getSifResult             = nullptr;
        inline bool isDynamicShape() const { return guidInfo.supportsDynamicShapes; }
    };

    const KernelMetadata* getKernelMeta(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;
    const KernelMetadata* getDynamicShapeKernelMeta(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;
    const KernelMetadata* getComplexGUIDMeta(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;
    const KernelMetadata* getDynamicShapeCGUIDMeta(const StringWithHash& guidAndHash, tpc_lib_api::DeviceId deviceId) const;

    using KernelName        = std::string;
    using KernelNameAndHash = StringWithHash;
    using KernelByName      = std::unordered_map<KernelNameAndHash, KernelMetadata>;
    using KernelPerDeviceDB = std::array<KernelByName, tpc_lib_api::DEVICE_ID_MAX>;

    KernelPerDeviceDB    m_kernelDB;
    KernelMetadata       m_fuserMeta;
    std::list<libHandle> m_libHandles;
    // Map between TPC library path to its version
    std::unordered_map<std::string, uint32_t> m_libsVersions;

    // Complex guids DB stores each one with its metadata, while a set contains the guids only.
    using ComplexGuidName              = KernelName;
    using ComplexGuidNameAndHash       = KernelNameAndHash;
    using ComplexGuidNamesSet          = std::unordered_set<ComplexGuidNameAndHash>;
    using ComplexGuidByName            = KernelByName;
    using ComplexGuidPerDeviceDB       = std::array<ComplexGuidByName, tpc_lib_api::DEVICE_ID_MAX>;
    using ComplexGuidNamesPerDeviceSet = std::array<ComplexGuidNamesSet, tpc_lib_api::DEVICE_ID_MAX>;
    // static and dynamic shaped CGUIDs are two distinct groups, while functional and performance aren't necessarily.
    ComplexGuidPerDeviceDB       m_complexGuidsDB;
    ComplexGuidNamesPerDeviceSet m_functionalComplexGuids;
    ComplexGuidNamesPerDeviceSet m_performanceComplexGuids;

    void registerSifFromDb(const KernelPerDeviceDB& db, int device, ShapeFuncRegistry& registry, ShapeFuncOrigin origin) const;

    std::set<tpc_lib_api::DeviceId> m_initializedIds;
    FeatureSupport m_featureSupport;
};

template<typename GetGuidFunc>
bool KernelDB::loadKernelGuids(GetGuidFunc                         getNames,
                               tpc_lib_api::DeviceId               deviceId,
                               std::vector<tpc_lib_api::GuidInfo>& guids)
{
    unsigned                    kernelCount = 0;
    tpc_lib_api::GlueCodeReturn ret         = getNames(deviceId, &kernelCount, nullptr);
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_WARN(KERNEL_DB, "Error loading kernels for {} device", deviceIdToString(deviceId));
        return false;
    }
    else if (kernelCount == 0)
    {
        return true;
    }

    guids.resize(kernelCount);

    ret = getNames(deviceId, &kernelCount, guids.data());
    if (ret != tpc_lib_api::GLUE_SUCCESS)
    {
        LOG_ERR(KERNEL_DB, "Failed getting kernel guids for {} device", deviceIdToString(deviceId));
        return false;
    }

    return true;
}
#endif
