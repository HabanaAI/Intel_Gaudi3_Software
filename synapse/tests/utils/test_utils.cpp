#include "test_utils.h"
#include <algorithm>
#include <cassert>
#include "node_factory.h"

#include "synapse_common_types.h"

using namespace gc;

std::string deviceTypeToString(const synDeviceType& deviceType)
{
    switch (deviceType)
    {
        case synDeviceGaudi:
            return "Gaudi";
        case synDeviceGaudi2:
            return "Gaudi2";
        case synDeviceGaudi3:
            return "Gaudi3";
        case synDeviceEmulator:
            return "Emulator";
        default:
            HB_ASSERT(0, "Invalid device type: {}", deviceType);
            return "";
    }
}

template<>
inline void fillWithRandomNeg<float>(float* p, uint64_t size)
{
    for (unsigned i = 0; i < size; ++i)
    {
        float frac = (float)rand() / (float)RAND_MAX; //in [0, 1]
        p[i] = frac - 0.5f; //in [-0.5, 0.5]
    }
}

void fillWithRandom(void* p, uint64_t size, synDataType dataType)
{
    switch (dataType)
    {
        case syn_type_int8:
            fillWithRandom(reinterpret_cast<int8_t*>(p), size, std::pair(-10, 10));
            break;
        case syn_type_uint8:
            fillWithRandom(reinterpret_cast<uint8_t*>(p), size, std::pair(0, 10));
            break;
        case syn_type_int16:
            fillWithRandom(reinterpret_cast<int16_t *>(p), size, std::pair(-10, 10));
            break;
        case syn_type_uint16:
            fillWithRandom(reinterpret_cast<uint16_t*>(p), size, std::pair(0, 10));
            break;
        case syn_type_int32:
            fillWithRandom(reinterpret_cast<int32_t*>(p), size, std::pair(-10, 10));
            break;
        case syn_type_uint32:
            fillWithRandom(reinterpret_cast<uint32_t*>(p), size, std::pair(0, 10));
            break;
        case syn_type_float:
            fillWithRandom(reinterpret_cast<float*>(p), size, std::pair(-10, 10));
            break;
        case syn_type_fp16:
            fillWithRandom(reinterpret_cast<fp16_t*>(p), size, std::pair(float(-10.0), float(10.0)));
            break;
        case syn_type_bf16:
            fillWithRandom(reinterpret_cast<bfloat16*>(p), size, std::pair(float(-10), float(10)));
            break;
        default:
            HB_ASSERT(false, "Unsupported data type for fill with random");
    }
}

bool float_eq(float a, float b, float eps)
{
    float err = std::abs(a - b);
    return (err / std::max(std::abs(a), eps) <= eps);
}

void init_3d_data(float* p, unsigned W, unsigned H, unsigned D)
{
    for (unsigned z = 0; z < D; ++z)
    {
        for (unsigned y = 0; y < H; ++y)
        {
            for (unsigned x = 0; x < W; ++x)
            {
                float val = (float)z * 10000.f + (float)y * 100.f + (float)x;
                *p++ = val;
            }
        }
    }
}

std::vector<char*> AllocateMemory(std::vector<uint64_t> vec)
{
    std::vector<char*> allocations;
    for (auto itr : vec)
    {
        allocations.push_back(new char[itr]);
    }
    return allocations;
}

void FreeMemory(std::vector<char*> vec)
{
    for (auto itr : vec)
    {
        delete [] itr;
    }
    vec.clear();
}

void* generateValuesArray(unsigned int array_size,
                          synDataType array_type,
                          const std::array<int,2>& range)
{
    switch (array_type)
    {
    case syn_type_fixed:
        return generateValuesArray<int8_t>(array_size, range);
    case syn_type_uint8:
        return generateValuesArray<uint8_t>(array_size, range);
    case syn_type_int16:
        return generateValuesArray<int16_t>(array_size, range);
    case syn_type_single:
    case syn_type_int32:
        return generateValuesArray<int32_t>(array_size, range);
    default:
        assert("Unknown tensor type" == nullptr);
        return nullptr;
    }
}

NodePtr getConvNodeWithGoyaLayouts(const TensorPtr& IFM, const TensorPtr& weights, const TensorPtr& bias, const TensorPtr& OFM,
                                   const synConvolutionParams& params, const std::string& name)
{
    Node::NodeProperties p;
    if (weights)
    {
        weights->setAsWeights();
    }
    if (bias)
    {
        bias->setAsBias();
    }
    p.inputLayouts = {Layout("CWHN"), Layout("KCSR"), Layout(), Layout("CWHN")};
    p.outputLayouts = {Layout("CWHN")};
    NodePtr node = NodeFactory::createNode({IFM, weights, bias, nullptr}, {OFM}, &params, NodeFactory::convolutionNodeTypeName, name);
    node->setInputLayouts(p.inputLayouts);
    node->setOutputLayouts(p.outputLayouts);
    return node;
}

NodePtr getConvPlusNodeWithGoyaLayouts(const TensorPtr& IFM, const TensorPtr& weights, const TensorPtr& bias, const TensorPtr& OFM,
                                       const TensorPtr& cin, const synConvolutionParams& params, const std::string& name)
{
    Node::NodeProperties p;
    p.inputLayouts = {Layout("CWHN"), Layout("KCSR"), Layout(), Layout("CWHN")};
    p.outputLayouts = {Layout("CWHN")};
    NodePtr node = NodeFactory::createNode({IFM, weights, bias, cin}, {OFM}, &params, NodeFactory::convolutionNodeTypeName, name);
    node->setInputLayouts(p.inputLayouts);
    node->setOutputLayouts(p.outputLayouts);
    return node;
}

uint32_t getElementSizeInBytes(synDataType type)
{
    switch (type)
    {
        case syn_type_int8:
        case syn_type_uint8:
        case syn_type_fp8_152:
        case syn_type_fp8_143:
            return sizeof(char);
        case syn_type_bf16:
        case syn_type_int16:
        case syn_type_uint16:
        case syn_type_fp16:
            return sizeof(int16_t);
        case syn_type_float:
            return sizeof(float);
        case syn_type_int32:
        case syn_type_uint32:
            return sizeof(int32_t);
        case syn_type_int64:
        case syn_type_uint64:
            return sizeof(int64_t);
        default:
            assert(0 && "Invalid argument");
            return 0;
    }
}

void fillWithRandomRoundedFloats(float* p, uint64_t size, std::pair<float, float> minMax)
{
    if (minMax.first > minMax.second)
    {
        std::swap(minMax.first, minMax.second);
    }
    std::default_random_engine  generator; // NOLINT(cert-msc32-c,cert-msc51-cpp) - deterministic on purpose
    uniform_distribution<float> distribution(minMax.first, minMax.second);

    for (uint64_t i = 0; i < size; ++i)
    {
        do
        {
            p[i] = std::roundf(distribution(generator)); // rounds to nearest int, even away from zero
        } while (!isValidValue(p[i]));
    }
}

float dequantize(int8_t value, double zp, double scale)
{
    return (value - zp) * scale;
}

/* dequantize (real = s_in * (value - z_in)) and quantize (quant = real / s_out + z_out) */
float dequantizeQuantize(float value, double scaleIn, double scaleOut, double zpIn, double zpOut)
{

    value *= scaleIn / scaleOut;
    value -= (scaleIn * zpIn - scaleOut * zpOut) / scaleOut;
    return value;
}

float roundHalfAwayFromZero(float value)
{
    int sign = (float(0) < value) - (value < float(0));
    float absN = fabs(value);
    return sign * floor(absN + 0.5);
}

float clamp(synDataType dType, float value)
{
    float minClip = 0, maxClip = 0;
    switch (dType)
    {
        case syn_type_int4:
            minClip = INT4_MIN_VAL;
            maxClip = INT4_MAX_VAL;
            break;
        case syn_type_uint4:
            minClip = UINT4_MIN_VAL;
            maxClip = UINT4_MAX_VAL;
            break;
        case syn_type_int8:
            minClip = std::numeric_limits<int8_t>::min();
            maxClip = std::numeric_limits<int8_t>::max();
            break;
        case syn_type_uint8:
            minClip = std::numeric_limits<uint8_t>::min();
            maxClip = std::numeric_limits<uint8_t>::max();
            break;
        default:
            assert(0 && "Unsupported data type");
            return 0;
    }
    return clip<float>(value, minClip, maxClip);
}

std::array<unsigned, (uint32_t)synapse::LogManager::LogType::LOG_MAX> testsEnableAllSynLogs(int newLevel)
{
    std::array<unsigned, (uint32_t)synapse::LogManager::LogType::LOG_MAX> current;
    current.fill(std::numeric_limits<uint32_t>::max());

    for (uint32_t i = 0; i < (uint32_t)synapse::LogManager::LogType::LOG_MAX; i++)
    {
        synapse::LogManager::LogType logType = (synapse::LogManager::LogType)i;
        std::string_view             logName = synapse::LogManager::instance().getLogTypeString(logType);
        if (logName.find("SYN_") == 0)
        {
            current[i] = synapse::LogManager::get_log_level(logType);
            synapse::LogManager::instance().set_log_level(logType, SPDLOG_LEVEL_TRACE);
        }
    }
    return current;
}

void testRecoverSynLogsLevel(const std::array<unsigned, (uint32_t)synapse::LogManager::LogType::LOG_MAX>& prev)
{
    for (uint32_t i = 0; i < (uint32_t)synapse::LogManager::LogType::LOG_MAX; i++)
    {
        if (prev[i] != std::numeric_limits<uint32_t>::max())
        {
            synapse::LogManager::LogType logType = (synapse::LogManager::LogType)i;
            synapse::LogManager::instance().set_log_level(logType, prev[i]);
        }
    }
}

void setTensorAsPersistent(TensorPtr& tensor, unsigned tensorsCount)
{
    {
        static synMemoryDescriptor memDesc(true);
        tensor->setMemoryDescriptor(memDesc);
        tensor->setDramOffset(0x1000);
        tensor->setMemorySectionID(MEMORY_ID_FOR_FIRST_PERSISTENT_TENSOR + tensorsCount);
    }
}
uint64_t getNumberOfElements(const TSize* sizes, unsigned dims)
{
    uint64_t elements = dims ? 1 : 0;

    for (int i = 0; i < dims; i++)
    {
        elements *= sizes[i];
    }

    return elements;
}

uint64_t getNumberOfElements(const unsigned* sizes, unsigned dims)
{
    uint64_t elements = dims ? 1 : 0;

    for (int i = 0; i < dims; i++)
    {
        elements *= sizes[i];
    }

    return elements;
}

void randomBufferValues(MemInitType initSelect,
                        synDataType type,
                        uint64_t    size,
                        void*       output,
                        bool        isDefaultGenerator)
{
    std::default_random_engine defaultGenerator = std::default_random_engine();  // NOLINT(cert-msc32-c,cert-msc51-cpp) - deterministic on purpose;

    std::random_device rd;
    std::default_random_engine randomGenerator(rd());

    std::default_random_engine* pGenerator;
    if (isDefaultGenerator)
    {
        pGenerator = &defaultGenerator;
    }
    else
    {
        pGenerator = &randomGenerator;
    }

    std::default_random_engine generator = *pGenerator;

    static const float                   maxRand     = 2.0f;   // divide by 10 to avoid overflow
    static const float                   minRand     = -2.0f;  // divide by 10 to avoid overflow
    static const std::pair<float, float> randomRange = {minRand, maxRand};
    static const std::pair<float, float> nonNegRange = {0, maxRand};

    static const float                   maxRandInt     = 120.0f;
    static const float                   minRandInt     = -120.0f;
    static const std::pair<float, float> randomRangeInt = {minRandInt, maxRandInt};
    static const std::pair<float, float> nonNegRangeInt = {0, maxRandInt};

    assert(initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE || initSelect == MEM_INIT_RANDOM_POSITIVE);
    std::pair<float, float> range = initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE ? randomRange : nonNegRange;
    std::pair<float, float> rangeInt = initSelect == MEM_INIT_RANDOM_WITH_NEGATIVE ? randomRangeInt : nonNegRangeInt;
    switch (type)
    {
        case syn_type_float:
            fillWithRandom<float>(generator, (float*)output, size, range);
            break;
        case syn_type_bf16:
            fillWithRandom(generator, (bfloat16*)output, size, range);
            break;
        case syn_type_fp16:
            fillWithRandom(generator, (fp16_t*)output, size, range);
            break;
        case syn_type_int8:
            fillWithRandom(generator, (int8_t*)output, size, rangeInt);
            break;
        case syn_type_uint8:
            fillWithRandom(generator, (uint8_t*)output, size, rangeInt);
            break;
        case syn_type_int16:
            fillWithRandom(generator, (int16_t*)output, size, rangeInt);
            break;
        case syn_type_uint16:
            fillWithRandom(generator, (uint16_t*)output, size, rangeInt);
            break;
        case syn_type_int32:
            fillWithRandom(generator, (int32_t*)output, size, rangeInt);
            break;
        case syn_type_uint32:
            fillWithRandom(generator, (uint32_t*)output, size, rangeInt);
            break;
        case syn_type_int64:
            fillWithRandom(generator, (int64_t*)output, size, rangeInt);
            break;
        case syn_type_uint64:
            fillWithRandom(generator, (uint64_t*)output, size, rangeInt);
            break;
        case syn_type_fp8_152:
            fillWithRandom(generator, (fp8_152_t*)output, size, range);
            break;
        default:
            LOG_ERR(SYN_TEST, "randomBufferValues: Unsupported data type for Gaudi test {}", type);
            assert(0 && "Unsupported data type for Gaudi test");
    }
}
