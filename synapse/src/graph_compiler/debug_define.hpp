#pragma once

// Debug features flags
//---------------------
// #define ENABLE_GAUDI_RECIPE_PACKETS_INSPECTION

// #define ENABLE_PRE_SUBMISSION_CS_INSPECTION

// We will want this validation be part of the operational code, by default
#define RECIPE_CACHE_OVERRUN_BY_USER_MEMCPY_VALIDATION

// #define ENABLE_DATA_CHUNKS_USAGE_VALIDATION

// #define ENABLE_PRINT_LDMA_PARAMS


// Debug features booleans
//------------------------
// TODO - handle different device-types
#ifdef ENABLE_GAUDI_RECIPE_PACKETS_INSPECTION
static const bool g_inspectRecipePackets = true;
#else
static const bool g_inspectRecipePackets = false;
#endif

#ifdef ENABLE_PRE_SUBMISSION_CS_INSPECTION
static const bool g_preSubmissionCsInspection = true;
#else
static const bool g_preSubmissionCsInspection = false;
#endif

#ifdef RECIPE_CACHE_OVERRUN_BY_USER_MEMCPY_VALIDATION
static const bool g_recipeCacheOverrunValidation = true;
#else
static const bool g_recipeCacheOverrunValidation = false;
#endif

#ifdef ENABLE_DATA_CHUNKS_USAGE_VALIDATION
static const bool g_validateDataChunksUsage = true;
#else
static const bool g_validateDataChunksUsage = false;
#endif

#ifdef ENABLE_PRINT_LDMA_PARAMS
static const bool g_printLdmaMemcpyParams = true;
#else
static const bool g_printLdmaMemcpyParams = false;
#endif
