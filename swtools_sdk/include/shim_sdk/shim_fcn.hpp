/*****************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 * Authors:
 * Yohai Gevim <ygevim@habana.ai>
 ******************************************************************************
 */

#include <dlfcn.h>
#include <stdint.h>
#include <shim_json_types.hpp>

#include "common/shim_typedefs.h"

#ifndef SHIM_FCN_H
#define SHIM_FCN_H

typedef void *LibHandle;

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef uint32_t (*PFN_ShimPluginLoad)(void); // returns a mask of supported API

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef void (*PFN_ShimPluginUnload)(void);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef ShimFunctions (*PFN_ShimInit)(ShimFunctions);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef void (*PFN_ShimFini)(void);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef const char **(*PFN_ShimPluginGetSchemeNames)(void);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef void (*PFN_ShimPluginSetSchemeValues)(const std::string &values);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef bool (*PFN_ShimPluginGetScheme)(const char *schemeName, json *scheme);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef const char **(*PFN_ShimPluginsGetPluginNames)(void);

//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef const char* (*PFN_ShimGetVersion)();


//!
/*!
***************************************************************************************************
*   @brief TODO: Document
***************************************************************************************************
*/
typedef void (*PFN_ShimTerminate)(void);

typedef std::string (*PFN_ShimSubscribe)(const std::string& msg, void* pvtData);

#define GET_PLUGINS_NAMES "GetPluginsNames"

inline std::string PLUGIN_FUNC(const std::string &plugin, const char *func)
{
    return plugin + "_" + func;
}

#define API_BIT(x) (1u << (x))

#endif // SHIM_FCN_H
