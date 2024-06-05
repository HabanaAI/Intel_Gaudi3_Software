#
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)
#
PROJECT(libarchive C)
#
SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/build/cmake")
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${libarchive_BINARY_DIR}/bin)
endif()
#
# Set the Build type for make based generators.
# You can choose following types:
#   Debug          : Debug build
#   Release        : Release build
#   RelWithDebInfo : Release build with Debug Info
#   MinSizeRel     : Release Min Size build
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
# Set a value type to properly display CMAKE_BUILD_TYPE on GUI if the
# value type is "UNINITIALIZED".
GET_PROPERTY(cached_type CACHE CMAKE_BUILD_TYPE PROPERTY TYPE)
IF("${cached_type}" STREQUAL "UNINITIALIZED")
  SET(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Build Type" FORCE)
ENDIF("${cached_type}" STREQUAL "UNINITIALIZED")
# Check the Build Type.
IF(NOT "${CMAKE_BUILD_TYPE}"
       MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)\$")
  MESSAGE(FATAL_ERROR
          "Unknown keyword for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}\n"
          "Acceptable keywords: Debug,Release,RelWithDebInfo,MinSizeRel")
ENDIF(NOT "${CMAKE_BUILD_TYPE}"
          MATCHES "^(Debug|Release|RelWithDebInfo|MinSizeRel)\$")

# On MacOS, prefer MacPorts libraries to system libraries.
# I haven't come up with a compelling argument for this to be conditional.
list(APPEND CMAKE_PREFIX_PATH /opt/local)
# Enable @rpath in the install name.
# detail in "cmake  --help-policy CMP0042"
SET(CMAKE_MACOSX_RPATH ON)

#
# Version - read from 'version' file.
#
FILE(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/build/version _version)
STRING(REGEX REPLACE
 "^([0-9])[0-9][0-9][0-9][0-9][0-9][0-9][a-z]*$" "\\1" _major ${_version})
STRING(REGEX REPLACE
 "^[0-9]([0-9][0-9][0-9])[0-9][0-9][0-9][a-z]*$" "\\1" _minor ${_version})
STRING(REGEX REPLACE
 "^[0-9][0-9][0-9][0-9]([0-9][0-9][0-9])[a-z]*$" "\\1" _revision ${_version})
STRING(REGEX REPLACE
 "^[0-9][0-9][0-9][0-9][0-9][0-9][0-9]([a-z]*)$" "\\1" _quality ${_version})
SET(_version_number ${_major}${_minor}${_revision})
STRING(REGEX REPLACE "[0]*([^0]*[0-9])$" "\\1" _trimmed_minor ${_minor})
STRING(REGEX REPLACE "[0]*([^0]*[0-9])$" "\\1" _trimmed_revision ${_revision})
#
SET(VERSION                    "${_major}.${_trimmed_minor}.${_trimmed_revision}${_quality}")
SET(BSDCPIO_VERSION_STRING     "${VERSION}")
SET(BSDTAR_VERSION_STRING      "${VERSION}")
SET(BSDCAT_VERSION_STRING      "${VERSION}")
SET(LIBARCHIVE_VERSION_NUMBER  "${_version_number}")
SET(LIBARCHIVE_VERSION_STRING  "${VERSION}")

# INTERFACE_VERSION increments with every release
# libarchive 2.7 == interface version 9 = 2 + 7
# libarchive 2.8 == interface version 10 = 2 + 8
# libarchive 2.9 == interface version 11 = 2 + 9
# libarchive 3.0 == interface version 12
# libarchive 3.1 == interface version 13
math(EXPR INTERFACE_VERSION  "13 + ${_minor}")

# Set SOVERSION == Interface version
# ?? Should there be more here ??
SET(SOVERSION "${INTERFACE_VERSION}")

# Enalbe CMAKE_PUSH_CHECK_STATE() and CMAKE_POP_CHECK_STATE() macros
# saving and restoring the state of the variables.
INCLUDE(CMakePushCheckState)

# Initialize the state of the variables. This initialization is not
# necessary but this shows you what value the variables initially have.
SET(CMAKE_REQUIRED_DEFINITIONS)
SET(CMAKE_REQUIRED_INCLUDES)
SET(CMAKE_REQUIRED_LIBRARIES)
SET(CMAKE_REQUIRED_FLAGS)

# Especially for early development, we want to be a little
# aggressive about diagnosing build problems; this can get
# relaxed somewhat in final shipping versions.
IF (CMAKE_C_COMPILER_ID MATCHES "^GNU$")
  SET(CMAKE_REQUIRED_FLAGS "-Wall -Wformat -Wformat-security")
  #################################################################
  # Set compile flags for all build types.
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wformat -Wformat-security")
  #################################################################
  # Set compile flags for debug build.
  # This is added into CMAKE_C_FLAGS when CMAKE_BUILD_TYPE is "Debug"
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wno-conversion")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Werror")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wunused")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wshadow")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wmissing-prototypes")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-qual")
ENDIF (CMAKE_C_COMPILER_ID MATCHES "^GNU$")
IF (CMAKE_C_COMPILER_ID MATCHES "^Clang$")
  SET(CMAKE_REQUIRED_FLAGS "-Wall -Wformat -Wformat-security")
  #################################################################
  # Set compile flags for all build types.
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wformat -Wformat-security")
  #################################################################
  # Set compile flags for debug build.
  # This is added into CMAKE_C_FLAGS when CMAKE_BUILD_TYPE is "Debug"
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Werror")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wextra")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wunused")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wshadow")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wmissing-prototypes")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wcast-qual")
ENDIF (CMAKE_C_COMPILER_ID MATCHES "^Clang$")
IF (CMAKE_C_COMPILER_ID MATCHES "^XL$")
  SET(CMAKE_C_COMPILER "xlc_r")
  SET(CMAKE_REQUIRED_FLAGS "-qflag=e:e -qformat=sec")
  #################################################################
  # Set compile flags for all build types.
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qflag=e:e -qformat=sec")
  #################################################################
  # Set compile flags for debug build.
  # This is added into CMAKE_C_FLAGS when CMAKE_BUILD_TYPE is "Debug"
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -qhalt=w")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -qflag=w:w")
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -qinfo=pro:use")
ENDIF(CMAKE_C_COMPILER_ID MATCHES "^XL$")
IF (MSVC)
  #################################################################
  # Set compile flags for debug build.
  # This is added into CMAKE_C_FLAGS when CMAKE_BUILD_TYPE is "Debug"
  # Enable level 4 C4061: The enumerate has no associated handler in a switch
  #                       statement.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4061")
  # Enable level 4 C4254: A larger bit field was assigned to a smaller bit
  #                       field.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4254")
  # Enable level 4 C4295: An array was initialized but the last character in
  #                       the array is not a null; accessing the array may
  #                       produce unexpected results.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4295")
  # Enable level 4 C4296: An unsigned variable was used in a comparison
  #                       operation with zero.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4296")
  # Enable level 4 C4389: An operation involved signed and unsigned variables.
  #                       This could result in a loss of data.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4389")
  # Enable level 4 C4505: The given function is local and not referenced in
  #                       the body of the module; therefore, the function is
  #                       dead code.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4505")
  # Enable level 4 C4514: The optimizer removed an inline function that is not
  #                       called.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4514")
  # Enable level 4 C4702: Unreachable code.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4702")
  # Enable level 4 C4706: The test value in a conditional expression was the
  #                       result of an assignment.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /we4706")
  # /WX option is the same as gcc's -Werror option.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /WX")
  # /Oi option enables built-in functions.
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Oi")
  #################################################################
  # Set compile flags for release build.
  SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Oi")
ENDIF (MSVC)

# Enable CTest/CDash support
# include(CTest)

OPTION(ENABLE_ZLIB "Enable the use of the system ZLIB library if found" ON)
OPTION(ENABLE_INSTALL "Enable installing of libraries" OFF)
if (DO_INSTALL)
  SET(ENABLE_INSTALL ON)
endif()

SET(POSIX_REGEX_LIB "AUTO" CACHE STRING "Choose what library should provide POSIX regular expression support")
SET(ENABLE_SAFESEH "AUTO" CACHE STRING "Enable use of /SAFESEH linker flag (MSVC only)")
SET(WINDOWS_VERSION "WIN7" CACHE STRING "Set Windows version to use (Windows only)")


IF(WIN32)
  IF(WINDOWS_VERSION STREQUAL "WIN8")
    SET(NTDDI_VERSION 0x06020000)
    SET(_WIN32_WINNT 0x0602)
    SET(WINVER 0x0602)
  ELSEIF(WINDOWS_VERSION STREQUAL "WIN7")
    SET(NTDDI_VERSION 0x06010000)
    SET(_WIN32_WINNT 0x0601)
    SET(WINVER 0x0601)
  ELSEIF(WINDOWS_VERSION STREQUAL "WS08")
    SET(NTDDI_VERSION 0x06000100)
    SET(_WIN32_WINNT 0x0600)
    SET(WINVER 0x0600)
  ELSEIF(WINDOWS_VERSION STREQUAL "VISTA")
    SET(NTDDI_VERSION 0x06000000)
    SET(_WIN32_WINNT 0x0600)
    SET(WINVER 0x0600)
  ELSEIF(WINDOWS_VERSION STREQUAL "WS03")
    SET(NTDDI_VERSION 0x05020000)
    SET(_WIN32_WINNT 0x0502)
    SET(WINVER 0x0502)
  ELSEIF(WINDOWS_VERSION STREQUAL "WINXP")
    SET(NTDDI_VERSION 0x05010000)
    SET(_WIN32_WINNT 0x0501)
    SET(WINVER 0x0501)
  ELSE(WINDOWS_VERSION STREQUAL "WIN8")
    # Default to Windows Server 2003 API if we don't recognize the specifier
    SET(NTDDI_VERSION 0x05020000)
    SET(_WIN32_WINNT 0x0502)
    SET(WINVER 0x0502)
  ENDIF(WINDOWS_VERSION STREQUAL "WIN8")
ENDIF(WIN32)

IF(MSVC)
  IF(ENABLE_SAFESEH STREQUAL "YES")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SAFESEH")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH")
    SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /SAFESEH")
    SET(ENV{LDFLAGS} "$ENV{LDFLAGS} /SAFESEH")
  ELSEIF(ENABLE_SAFESEH STREQUAL "NO")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SAFESEH:NO")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO")
    SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /SAFESEH:NO")
    SET(ENV{LDFLAGS} "$ENV{LDFLAGS} /SAFESEH:NO")
  ENDIF(ENABLE_SAFESEH STREQUAL "YES")
ENDIF(MSVC)

IF("${CMAKE_C_PLATFORM_ID}" MATCHES "^(HP-UX)$")
  ADD_DEFINITIONS(-D_XOPEN_SOURCE=500) # Ask wchar.h for mbstate_t
ENDIF()

#
INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCSourceRuns)
INCLUDE(CheckFileOffsetBits)
INCLUDE(CheckFuncs)
INCLUDE(CheckHeaderDirent)
INCLUDE(CheckIncludeFile)
INCLUDE(CheckIncludeFiles)
INCLUDE(CheckLibraryExists)
INCLUDE(CheckStructHasMember)
INCLUDE(CheckSymbolExists)
INCLUDE(CheckTypeExists)
INCLUDE(CheckTypeSize)


#
# Find out what macro is needed to use libraries on Windows.
#
MACRO (TRY_MACRO_FOR_LIBRARY INCLUDES LIBRARIES
       TRY_TYPE SAMPLE_SOURCE MACRO_LIST)
  IF(WIN32 AND NOT CYGWIN)
    CMAKE_PUSH_CHECK_STATE()	# Save the state of the variables
    SET(CMAKE_REQUIRED_INCLUDES ${INCLUDES})
    SET(CMAKE_REQUIRED_LIBRARIES ${LIBRARIES})
    FOREACH(VAR ${MACRO_LIST})
      # Clear ${VAR} from CACHE If the libraries which ${VAR} was
      # checked with are changed.
      SET(VAR_WITH_LIB "${VAR}_WITH_LIB")
      GET_PROPERTY(PREV_VAR_WITH_LIB VARIABLE PROPERTY ${VAR_WITH_LIB})
      IF(NOT "${PREV_VAR_WITH_LIB}" STREQUAL "${LIBRARIES}")
        UNSET(${VAR} CACHE)
      ENDIF(NOT "${PREV_VAR_WITH_LIB}" STREQUAL "${LIBRARIES}")
      # Check if the library can be used with the macro.
      IF("${TRY_TYPE}" MATCHES "COMPILES")
        CHECK_C_SOURCE_COMPILES("${SAMPLE_SOURCE}" ${VAR})
      ELSEIF("${TRY_TYPE}" MATCHES "RUNS")
        CHECK_C_SOURCE_RUNS("${SAMPLE_SOURCE}" ${VAR})
      ELSE("${TRY_TYPE}" MATCHES "COMPILES")
        MESSAGE(FATAL_ERROR "UNKNOWN KEYWORD \"${TRY_TYPE}\" FOR TRY_TYPE")
      ENDIF("${TRY_TYPE}" MATCHES "COMPILES")
      # Save the libraries which ${VAR} is checked with.
      SET(${VAR_WITH_LIB} "${LIBRARIES}" CACHE INTERNAL
          "Macro ${VAR} is checked with")
    ENDFOREACH(VAR)
    CMAKE_POP_CHECK_STATE()	# Restore the state of the variables
  ENDIF(WIN32 AND NOT CYGWIN)
ENDMACRO (TRY_MACRO_FOR_LIBRARY)
#
# Check compress/decompress libraries
#
IF(WIN32 AND NOT CMAKE_CL_64 AND NOT CYGWIN)
  # GnuWin32 is only for Win32, not Win64.
  SET(__GNUWIN32PATH "C:/Program Files/GnuWin32")
ENDIF(WIN32 AND NOT CMAKE_CL_64 AND NOT CYGWIN)
IF(DEFINED __GNUWIN32PATH AND EXISTS "${__GNUWIN32PATH}")
  # You have to add a path availabel DLL file into PATH environment variable.
  # Maybe DLL path is "C:/Program Files/GnuWin32/bin".
  # The zlib and the bzip2 Setup program have installed programs and DLLs into
  # "C:/Program Files/GnuWin32" by default.
  # This is convenience setting for Windows.
  SET(CMAKE_PREFIX_PATH ${__GNUWIN32PATH} $(CMAKE_PREFIX_PATH))
  #
  # If you didn't use Setup program or installed into nonstandard path,
  # cmake cannot find out your zlib or bzip2 libraries and include files,
  # you should execute cmake with  -DCMAKE_PREFIX_PATH option.
  #   e.g.
  #     cmake -DCMAKE_PREFIX_PATH=<your-GnuWin32-path> <path-to-source>
  #
  # If compiling error occurred in zconf.h, You may need patch to zconf.h.
  #--- zconf.h.orig	2005-07-21 00:40:26.000000000
  #+++ zconf.h	2009-01-19 11:39:10.093750000
  #@@ -286,7 +286,7 @@
  #
  # #if 1           /* HAVE_UNISTD_H -- this line is updated by ./configure */
  # #  include <sys/types.h> /* for off_t */
  #-#  include <unistd.h>    /* for SEEK_* and off_t */
  #+#  include <stdio.h>    /* for SEEK_* and off_t */
  # #  ifdef VMS
  # #    include <unixio.h>   /* for off_t */
  # #  endif
ENDIF(DEFINED __GNUWIN32PATH AND EXISTS "${__GNUWIN32PATH}")

SET(ADDITIONAL_LIBS "")
#
# Find ZLIB
#
IF(ENABLE_ZLIB)
  FIND_PACKAGE(ZLIB)
ELSE()
  SET(ZLIB_FOUND FALSE) # Override cached value
ENDIF()
IF(ZLIB_FOUND)
  SET(HAVE_LIBZ 1)
  SET(HAVE_ZLIB_H 1)
  INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR})
  LIST(APPEND ADDITIONAL_LIBS ${ZLIB_LIBRARIES})
  IF(WIN32 AND NOT CYGWIN)
    #
    # Test if ZLIB_WINAPI macro is needed to use.
    #
    TRY_MACRO_FOR_LIBRARY(
      "${ZLIB_INCLUDE_DIR}" "${ZLIB_LIBRARIES}"
      RUNS
      "#include <zlib.h>\nint main() {uLong f = zlibCompileFlags(); return (f&(1U<<10))?0:-1; }"
      ZLIB_WINAPI)
    IF(ZLIB_WINAPI)
      ADD_DEFINITIONS(-DZLIB_WINAPI)
    ELSE(ZLIB_WINAPI)
      # Test if a macro is needed for the library.
      TRY_MACRO_FOR_LIBRARY(
        "${ZLIB_INCLUDE_DIR}" "${ZLIB_LIBRARIES}"
        COMPILES
        "#include <zlib.h>\nint main() {return zlibVersion()?1:0; }"
        "ZLIB_DLL;WITHOUT_ZLIB_DLL")
      IF(ZLIB_DLL)
        ADD_DEFINITIONS(-DZLIB_DLL)
      ENDIF(ZLIB_DLL)
    ENDIF(ZLIB_WINAPI)
  ENDIF(WIN32 AND NOT CYGWIN)
ENDIF(ZLIB_FOUND)
MARK_AS_ADVANCED(CLEAR ZLIB_INCLUDE_DIR)
MARK_AS_ADVANCED(CLEAR ZLIB_LIBRARY)
#
# Check headers
#
CHECK_HEADER_DIRENT()

SET(INCLUDES "")
MACRO (LA_CHECK_INCLUDE_FILE header var)
      CHECK_INCLUDE_FILES("${INCLUDES};${header}" ${var})
      IF (${var})
      	 SET(INCLUDES ${INCLUDES} ${header})
      ENDIF (${var})
ENDMACRO (LA_CHECK_INCLUDE_FILE)

# Some FreeBSD headers assume sys/types.h was already included.
LA_CHECK_INCLUDE_FILE("sys/types.h" HAVE_SYS_TYPES_H)

# Alphabetize the rest unless there's a compelling reason
LA_CHECK_INCLUDE_FILE("attr/xattr.h" HAVE_ATTR_XATTR_H)
LA_CHECK_INCLUDE_FILE("copyfile.h" HAVE_COPYFILE_H)
LA_CHECK_INCLUDE_FILE("direct.h" HAVE_DIRECT_H)
LA_CHECK_INCLUDE_FILE("errno.h" HAVE_ERRNO_H)
LA_CHECK_INCLUDE_FILE("ext2fs/ext2_fs.h" HAVE_EXT2FS_EXT2_FS_H)

CHECK_C_SOURCE_COMPILES("#include <sys/ioctl.h>
#include <ext2fs/ext2_fs.h>
int main(void) { return EXT2_IOC_GETFLAGS; }" HAVE_WORKING_EXT2_IOC_GETFLAGS)

LA_CHECK_INCLUDE_FILE("fcntl.h" HAVE_FCNTL_H)
LA_CHECK_INCLUDE_FILE("grp.h" HAVE_GRP_H)
LA_CHECK_INCLUDE_FILE("inttypes.h" HAVE_INTTYPES_H)
LA_CHECK_INCLUDE_FILE("io.h" HAVE_IO_H)
LA_CHECK_INCLUDE_FILE("langinfo.h" HAVE_LANGINFO_H)
LA_CHECK_INCLUDE_FILE("limits.h" HAVE_LIMITS_H)
LA_CHECK_INCLUDE_FILE("linux/types.h" HAVE_LINUX_TYPES_H)
LA_CHECK_INCLUDE_FILE("linux/fiemap.h" HAVE_LINUX_FIEMAP_H)
LA_CHECK_INCLUDE_FILE("linux/fs.h" HAVE_LINUX_FS_H)

CHECK_C_SOURCE_COMPILES("#include <sys/ioctl.h>
#include <linux/fs.h>
int main(void) { return FS_IOC_GETFLAGS; }" HAVE_WORKING_FS_IOC_GETFLAGS)

LA_CHECK_INCLUDE_FILE("linux/magic.h" HAVE_LINUX_MAGIC_H)
LA_CHECK_INCLUDE_FILE("paths.h" HAVE_PATHS_H)
LA_CHECK_INCLUDE_FILE("pthread.h" HAVE_PTHREAD_H)
LA_CHECK_INCLUDE_FILE("pwd.h" HAVE_PWD_H)
LA_CHECK_INCLUDE_FILE("stdint.h" HAVE_STDINT_H)
LA_CHECK_INCLUDE_FILE("stdlib.h" HAVE_STDLIB_H)
LA_CHECK_INCLUDE_FILE("string.h" HAVE_STRING_H)
LA_CHECK_INCLUDE_FILE("sys/acl.h" HAVE_SYS_ACL_H)
LA_CHECK_INCLUDE_FILE("sys/cdefs.h" HAVE_SYS_CDEFS_H)
LA_CHECK_INCLUDE_FILE("sys/extattr.h" HAVE_SYS_EXTATTR_H)
LA_CHECK_INCLUDE_FILE("sys/ioctl.h" HAVE_SYS_IOCTL_H)
LA_CHECK_INCLUDE_FILE("sys/mount.h" HAVE_SYS_MOUNT_H)
LA_CHECK_INCLUDE_FILE("sys/param.h" HAVE_SYS_PARAM_H)
LA_CHECK_INCLUDE_FILE("sys/stat.h" HAVE_SYS_STAT_H)
LA_CHECK_INCLUDE_FILE("sys/statfs.h" HAVE_SYS_STATFS_H)
LA_CHECK_INCLUDE_FILE("sys/statvfs.h" HAVE_SYS_STATVFS_H)
LA_CHECK_INCLUDE_FILE("sys/sysmacros.h" HAVE_SYS_SYSMACROS_H)
LA_CHECK_INCLUDE_FILE("sys/time.h" HAVE_SYS_TIME_H)
LA_CHECK_INCLUDE_FILE("sys/utime.h" HAVE_SYS_UTIME_H)
LA_CHECK_INCLUDE_FILE("sys/wait.h" HAVE_SYS_WAIT_H)
LA_CHECK_INCLUDE_FILE("sys/xattr.h" HAVE_SYS_XATTR_H)
LA_CHECK_INCLUDE_FILE("unistd.h" HAVE_UNISTD_H)
LA_CHECK_INCLUDE_FILE("utime.h" HAVE_UTIME_H)
LA_CHECK_INCLUDE_FILE("wchar.h" HAVE_WCHAR_H)
LA_CHECK_INCLUDE_FILE("windows.h" HAVE_WINDOWS_H)
# Following files need windows.h, so we should test it after windows.h test.
LA_CHECK_INCLUDE_FILE("wincrypt.h" HAVE_WINCRYPT_H)
LA_CHECK_INCLUDE_FILE("winioctl.h" HAVE_WINIOCTL_H)

#
# Check whether use of __EXTENSIONS__ is safe.
# We need some macro such as _GNU_SOURCE to use extension functions.
#
SET(_INCLUDE_FILES)
FOREACH (it ${_HEADER})
   SET(_INCLUDE_FILES "${_INCLUDE_FILES}#include <${it}>\n")
ENDFOREACH (it)

CHECK_C_SOURCE_COMPILES(
  "#define __EXTENSIONS__ 1
   ${_INCLUDE_FILES}
   int main() { return 0;}"
 SAFE_TO_DEFINE_EXTENSIONS)

#
# Check functions
#
CMAKE_PUSH_CHECK_STATE()	# Save the state of the variables
IF (CMAKE_C_COMPILER_ID MATCHES "^GNU$" OR
    CMAKE_C_COMPILER_ID MATCHES "^Clang$")
  #
  # During checking functions, we should use -fno-builtin to avoid the
  # failure of function detection which failure is an error "conflicting
  # types for built-in function" caused by using -Werror option.
  #
  SET(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-builtin")
ENDIF (CMAKE_C_COMPILER_ID MATCHES "^GNU$" OR
       CMAKE_C_COMPILER_ID MATCHES "^Clang$")
CHECK_SYMBOL_EXISTS(_CrtSetReportMode "crtdbg.h" HAVE__CrtSetReportMode)
IF(WIN32)
  CHECK_FUNCTION_EXISTS_GLIBC(arc4random_buf HAVE_ARC4RANDOM_BUF)
  CHECK_FUNCTION_EXISTS_GLIBC(chflags HAVE_CHFLAGS)
  CHECK_FUNCTION_EXISTS_GLIBC(chown HAVE_CHOWN)
  CHECK_FUNCTION_EXISTS_GLIBC(fchdir HAVE_FCHDIR)
  CHECK_FUNCTION_EXISTS_GLIBC(fchflags HAVE_FCHFLAGS)
  CHECK_FUNCTION_EXISTS_GLIBC(fchmod HAVE_FCHMOD)
  CHECK_FUNCTION_EXISTS_GLIBC(fchown HAVE_FCHOWN)
  CHECK_FUNCTION_EXISTS_GLIBC(fcntl HAVE_FCNTL)
  CHECK_FUNCTION_EXISTS_GLIBC(fdopendir HAVE_FDOPENDIR)
  CHECK_FUNCTION_EXISTS_GLIBC(fstat HAVE_FSTAT)
  CHECK_FUNCTION_EXISTS_GLIBC(fstatat HAVE_FSTATAT)
  CHECK_FUNCTION_EXISTS_GLIBC(fstatfs HAVE_FSTATFS)
  CHECK_FUNCTION_EXISTS_GLIBC(fstatvfs HAVE_FSTATVFS)
  CHECK_FUNCTION_EXISTS_GLIBC(ftruncate HAVE_FTRUNCATE)
  CHECK_FUNCTION_EXISTS_GLIBC(futimens HAVE_FUTIMENS)
  CHECK_FUNCTION_EXISTS_GLIBC(futimes HAVE_FUTIMES)
  CHECK_FUNCTION_EXISTS_GLIBC(futimesat HAVE_FUTIMESAT)
  CHECK_FUNCTION_EXISTS_GLIBC(geteuid HAVE_GETEUID)
  CHECK_FUNCTION_EXISTS_GLIBC(getgrgid_r HAVE_GETGRGID_R)
  CHECK_FUNCTION_EXISTS_GLIBC(getgrnam_r HAVE_GETGRNAM_R)
  CHECK_FUNCTION_EXISTS_GLIBC(getpwnam_r HAVE_GETPWNAM_R)
  CHECK_FUNCTION_EXISTS_GLIBC(getpwuid_r HAVE_GETPWUID_R)
  CHECK_FUNCTION_EXISTS_GLIBC(getvfsbyname HAVE_GETVFSBYNAME)
  CHECK_FUNCTION_EXISTS_GLIBC(lchflags HAVE_LCHFLAGS)
  CHECK_FUNCTION_EXISTS_GLIBC(lchmod HAVE_LCHMOD)
  CHECK_FUNCTION_EXISTS_GLIBC(lchown HAVE_LCHOWN)
  CHECK_FUNCTION_EXISTS_GLIBC(link HAVE_LINK)
  CHECK_FUNCTION_EXISTS_GLIBC(lstat HAVE_LSTAT)
  CHECK_FUNCTION_EXISTS_GLIBC(lutimes HAVE_LUTIMES)
  CHECK_FUNCTION_EXISTS_GLIBC(mbrtowc HAVE_MBRTOWC)
  CHECK_FUNCTION_EXISTS_GLIBC(mkfifo HAVE_MKFIFO)
  CHECK_FUNCTION_EXISTS_GLIBC(mknod HAVE_MKNOD)
  CHECK_FUNCTION_EXISTS_GLIBC(mkstemp HAVE_MKSTEMP)
  CHECK_FUNCTION_EXISTS_GLIBC(nl_langinfo HAVE_NL_LANGINFO)
  CHECK_FUNCTION_EXISTS_GLIBC(openat HAVE_OPENAT)
  CHECK_FUNCTION_EXISTS_GLIBC(readlink HAVE_READLINK)
  CHECK_FUNCTION_EXISTS_GLIBC(statfs HAVE_STATFS)
  CHECK_FUNCTION_EXISTS_GLIBC(statvfs HAVE_STATVFS)
  CHECK_FUNCTION_EXISTS_GLIBC(symlink HAVE_SYMLINK)
  CHECK_FUNCTION_EXISTS_GLIBC(utime HAVE_UTIME)
  CHECK_FUNCTION_EXISTS_GLIBC(utimes HAVE_UTIMES)
  CHECK_FUNCTION_EXISTS_GLIBC(utimensat HAVE_UTIMENSAT)
  CHECK_FUNCTION_EXISTS_GLIBC(wcrtomb HAVE_WCRTOMB)
  CHECK_FUNCTION_EXISTS_GLIBC(wcscpy HAVE_WCSCPY)
  CHECK_FUNCTION_EXISTS_GLIBC(wcslen HAVE_WCSLEN)
  CHECK_FUNCTION_EXISTS_GLIBC(wctomb HAVE_WCTOMB)
  CHECK_FUNCTION_EXISTS_GLIBC(_fseeki64 HAVE__FSEEKI64)
ELSE(WIN32)
  set(HAVE_ARC4RANDOM_BUF 0)
  set(HAVE_CHFLAGS 1)
  set(HAVE_CHOWN 1)
  set(HAVE_FCHDIR 1)
  set(HAVE_FCHFLAGS 1)
  set(HAVE_FCHMOD 1)
  set(HAVE_FCHOWN 1)
  set(HAVE_FCNTL 1)
  set(HAVE_FDOPENDIR 1)
  set(HAVE_FSTAT 1)
  set(HAVE_FSTATAT 1)
  set(HAVE_FSTATFS 1)
  set(HAVE_FSTATVFS 1)
  set(HAVE_FTRUNCATE 1)
  set(HAVE_FUTIMENS 1)
  set(HAVE_FUTIMES 1)
  set(HAVE_FUTIMESAT 1)
  set(HAVE_GETEUID 1)
  set(HAVE_GETGRGID_R 1)
  set(HAVE_GETGRNAM_R 1)
  set(HAVE_GETPWNAM_R 1)
  set(HAVE_GETPWUID_R 1)
  set(HAVE_GETVFSBYNAME 1)
  set(HAVE_LCHFLAGS 1)
  set(HAVE_LCHMOD 1)
  set(HAVE_LCHOWN 1)
  set(HAVE_LINK 1)
  set(HAVE_LSTAT 1)
  set(HAVE_LUTIMES 1)
  set(HAVE_MBRTOWC 1)
  set(HAVE_MKFIFO 1)
  set(HAVE_MKNOD 1)
  set(HAVE_MKSTEMP 1)
  set(HAVE_NL_LANGINFO 1)
  set(HAVE_OPENAT 1)
  set(HAVE_READLINK 1)
  set(HAVE_STATFS 1)
  set(HAVE_STATVFS 1)
  set(HAVE_SYMLINK 1)
  set(HAVE_UTIME 1)
  set(HAVE_UTIMES 1)
  set(HAVE_UTIMENSAT 1)
  set(HAVE_WCRTOMB 1)
  set(HAVE_WCSCPY 1)
  set(HAVE_WCSLEN 1)
  set(HAVE_WCTOMB 1)
  set(HAVE__FSEEKI64 1)
ENDIF(WIN32)

SET(CMAKE_REQUIRED_LIBRARIES "")
CHECK_FUNCTION_EXISTS(cygwin_conv_path HAVE_CYGWIN_CONV_PATH)
CHECK_FUNCTION_EXISTS(fseeko HAVE_FSEEKO)
CHECK_FUNCTION_EXISTS(strerror_r HAVE_STRERROR_R)
CHECK_FUNCTION_EXISTS(wmemcmp HAVE_WMEMCMP)
CHECK_FUNCTION_EXISTS(wmemcpy HAVE_WMEMCPY)
CHECK_FUNCTION_EXISTS(wmemmove HAVE_WMEMMOVE)

CMAKE_POP_CHECK_STATE()	# Restore the state of the variables

CHECK_C_SOURCE_COMPILES(
  "#include <sys/types.h>\n#include <sys/mount.h>\nint main(void) { struct vfsconf v; return sizeof(v);}"
  HAVE_STRUCT_VFSCONF)

CHECK_C_SOURCE_COMPILES(
  "#include <sys/types.h>\n#include <sys/mount.h>\nint main(void) { struct xvfsconf v; return sizeof(v);}"
  HAVE_STRUCT_XVFSCONF)

# Make sure we have the POSIX version of readdir_r, not the
# older 2-argument version.
CHECK_C_SOURCE_COMPILES(
  "#include <dirent.h>\nint main() {DIR *d = opendir(\".\"); struct dirent e,*r; return readdir_r(d,&e,&r);}"
  HAVE_READDIR_R)

# dirfd can be either a function or a macro.
CHECK_C_SOURCE_COMPILES(
  "#include <dirent.h>\nint main() {DIR *d = opendir(\".\"); return dirfd(d);}"
  HAVE_DIRFD)

# Only detect readlinkat() if we also have AT_FDCWD in unistd.h.
# NOTE: linux requires fcntl.h for AT_FDCWD.
CHECK_C_SOURCE_COMPILES(
  "#include <fcntl.h>\n#include <unistd.h>\nint main() {char buf[10]; return readlinkat(AT_FDCWD, \"\", buf, 0);}"
  HAVE_READLINKAT)


# To verify major(), we need to both include the header
# of interest and verify that the result can be linked.
# CHECK_FUNCTION_EXISTS doesn't accept a header argument,
# CHECK_SYMBOL_EXISTS doesn't test linkage.
CHECK_C_SOURCE_COMPILES(
  "#include <sys/mkdev.h>\nint main() { return major(256); }"
  MAJOR_IN_MKDEV)
CHECK_C_SOURCE_COMPILES(
  "#include <sys/sysmacros.h>\nint main() { return major(256); }"
  MAJOR_IN_SYSMACROS)

IF(HAVE_STRERROR_R)
  SET(HAVE_DECL_STRERROR_R 1)
ENDIF(HAVE_STRERROR_R)

#
# Check defines
#
SET(headers "limits.h")
IF(HAVE_STDINT_H)
  LIST(APPEND headers "stdint.h")
ENDIF(HAVE_STDINT_H)
IF(HAVE_INTTYPES_H)
  LIST(APPEND headers "inttypes.h")
ENDIF(HAVE_INTTYPES_H)
CHECK_SYMBOL_EXISTS(EFTYPE           "errno.h"    HAVE_EFTYPE)
CHECK_SYMBOL_EXISTS(EILSEQ           "errno.h"    HAVE_EILSEQ)
CHECK_SYMBOL_EXISTS(D_MD_ORDER       "langinfo.h" HAVE_D_MD_ORDER)
CHECK_SYMBOL_EXISTS(INT32_MAX        "${headers}" HAVE_DECL_INT32_MAX)
CHECK_SYMBOL_EXISTS(INT32_MIN        "${headers}" HAVE_DECL_INT32_MIN)
CHECK_SYMBOL_EXISTS(INT64_MAX        "${headers}" HAVE_DECL_INT64_MAX)
CHECK_SYMBOL_EXISTS(INT64_MIN        "${headers}" HAVE_DECL_INT64_MIN)
CHECK_SYMBOL_EXISTS(INTMAX_MAX       "${headers}" HAVE_DECL_INTMAX_MAX)
CHECK_SYMBOL_EXISTS(INTMAX_MIN       "${headers}" HAVE_DECL_INTMAX_MIN)
CHECK_SYMBOL_EXISTS(UINT32_MAX       "${headers}" HAVE_DECL_UINT32_MAX)
CHECK_SYMBOL_EXISTS(UINT64_MAX       "${headers}" HAVE_DECL_UINT64_MAX)
CHECK_SYMBOL_EXISTS(UINTMAX_MAX      "${headers}" HAVE_DECL_UINTMAX_MAX)
CHECK_SYMBOL_EXISTS(SIZE_MAX         "${headers}" HAVE_DECL_SIZE_MAX)
CHECK_SYMBOL_EXISTS(SSIZE_MAX        "limits.h"   HAVE_DECL_SSIZE_MAX)

#
# Check struct members
#
# Check for tm_gmtoff in struct tm
CHECK_STRUCT_HAS_MEMBER("struct tm" tm_gmtoff
    "time.h" HAVE_STRUCT_TM_TM_GMTOFF)
CHECK_STRUCT_HAS_MEMBER("struct tm" __tm_gmtoff
    "time.h" HAVE_STRUCT_TM___TM_GMTOFF)

# Check for f_namemax in struct statfs
CHECK_STRUCT_HAS_MEMBER("struct statfs" f_namemax
    "sys/param.h;sys/mount.h" HAVE_STRUCT_STATFS_F_NAMEMAX)

# Check for birthtime in struct stat
CHECK_STRUCT_HAS_MEMBER("struct stat" st_birthtime
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_BIRTHTIME)

# Check for high-resolution timestamps in struct stat
CHECK_STRUCT_HAS_MEMBER("struct stat" st_birthtimespec.tv_nsec
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC)
CHECK_STRUCT_HAS_MEMBER("struct stat" st_mtimespec.tv_nsec
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_MTIMESPEC_TV_NSEC)
CHECK_STRUCT_HAS_MEMBER("struct stat" st_mtim.tv_nsec
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC)
CHECK_STRUCT_HAS_MEMBER("struct stat" st_mtime_n
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_MTIME_N)
CHECK_STRUCT_HAS_MEMBER("struct stat" st_umtime
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_UMTIME)
CHECK_STRUCT_HAS_MEMBER("struct stat" st_mtime_usec
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_MTIME_USEC)
# Check for block size support in struct stat
CHECK_STRUCT_HAS_MEMBER("struct stat" st_blksize
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_BLKSIZE)
# Check for st_flags in struct stat (BSD fflags)
CHECK_STRUCT_HAS_MEMBER("struct stat" st_flags
    "sys/types.h;sys/stat.h" HAVE_STRUCT_STAT_ST_FLAGS)

IF(HAVE_SYS_STATVFS_H)
  CHECK_STRUCT_HAS_MEMBER("struct statvfs" f_iosize
    "sys/types.h;sys/statvfs.h" HAVE_STRUCT_STATVFS_F_IOSIZE)
ENDIF()

#
#
CHECK_STRUCT_HAS_MEMBER("struct tm" tm_sec
    "sys/types.h;sys/time.h;time.h" TIME_WITH_SYS_TIME)

#
# Check for integer types
#
#
CHECK_TYPE_SIZE("short" SIZE_OF_SHORT)
CHECK_TYPE_SIZE("int" SIZE_OF_INT)
CHECK_TYPE_SIZE("long" SIZE_OF_LONG)
CHECK_TYPE_SIZE("long long"     SIZE_OF_LONG_LONG)

CHECK_TYPE_SIZE("unsigned short" SIZE_OF_UNSIGNED_SHORT)
CHECK_TYPE_SIZE("unsigned" SIZE_OF_UNSIGNED)
CHECK_TYPE_SIZE("unsigned long" SIZE_OF_UNSIGNED_LONG)
CHECK_TYPE_SIZE("unsigned long long" SIZE_OF_UNSIGNED_LONG_LONG)

CHECK_TYPE_SIZE("__int64" __INT64)
CHECK_TYPE_SIZE("unsigned __int64" UNSIGNED___INT64)

CHECK_TYPE_SIZE(int16_t INT16_T)
CHECK_TYPE_SIZE(int32_t INT32_T)
CHECK_TYPE_SIZE(int64_t INT64_T)
CHECK_TYPE_SIZE(intmax_t INTMAX_T)
CHECK_TYPE_SIZE(uint8_t UINT8_T)
CHECK_TYPE_SIZE(uint16_t UINT16_T)
CHECK_TYPE_SIZE(uint32_t UINT32_T)
CHECK_TYPE_SIZE(uint64_t UINT64_T)
CHECK_TYPE_SIZE(uintmax_t UINTMAX_T)

CHECK_TYPE_SIZE(dev_t       DEV_T)
IF(NOT HAVE_DEV_T)
  IF(MSVC)
    SET(dev_t "unsigned int")
  ENDIF(MSVC)
ENDIF(NOT HAVE_DEV_T)
#
CHECK_TYPE_SIZE(gid_t       GID_T)
IF(NOT HAVE_GID_T)
  IF(WIN32)
    SET(gid_t "short")
  ELSE(WIN32)
    SET(gid_t "unsigned int")
  ENDIF(WIN32)
ENDIF(NOT HAVE_GID_T)
#
CHECK_TYPE_SIZE(id_t        ID_T)
IF(NOT HAVE_ID_T)
  IF(WIN32)
    SET(id_t "short")
  ELSE(WIN32)
    SET(id_t "unsigned int")
  ENDIF(WIN32)
ENDIF(NOT HAVE_ID_T)
#
CHECK_TYPE_SIZE(mode_t      MODE_T)
IF(NOT HAVE_MODE_T)
  IF(WIN32)
    SET(mode_t "unsigned short")
  ELSE(WIN32)
    SET(mode_t "int")
  ENDIF(WIN32)
ENDIF(NOT HAVE_MODE_T)
#
CHECK_TYPE_SIZE(off_t       OFF_T)
IF(NOT HAVE_OFF_T)
  SET(off_t "__int64")
ENDIF(NOT HAVE_OFF_T)
#
CHECK_TYPE_SIZE(size_t      SIZE_T)
IF(NOT HAVE_SIZE_T)
  IF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(size_t "uint64_t")
  ELSE("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(size_t   "uint32_t")
  ENDIF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
ENDIF(NOT HAVE_SIZE_T)
#
CHECK_TYPE_SIZE(ssize_t     SSIZE_T)
IF(NOT HAVE_SSIZE_T)
  IF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(ssize_t "int64_t")
  ELSE("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(ssize_t "long")
  ENDIF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
ENDIF(NOT HAVE_SSIZE_T)
#
CHECK_TYPE_SIZE(uid_t       UID_T)
IF(NOT HAVE_UID_T)
  IF(WIN32)
    SET(uid_t "short")
  ELSE(WIN32)
    SET(uid_t "unsigned int")
  ENDIF(WIN32)
ENDIF(NOT HAVE_UID_T)
#
CHECK_TYPE_SIZE(pid_t       PID_T)
IF(NOT HAVE_PID_T)
  IF(WIN32)
    SET(pid_t "int")
  ELSE(WIN32)
    MESSAGE(FATAL_ERROR "pid_t doesn't exist on this platform?")
  ENDIF(WIN32)
ENDIF(NOT HAVE_PID_T)
#
CHECK_TYPE_SIZE(intptr_t   INTPTR_T)
IF(NOT HAVE_INTPTR_T)
  IF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(intptr_t "int64_t")
  ELSE()
    SET(intptr_t "int32_t")
  ENDIF()
ENDIF(NOT HAVE_INTPTR_T)
#
CHECK_TYPE_SIZE(uintptr_t   UINTPTR_T)
IF(NOT HAVE_UINTPTR_T)
  IF("${CMAKE_SIZEOF_VOID_P}" EQUAL 8)
    SET(uintptr_t "uint64_t")
  ELSE()
    SET(uintptr_t "uint32_t")
  ENDIF()
ENDIF(NOT HAVE_UINTPTR_T)
#
CHECK_TYPE_SIZE(wchar_t     SIZEOF_WCHAR_T)
IF(HAVE_SIZEOF_WCHAR_T)
  SET(HAVE_WCHAR_T 1)
ENDIF(HAVE_SIZEOF_WCHAR_T)
#
# Check if _FILE_OFFSET_BITS macro needed for large files
#
CHECK_FILE_OFFSET_BITS()

# Generate "config.h" from "build/cmake/config.h.in"
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/build/cmake/config.h.in
	${CMAKE_CURRENT_BINARY_DIR}/config.h)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR})
ADD_DEFINITIONS(-DHAVE_CONFIG_H)

# Handle generation of the libarchive.pc file for pkg-config
# INCLUDE(CreatePkgConfigFile)

#
# Register installation of PDF documents.
#
IF(WIN32 AND NOT CYGWIN)
  #
  # On Windows platform, It's better that we install PDF documents
  # on one's computer.
  # These PDF documents are available in the release package.
  #
  IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/doc/pdf)
    INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doc/pdf
            DESTINATION share/man
            FILES_MATCHING PATTERN "*.pdf"
    )
  ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/doc/pdf)
ENDIF(WIN32 AND NOT CYGWIN)
#
#
#
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/libarchive)
#
IF(MSVC)
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
ENDIF(MSVC)


add_subdirectory(libarchive)
