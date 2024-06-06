# - Find ROOT instalation
# This module tries to find the ROOT installation on your system.
# It tries to find the root-config script which gives you all the needed information.
# If the system variable ROOTSYS is set this is straight forward.
# If not the module uses the pathes given in ROOT_CONFIG_SEARCHPATH.
# If you need an other path you should add this path to this varaible.  
# The root-config script is then used to detect basically everything else.
# This module defines a number of key variables and macros.

# F.Uhlig@gsi.de (fairroot.gsi.de)


MESSAGE(STATUS "Looking for Root...")

SET(ROOT_CONFIG_SEARCHPATH
  ${SIMPATH}/tools/root/bin
  $ENV{ROOTSYS}/bin
  /usr/local/bin/
)

SET(ROOT_DEFINITIONS "")

SET(ROOT_INSTALLED_VERSION_TOO_OLD FALSE)

SET(ROOT_CONFIG_EXECUTABLE ROOT_CONFIG_EXECUTABLE-NOTFOUND)

FIND_PROGRAM(ROOT_CONFIG_EXECUTABLE NAMES root-config PATHS
   ${ROOT_CONFIG_SEARCHPATH}
   NO_DEFAULT_PATH)
    
IF (${ROOT_CONFIG_EXECUTABLE} MATCHES "ROOT_CONFIG_EXECUTABLE-NOTFOUND")
  MESSAGE( FATAL_ERROR "ROOT not installed in the searchpath and ROOTSYS is not set. Please
 set ROOTSYS or add the path to your ROOT installation in the Macro FindROOT.cmake in the
 subdirectory cmake/modules.")
ELSE (${ROOT_CONFIG_EXECUTABLE} MATCHES "ROOT_CONFIG_EXECUTABLE-NOTFOUND")
  STRING(REGEX REPLACE "(^.*)/bin/root-config" "\\1" test ${ROOT_CONFIG_EXECUTABLE}) 
  SET( ENV{ROOTSYS} ${test})
  set( ROOTSYS ${test})
ENDIF (${ROOT_CONFIG_EXECUTABLE} MATCHES "ROOT_CONFIG_EXECUTABLE-NOTFOUND")  

 
IF (ROOT_CONFIG_EXECUTABLE)
   
  SET(ROOT_FOUND FALSE)

  EXEC_PROGRAM(${ROOT_CONFIG_EXECUTABLE} ARGS "--version" OUTPUT_VARIABLE ROOTVERSION)

  MESSAGE(STATUS "Looking for Root... - found $ENV{ROOTSYS}/bin/root")
  MESSAGE(STATUS "Looking for Root... - version ${ROOTVERSION} ")   

  # we need at least version 5.00/00
  IF (NOT ROOT_MIN_VERSION)
    SET(ROOT_MIN_VERSION "5.00/00")
  ENDIF (NOT ROOT_MIN_VERSION)
   
  # now parse the parts of the user given version string into variables
  STRING(REGEX REPLACE "^([0-9]+)\\.[0-9][0-9]+\\/[0-9][0-9]+" "\\1" req_root_major_vers "${ROOT_MIN_VERSION}")
  STRING(REGEX REPLACE "^[0-9]+\\.([0-9][0-9])+\\/[0-9][0-9]+.*" "\\1" req_root_minor_vers "${ROOT_MIN_VERSION}")
  STRING(REGEX REPLACE "^[0-9]+\\.[0-9][0-9]+\\/([0-9][0-9]+)" "\\1" req_root_patch_vers "${ROOT_MIN_VERSION}")
   
  # and now the version string given by qmake
  STRING(REGEX REPLACE "^([0-9]+)\\.[0-9][0-9]+\\/[0-9][0-9]+.*" "\\1" found_root_major_vers "${ROOTVERSION}")
  STRING(REGEX REPLACE "^[0-9]+\\.([0-9][0-9])+\\/[0-9][0-9]+.*" "\\1" found_root_minor_vers "${ROOTVERSION}")
  STRING(REGEX REPLACE "^[0-9]+\\.[0-9][0-9]+\\/([0-9][0-9]+).*" "\\1" found_root_patch_vers "${ROOTVERSION}")

  IF (found_root_major_vers LESS 5)
    MESSAGE( FATAL_ERROR "Invalid ROOT version \"${ROOTERSION}\", at least major version 4 is required, e.g. \"5.00/00\"")
  ENDIF (found_root_major_vers LESS 5)

  # compute an overall version number which can be compared at once
  MATH(EXPR req_vers "${req_root_major_vers}*10000 + ${req_root_minor_vers}*100 + ${req_root_patch_vers}")
  MATH(EXPR found_vers "${found_root_major_vers}*10000 + ${found_root_minor_vers}*100 + ${found_root_patch_vers}")
   
  IF (found_vers LESS req_vers)
    SET(ROOT_FOUND FALSE)
    SET(ROOT_INSTALLED_VERSION_TOO_OLD TRUE)
  ELSE (found_vers LESS req_vers)
    SET(ROOT_FOUND TRUE)
  ENDIF (found_vers LESS req_vers)

ENDIF (ROOT_CONFIG_EXECUTABLE)


IF (ROOT_FOUND)

  # ask root-config for the library dir
  # Set ROOT_LIBRARY_DIR

  EXEC_PROGRAM( ${ROOT_CONFIG_EXECUTABLE}
    ARGS "--libdir"
    OUTPUT_VARIABLE ROOT_LIBRARY_DIR_TMP )

  IF(EXISTS "${ROOT_LIBRARY_DIR_TMP}")
    SET(ROOT_LIBRARY_DIR ${ROOT_LIBRARY_DIR_TMP} )
  ELSE(EXISTS "${ROOT_LIBRARY_DIR_TMP}")
    MESSAGE("Warning: ROOT_CONFIG_EXECUTABLE reported ${ROOT_LIBRARY_DIR_TMP} as library path,")
    MESSAGE("Warning: but ${ROOT_LIBRARY_DIR_TMP} does NOT exist, ROOT must NOT be installed correctly.")
  ENDIF(EXISTS "${ROOT_LIBRARY_DIR_TMP}")
    
  # ask root-config for the binary dir
  EXEC_PROGRAM(${ROOT_CONFIG_EXECUTABLE}
    ARGS "--bindir"
    OUTPUT_VARIABLE root_bins )
  SET(ROOT_BINARY_DIR ${root_bins})

  # ask root-config for the include dir
  EXEC_PROGRAM( ${ROOT_CONFIG_EXECUTABLE}
    ARGS "--incdir" 
    OUTPUT_VARIABLE root_headers )
  SET(ROOT_INCLUDE_DIR ${root_headers})
      # CACHE INTERNAL "")

  # ask root-config for the library varaibles
  EXEC_PROGRAM( ${ROOT_CONFIG_EXECUTABLE}
#    ARGS "--noldflags --noauxlibs --libs" 
    ARGS "--glibs" 
    OUTPUT_VARIABLE root_flags )

#  STRING(REGEX MATCHALL "([^ ])+"  root_libs_all ${root_flags})
#  STRING(REGEX MATCHALL "-L([^ ])+"  root_library ${root_flags})
#  REMOVE_FROM_LIST(root_flags "${root_libs_all}" "${root_library}")

  SET(ROOT_LIBRARIES ${root_flags})
  SET(ROOFIT_LIBRARIES "-lRooFit -lRooFitCore -lRooStats -lHtml -lMinuit -lThread -lFoam -lMatrix -lMathMore")

  # Make variables changeble to the advanced user
  MARK_AS_ADVANCED( ROOT_LIBRARY_DIR ROOT_INCLUDE_DIR ROOT_DEFINITIONS)

  # Set ROOT_INCLUDES
  SET( ROOT_INCLUDES ${ROOT_INCLUDE_DIR})

  SET(LD_LIBRARY_PATH ${LD_LIBRARY_PATH} ${ROOT_LIBRARY_DIR})


  #######################################
  #
  #       Check the executables of ROOT 
  #          ( rootcint ) 
  #
  #######################################

  FIND_PROGRAM(ROOT_CINT_EXECUTABLE
    NAMES rootcint
    PATHS ${ROOT_BINARY_DIR}
    NO_DEFAULT_PATH
    )


ENDIF (ROOT_FOUND)



  ###########################################
  #
  #       Macros for building ROOT dictionary
  #
  ###########################################

MACRO (ROOT_GENERATE_DICTIONARY_OLD )
 
   set(INFILES "")    

   foreach (_current_FILE ${ARGN})

     IF (${_current_FILE} MATCHES "^.*\\.h$")
       IF (${_current_FILE} MATCHES "^.*Link.*$")
         set(LINKDEF_FILE ${_current_FILE})
       ELSE (${_current_FILE} MATCHES "^.*Link.*$")
         set(INFILES ${INFILES} ${_current_FILE})
       ENDIF (${_current_FILE} MATCHES "^.*Link.*$")
     ELSE (${_current_FILE} MATCHES "^.*\\.h$")
       IF (${_current_FILE} MATCHES "^.*\\.cxx$")
         set(OUTFILE ${_current_FILE})
       ELSE (${_current_FILE} MATCHES "^.*\\.cxx$")
         set(INCLUDE_DIRS ${INCLUDE_DIRS} -I${_current_FILE})   
       ENDIF (${_current_FILE} MATCHES "^.*\\.cxx$")
     ENDIF (${_current_FILE} MATCHES "^.*\\.h$")
     
   endforeach (_current_FILE ${ARGN})
   
#  MESSAGE("INFILES: ${INFILES}")
#  MESSAGE("OutFILE: ${OUTFILE}")
#  MESSAGE("LINKDEF_FILE: ${LINKDEF_FILE}")
#  MESSAGE("INCLUDE_DIRS: ${INCLUDE_DIRS}")

   STRING(REGEX REPLACE "(^.*).cxx" "\\1.h" bla "${OUTFILE}")
#   MESSAGE("BLA: ${bla}")
   SET (OUTFILES ${OUTFILE} ${bla})

   ADD_CUSTOM_COMMAND(OUTPUT ${OUTFILES}
      COMMAND ${ROOT_CINT_EXECUTABLE}
      ARGS -f ${OUTFILE} -c -DHAVE_CONFIG_H ${INCLUDE_DIRS} ${INFILES} ${LINKDEF_FILE} DEPENDS ${INFILES})

#   MESSAGE("ROOT_CINT_EXECUTABLE has created the dictionary ${OUTFILE}")

ENDMACRO (ROOT_GENERATE_DICTIONARY_OLD)

  ###########################################
  #
  #       Macros for building ROOT dictionary
  #
  ###########################################

MACRO (ROOT_GENERATE_DICTIONARY INFILES LINKDEF_FILE OUTFILE INCLUDE_DIRS_IN)
 
  set(INCLUDE_DIRS)

  foreach (_current_FILE ${INCLUDE_DIRS_IN})
    set(INCLUDE_DIRS ${INCLUDE_DIRS} -I${_current_FILE})   
  endforeach (_current_FILE ${INCLUDE_DIRS_IN})
 

#  MESSAGE("INFILES: ${INFILES}")
#  MESSAGE("OutFILE: ${OUTFILE}")
#  MESSAGE("LINKDEF_FILE: ${LINKDEF_FILE}")
#  MESSAGE("INCLUDE_DIRS: ${INCLUDE_DIRS}")

  STRING(REGEX REPLACE "^(.*)\\.(.*)$" "\\1.h" bla "${OUTFILE}")
#  MESSAGE("BLA: ${bla}")
  SET (OUTFILES ${OUTFILE} ${bla})


  if (CMAKE_SYSTEM_NAME MATCHES Linux)
    ADD_CUSTOM_COMMAND(OUTPUT ${OUTFILES}
       COMMAND LD_LIBRARY_PATH=${ROOT_LIBRARY_DIR} ROOTSYS=${ROOTSYS} ${ROOT_CINT_EXECUTABLE}
       ARGS -f ${OUTFILE} -c -DHAVE_CONFIG_H ${INCLUDE_DIRS} ${INFILES} ${LINKDEF_FILE} DEPENDS ${INFILES} ${LINKDEF_FILE})
  else (CMAKE_SYSTEM_NAME MATCHES Linux)
    if (CMAKE_SYSTEM_NAME MATCHES Darwin)
      ADD_CUSTOM_COMMAND(OUTPUT ${OUTFILES}
       COMMAND DYLD_LIBRARY_PATH=${ROOT_LIBRARY_DIR} ROOTSYS=${ROOTSYS} ${ROOT_CINT_EXECUTABLE}
       ARGS -f ${OUTFILE} -c -DHAVE_CONFIG_H ${INCLUDE_DIRS} ${INFILES} ${LINKDEF_FILE} DEPENDS ${INFILES} ${LINKDEF_FILE})
    endif (CMAKE_SYSTEM_NAME MATCHES Darwin)
  endif (CMAKE_SYSTEM_NAME MATCHES Linux)

ENDMACRO (ROOT_GENERATE_DICTIONARY)

MACRO (GENERATE_ROOT_TEST_SCRIPT SCRIPT_FULL_NAME)

  get_filename_component(path_name ${SCRIPT_FULL_NAME} PATH)
  get_filename_component(file_extension ${SCRIPT_FULL_NAME} EXT)
  get_filename_component(file_name ${SCRIPT_FULL_NAME} NAME_WE)
  set(shell_script_name "${file_name}.sh")

  #MESSAGE("PATH: ${path_name}")
  #MESSAGE("Ext: ${file_extension}")
  #MESSAGE("Name: ${file_name}")
  #MESSAGE("Shell Name: ${shell_script_name}")

  string(REPLACE ${PROJECT_SOURCE_DIR} 
         ${PROJECT_BINARY_DIR} new_path ${path_name}
        )

  #MESSAGE("New PATH: ${new_path}")

  file(MAKE_DIRECTORY ${new_path}/data)

  CONVERT_LIST_TO_STRING(${LD_LIBRARY_PATH})
  set(MY_LD_LIBRARY_PATH ${output})
  set(my_script_name ${SCRIPT_FULL_NAME})

  if(CMAKE_SYSTEM MATCHES Darwin)
    configure_file(${PROJECT_SOURCE_DIR}/cmake/scripts/root_macro_macos.sh.in
                   ${new_path}/${shell_script_name}
                  )
  else(CMAKE_SYSTEM MATCHES Darwin)
    configure_file(${PROJECT_SOURCE_DIR}/cmake/scripts/root_macro.sh.in
                   ${new_path}/${shell_script_name}
                  )
  endif(CMAKE_SYSTEM MATCHES Darwin)

  EXEC_PROGRAM(/bin/chmod ARGS "u+x  ${new_path}/${shell_script_name}")

ENDMACRO (GENERATE_ROOT_TEST_SCRIPT)

#
# A macro that uses rootcint to generate dictionaries for ROOT classes.
#
# The first argument is the final library that will hold the dictionaries, the
# other arguments are the header files of the classes that need a dictionary.
#

macro (root_generate_dictionaries_local)

  set (_INHEADERS ${ARGV})
  list (GET _INHEADERS 0 _FINALDICT)
  list (REMOVE_AT _INHEADERS 0)

  foreach (_INHEADER ${_INHEADERS})

    get_filename_component (_CLASS_BASE_NAME ${_INHEADER} NAME_WE)
    set (_OUTDICT ${_CLASS_BASE_NAME}_dict.cc)
    set (_OUTCOMP ${_CLASS_BASE_NAME}_dict.cc_completed)
    set (_OUTTARGET ${_CLASS_BASE_NAME}_TARGET)
    set (_OUTDICT_HEADER ${_CLASS_BASE_NAME}_dict.h)
    set (_INHEADER_LINKDEF ${_CLASS_BASE_NAME}_LinkDef.h)
    get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)

    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER_LINKDEF})
      add_custom_command (
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_OUTCOMP} ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT_HEADER}
        #COMMAND LD_LIBRARY_PATH=${Root_LIBDIR}:$ENV{LD_LIBRARY_PATH} ${ROOTSYS}/bin/rootcint -f ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} -c -I${Root_INCDIR} -p ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER} ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER_LINKDEF}
        COMMAND LD_LIBRARY_PATH=${Root_LIBDIR}:$ENV{LD_LIBRARY_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules/rootcint_handler.py ${ROOTSYS}/bin/rootcint ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER} ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER_LINKDEF} ${inc_dirs}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER}
        IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER}
        COMMENT "Generating dictionary for class ${_CLASS_BASE_NAME} with LinkDef ${_INHEADER_LINKDEF}"
      )
    else (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER_LINKDEF})
      add_custom_command (
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${_OUTCOMP} ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT_HEADER}
        #COMMAND LD_LIBRARY_PATH=${Root_LIBDIR}:$ENV{LD_LIBRARY_PATH} ${ROOTSYS}/bin/rootcint -f ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} -c -I${Root_INCDIR} -p ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER}
        COMMAND LD_LIBRARY_PATH=${Root_LIBDIR}:$ENV{LD_LIBRARY_PATH} ${CMAKE_SOURCE_DIR}/cmake/Modules/rootcint_handler.py ${ROOTSYS}/bin/rootcint ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER}  ${inc_dirs}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER}
        IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER}
        COMMENT "Generating dictionary for class ${_CLASS_BASE_NAME}"
      )
    endif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${_INHEADER_LINKDEF})

    add_custom_target(${_OUTTARGET} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${_OUTCOMP} ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT_HEADER})
    list (APPEND _ROOT_DICTS ${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT})

  endforeach ()

  # This command adds at once the dictionaries to a single static library
  #add_library(${_FINALDICT} SHARED ${_ROOT_DICTS}) # Prevent an isolated library for the dictionary => commented out

  # make sure library depends on rootcint output (otherwise multi-core builds will fail)
  foreach (_INHEADER ${_INHEADERS})
    get_filename_component (_CLASS_BASE_NAME ${_INHEADER} NAME_WE)
    set (_OUTDICT ${_CLASS_BASE_NAME}_dict.cc)
    set (_OUTCOMP ${_CLASS_BASE_NAME}_dict.cc_completed)
    set (_OUTDICT_HEADER ${_CLASS_BASE_NAME}_dict.h)
    set (_OUTTARGET ${_CLASS_BASE_NAME}_TARGET)
    # Tell CMake the source won't be available until build time.
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT} PROPERTIES GENERATED 1)
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_OUTCOMP} PROPERTIES GENERATED 1)
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/${_OUTDICT_HEADER} PROPERTIES GENERATED 1)
    #add_dependencies(${_FINALDICT} ${_OUTTARGET}) # As no library is built, this needed to be commented out. Transfer to your local CMakeLists.txt
  endforeach ()

  set(${_FINALDICT} "${_ROOT_DICTS}")
endmacro ()
