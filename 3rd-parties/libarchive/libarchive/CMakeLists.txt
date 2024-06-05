
############################################
#
# How to build libarchive
#
############################################

# Public headers
SET(include_HEADERS
  archive.h
  archive_entry.h
)

# Sources and private headers
SET(libarchive_SOURCES
  archive_acl.c
  archive_acl_private.h
  archive_check_magic.c
  archive_crc32.h
  archive_cryptor.c
  archive_cryptor_private.h
  archive_digest.c
  archive_digest_private.h
  archive_endian.h
  archive_entry.c
  archive_entry.h
  archive_entry_copy_stat.c
  archive_entry_link_resolver.c
  archive_entry_locale.h
  archive_entry_private.h
  archive_entry_sparse.c
  archive_entry_stat.c
  archive_entry_strmode.c
  archive_entry_xattr.c
  archive_getdate.c
  archive_getdate.h
  archive_hmac.c
  archive_hmac_private.h
  archive_pack_dev.h
  archive_pack_dev.c
  archive_pathmatch.c
  archive_pathmatch.h
  archive_platform.h
  archive_private.h
  archive_random.c
  archive_random_private.h
  archive_rb.c
  archive_rb.h
  archive_read.c
  archive_read_add_passphrase.c
  archive_read_append_filter.c
  archive_read_data_into_fd.c
  archive_read_disk_private.h
  archive_read_open_fd.c
  archive_read_open_file.c
  archive_read_open_filename.c
  archive_read_open_memory.c
  archive_read_private.h
  archive_read_set_format.c
  archive_read_support_format_zip.c
  archive_string.c
  archive_string.h
  archive_string_composition.h
  archive_string_sprintf.c
  archive_util.c
  archive_version_details.c
  archive_virtual.c
  archive_write.c
  archive_write_disk_private.h
  archive_write_private.h
  archive_write_open_fd.c
  archive_write_open_file.c
  archive_write_open_filename.c
  archive_write_open_memory.c
  archive_write_add_filter.c
  archive_write_add_filter_gzip.c
  archive_write_set_format.c
  archive_write_set_format_zip.c
  archive_write_set_passphrase.c
)

# archive_static is a static library
ADD_LIBRARY(archive_static STATIC ${libarchive_SOURCES} ${include_HEADERS})
TARGET_LINK_LIBRARIES(archive_static ${ADDITIONAL_LIBS})
SET_TARGET_PROPERTIES(archive_static PROPERTIES COMPILE_DEFINITIONS
  LIBARCHIVE_STATIC)
# On Posix systems, libarchive.so and libarchive.a can co-exist.
IF(NOT WIN32 OR CYGWIN)
  SET_TARGET_PROPERTIES(archive_static PROPERTIES OUTPUT_NAME archive)
ENDIF(NOT WIN32 OR CYGWIN)

IF(ENABLE_INSTALL)
  # How to install the libraries
  INSTALL(TARGETS archive_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
  # INSTALL(FILES ${include_HEADERS} DESTINATION include)
ENDIF()
