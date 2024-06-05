#include <string.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "logger.h"
#include "scal_utilities.h"
#include "gaudi2_arc_host_packets.h"

bool loadFWImageFromFile_gaudi2(
    int fd,
    const std::string &binFileName,
    unsigned& dccmImageSize,
    unsigned& hbmImageSize,
    uint8_t * pDccm,
    uint8_t * hbm,
    struct arc_fw_metadata_t* meta)
{
    static const char *c_fw_bin_file_extension = ".bin";
    char* dccm = (char*)pDccm;

    size_t extStartPos = binFileName.rfind(c_fw_bin_file_extension);

    // verify the bin file extension
    if ((extStartPos == std::string::npos) ||
        (extStartPos != binFileName.size() - std::string(c_fw_bin_file_extension).size()))
    {
        // illegal file name extension
        LOG_ERR(SCAL,"{}: fd={} illegal file name extension {}", __FUNCTION__, fd, binFileName.c_str());
        assert(0);
        return false;
    }

    std::ifstream image(binFileName, std::ios::binary);
    if (!image)
    {
        // file not found
        LOG_ERR(SCAL,"{}: fd={} file not found {}", __FUNCTION__, fd, binFileName.c_str());
        assert(0);
        return false;
    }

    image.seekg(0, std::ios::end);
    unsigned imageSize = image.tellg();
    image.seekg(0, std::ios::beg);
    struct arc_fw_metadata_t fwMetaData;
    unsigned metaDataSize = sizeof(fwMetaData);
    if (imageSize < metaDataSize)
    {
        LOG_ERR(SCAL,"{}: fd={} wrong file size {}", __FUNCTION__, fd, binFileName.c_str());
        assert(0);
        return false;
    }
    image.read((char*)&fwMetaData, metaDataSize);
    bool bMetaFound = false;
    // check for scheduler or engine uuid
    if ((fwMetaData.uuid[0] == SCHED_FW_UUID_0) && (fwMetaData.uuid[1] == SCHED_FW_UUID_1) &&
        (fwMetaData.uuid[2] == SCHED_FW_UUID_2) && (fwMetaData.uuid[3] == SCHED_FW_UUID_3))
    {
        // scheduler meta data
        memcpy(meta,&fwMetaData,metaDataSize);
        LOG_DEBUG(SCAL,"{}: scheduler meta data. hbm section (size,offset) {} {} dccm section  (size,offset) {} {} in {}", __FUNCTION__,
                fwMetaData.hbm_section_size, fwMetaData.hbm_section_offset,
                fwMetaData.dccm_section_size, fwMetaData.dccm_section_offset,
                binFileName);

        image.seekg(fwMetaData.dccm_section_offset, std::ios::beg);
        bMetaFound = true;
    }
    else
    if ((fwMetaData.uuid[0] == ENG_FW_UUID_0) && (fwMetaData.uuid[1] == ENG_FW_UUID_1) &&
        (fwMetaData.uuid[2] == ENG_FW_UUID_2) && (fwMetaData.uuid[3] == ENG_FW_UUID_3))
    {
        //engine meta data
        memcpy(meta,&fwMetaData,metaDataSize);
        LOG_DEBUG(SCAL,"{}: engine meta data. hbm section (size,offset) {} {} dccm section  (size,offset) {} {}", __FUNCTION__,
                fwMetaData.hbm_section_size, fwMetaData.hbm_section_offset,
                fwMetaData.dccm_section_size, fwMetaData.dccm_section_offset);
        image.seekg(fwMetaData.dccm_section_offset, std::ios::beg);
        bMetaFound = true;
    }
    if (bMetaFound)
    {
        // TBD update dccmImageSize to fwMetaData.dccm_section_size and use it when sending image to device
        if (fwMetaData.specs_version != ARC_FW_INIT_CONFIG_VER)
        {
             LOG_ERR(SCAL,"{}: fd={} mismatch in fw metadata specs version (meta {} specs {} in {})", __FUNCTION__, fd,
                     fwMetaData.specs_version, ARC_FW_INIT_CONFIG_VER, binFileName);
             assert(0);
             return false;
         }
         if ((fwMetaData.dccm_section_size > dccmImageSize) || (fwMetaData.hbm_section_size > hbmImageSize))
         {
             LOG_ERR(SCAL,"{}: fd={} illegal meta section sizes (dccm {} hbm {}) in {}", __FUNCTION__, fd,
                     fwMetaData.dccm_section_size, fwMetaData.hbm_section_size, binFileName);
             assert(0);
             return false;
         }
        dccmImageSize = fwMetaData.dccm_section_size;
        hbmImageSize = fwMetaData.hbm_section_size;
        image.read(dccm, dccmImageSize);
    }
    else
    {   // no meta
        image.seekg(0, std::ios::beg);
        if (imageSize != dccmImageSize + hbmImageSize)
        {
            // bin file size mismatch
            LOG_ERR(SCAL,"{}: fd={} imageSize {} != dccmImageSize + hbmImageSize {} in {}", __FUNCTION__, fd,
                    imageSize, dccmImageSize + hbmImageSize, binFileName);
            assert(0);
            return false;
        }
        image.read((char*)dccm, dccmImageSize);
    }

    if (isSimFD(fd))
    {
        // construct the bfm file name
        // <path>/lib<imageName>.so
        size_t pathEndPos = binFileName.rfind("/");
        pathEndPos = (pathEndPos == std::string::npos) ? 0 : pathEndPos + 1;

        std::string bfmFile =
            binFileName.substr(0, pathEndPos) + "lib" + binFileName.substr(pathEndPos, extStartPos - pathEndPos) + "_bfm.so";

        // verify that the BFM file exists
        if (!fileExists(bfmFile))
        {
            // bfm file not found
            LOG_ERR(SCAL,"{}: fd={} Failed to load  bfmFile {}", __FUNCTION__, fd, bfmFile);
            assert(0);
            return false;
        }

        *((uint32_t*)hbm) = bfmFile.size();
        strcpy((char*)hbm + sizeof(uint32_t), bfmFile.c_str());
    }
    else
    {
        image.read((char*)hbm, hbmImageSize);
    }

    return true;
}
