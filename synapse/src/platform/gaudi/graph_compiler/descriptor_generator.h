#pragma once

#include <mutex>

#include "types.h"
#include "gaudi_types.h"
#include "include/gaudi/mme_descriptor_generator.h"
#include "node_visitor.h"
#include "sync/sync_object_manager.h"
#include "types.h"
#include "node_roi.h"
#include "tpc_slice.h"
#include "tpc_slice_desc_update.h"

class GaudiCodeGenerator;

namespace gaudi
{
typedef std::map<MmeCommon::MmeLayerParams, std::list<MmeActivation>>::iterator descriptorCacheIt;

class DescriptorGenerator : public NodeVisitor
{
public:
    DescriptorGenerator(GaudiCodeGenerator* codeGenerator);

    virtual void visit(MmeNode* node) override;

    virtual void visit(TPCNode* node) override;
    virtual void visit(TPCSlice* node) override;

    virtual void visit(ConvolutionNode* node) override;
    virtual void visit(GEMMNode* node) override;
    virtual void visit(DeToDwNode* node) override;
    virtual void visit(DeToDxNode* node) override;
    virtual void visit(DMANode* node) override;
    virtual void visit(MmeTransposeNode* node) override;


    static void generateMmeDescriptor(const MmeNode& node, std::list<MmeActivation>& activations);

    static MmeCommon::EMmeOpType getOperationType(const MmeNode& node);
    static void                  generateTpcDescriptors(const TPCNode&                          node,
                                                        const std::list<NodeROI>&               rois,
                                                        deviceAddrOffset                        kernelAddr,
                                                        std::list<DescAndMask<gaudi::TpcDesc>>& descriptors);

    static void generateDmaDescriptors(const DMANode&                          node,
                                       const std::list<NodeROI>&               physicalRois,
                                       std::list<DescAndMask<gaudi::DmaDesc>>& descriptors,
                                       SyncObjectManager::SyncId               dummySyncObj);

    static MmeCommon::MmeStrategy getMmeStrategy(const MmeNode& node);

    static MmeCommon::MmeStrategy getMmeStrategy(const MmeNode&        node,
                                                 MmeCommon::EMmeOpType operationType,
                                                 synDataType           inputAElementType,
                                                 const SizeArray&      inputASizes,
                                                 const SizeArray&      outputSizes,
                                                 const std::string&    nodeNameForLog,
                                                 unsigned              packingFactor);

private:
    void addMmeDescriptorsToGraph(const MmeNode& node);

    static void patchTensorAddresses(std::list<MmeActivation>& activations,
                                     const pTensor& xTensor,
                                     const pTensor& wTensor,
                                     const pTensor& yTensor);
    void        addTpcDescriptorsToGraph(const TPCNode& node, const TPCSliceDescUpdate* updater = nullptr);
    void addDmaDescriptorsToGraph(const DMANode& node);

    GaudiCodeGenerator* m_codeGenerator;
};


class DescriptorsCache
{

public:
    static DescriptorsCache& get()
    {
        static DescriptorsCache instance;
        return instance;
    }

    //Clear descriptor's cache and set it max size
    void DescriptorsCacheInit(int size);
    //Generate descriptors
    void generateDescriptorsCache(const MmeCommon::MmeLayerParams& params, std::list<MmeActivation>& activations);
    //Get descriptor's cache max size in elements
    int getDesCacheMaxSize() { return m_csize;};
    //print descriptor's cache statistics
    void printDesCacheStats();
    //get descriptor's cache state (enabled/disabled)
    bool isDesCacheEnabled() const { return m_DesCacheEnabled; };
    //Get current cache size
    uint32_t getCacheSize() { return m_DesCacheMap.size();};
    //Return true if element in descriptor's cache, else false
    bool isElementInDesCache(const MmeCommon::MmeLayerParams& params, descriptorCacheIt& it);

private:
    DescriptorsCache();
    //Get descriptor from cache if found, else - generate new
    void getDesFromCacheOrGenNew(const MmeCommon::MmeLayerParams& params,
                                 std::list<MmeActivation>&        activations,
                                 bool                             isInCache,
                                 descriptorCacheIt                it);

    //Store iterator of cache
    std::list<descriptorCacheIt> m_dqDesCache;
    //Store references of key in cache
    std::map<MmeCommon::MmeLayerParams, std::list<MmeActivation>> m_DesCacheMap;
    //Maximum capacity of cache
    int m_csize = 0;
    //Descriptor's cache state (enabled/disabled)
    bool m_DesCacheEnabled = false;
    //Mutex to protect the cache
    std::recursive_mutex m_mutex;
    //Descriptors generation from cache count
    uint32_t m_cacheDesCount = 0;
    //New descriptors generation count
    uint32_t m_newDesGenCount = 0;
};

#define  DES_CACHE ::gaudi::DescriptorsCache::get()

}

