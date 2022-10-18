#include <list>
#include <algorithm>
#include <unordered_map>
#include <assert.h>
#include <set>
#include <map>
#include <cstring>
#include <bitset>
#include <utility>
#include <unordered_map>

#define WARPS_PER_SM (64U)
#define MAX_SM (100U)
#define WARP_PER_BUFF_ENTRY (WARPS_PER_SM)

#ifndef PERSIST_H
#define PERSIST_H

class data_cache;
class shd_warp_t;
class OptionParser;
class gpgpu_sim;
typedef unsigned long long addr_t;
typedef std::bitset<WARP_PER_BUFF_ENTRY> warp_mask_t;

extern int pm_model;
extern uint32_t l1BufferSize, l2BufferSize;
extern uint32_t useL2Buffer;
extern uint32_t adrEnabled;
extern std::map<int, int> num_coalesced;
extern uint64_t pBufferReordered;
extern uint64_t pBufferNotReordered;
extern uint32_t pmScaleDownRead;
extern uint32_t pmScaleDownWrite;

void pm_reg_options(class OptionParser* opp);
bool is_pm(addr_t addr);
bool pm_warp_stalled(int sid, int warpId);
void pm_warps_clear(int sid);

typedef enum {
    PM_NONE           = 0,
    PM_SEQUENTIAL     = 1,
    PM_EPOCH          = 2,
    PM_UNBUFF_SRP     = 3,
    PM_BUFF_SRP_INTEG = 4,
    PM_BUFF_SRP_DISC  = 5,
} pm_order_types;

class persist_buffer
{
public:
    persist_buffer(bool isL1, persist_buffer *l2PBuffer, gpgpu_sim *gpu = NULL, char *name = NULL, int sid = -1) {
        // Need to supply L2 buffer if is L1 buffer
        if(isL1Buffer)
            assert(l2PBuffer);
        
        isL1Buffer = isL1;
        l2Buffer   = l2PBuffer;
        orderAfterFlush = false;
        invalidateAfterFlush = false;
        flushingWarps.reset();
        m_gpu = gpu;
        if(name != NULL)
            strncpy(m_name, name, 1024);
        m_sid = sid;
        stats_max_buffer_size = 0;
        stats_flush_start_cycle = 0;
        stats_flush_cycles = 0;
        m_cache = NULL;
    }

    bool checkAddEntry(addr_t, int);
    void addEntry(addr_t, data_cache *, int);
    bool checkEvict(addr_t address);
    void evict(addr_t address, int num);

    inline bool hasEntry(addr_t address) {
        return entryMap.find(address) != entryMap.end();
    }

    void setOrder(int warpId, bool delayWarp = false);
    void acquireOp(shd_warp_t *warp, bool haltWarp = true, int sid = -1);
    void releaseOp(shd_warp_t *warp, bool haltWarp = true, int sid = -1);
    void deepFlush(shd_warp_t *warp, int sid = -1, warp_mask_t warpIds = warp_mask_t().set());

    void ackFlush(addr_t, int numAcks = 1);

    void setBufferSize(uint32_t size) { bufferSize = size; }
    void setCache(data_cache *cache) { m_cache = cache; }
    void setGPU(gpgpu_sim *gpu) { m_gpu = gpu; }

    bool hasSpace(uint warpId, uint amount = 1) { 
        if(!(bufferSize == 0 || bufferEntries.size() < bufferSize))
            beginFlush(warp_mask_t().set(warpId % WARPS_PER_SM), true);
        return (bufferSize == 0 || bufferEntries.size() < bufferSize);
    }

    // Used to flush all buffers at the end of kernel
    void warpStart(shd_warp_t *warp);
    int warpEnd(shd_warp_t *warp);

    // For debug
    void print_stats();
    void dump();
 
private:
    typedef struct buff_entry_t {
        uint64_t address;
        bool orderBit;
        bool acquireBit;
        bool releaseBit;
        bool durable; // In hardware, indicated by orderBit and releaseBit set
        struct {
            data_cache *cache;
            shd_warp_t *warp;
        } u;
        warp_mask_t warpIds;
        buff_entry_t() {
            warpIds.reset();
            durable = false;
            orderBit = false;
            acquireBit = false;
            releaseBit = false;
        }
    } bufferEntry;
    std::list<bufferEntry> bufferEntries;

    bool hasOrderAfter(std::list<bufferEntry>::iterator it, warp_mask_t warpIds);
    bool hasOrderBefore(std::list<bufferEntry>::iterator it, warp_mask_t warpIds, 
        std::list<persist_buffer::bufferEntry>::iterator end);

    void beginFlush(warp_mask_t, bool = false);
    void flushesComplete();

    std::unordered_map<addr_t, std::list<bufferEntry>::iterator> entryMap;
    uint32_t bufferSize;
    bool isL1Buffer;
    std::unordered_map<addr_t, int> flushedEntries;
    bool orderAfterFlush;
    bool invalidateAfterFlush;
    warp_mask_t flushingWarps;
    persist_buffer *l2Buffer;
    gpgpu_sim *m_gpu;
    char m_name[1024];
    int m_sid;
    int activeWarps;
    data_cache *m_cache;

    int stats_max_buffer_size;
    unsigned long long stats_flush_start_cycle;
    int stats_flush_cycles;

    std::set<shd_warp_t *> curr_warps;
};

extern persist_buffer m_L2PBuffer;

#endif /*PERSIST_H*/
