#include "persist.h"
#include "shader.h"
#include "gpu-cache.h"
#include "../option_parser.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"

#define pb_printf(...) printf("%s (%llu):", isL1Buffer ? m_name : "L2", m_gpu->gpu_sim_cycle); printf(__VA_ARGS__);

//#define PB_DEBUG
#ifdef PB_DEBUG
#define PRINT_DEBUG(...) pb_printf(__VA_ARGS__);
#define QPRINT_DEBUG(...) printf(__VA_ARGS__);
#else
#define PRINT_DEBUG(...)
#define QPRINT_DEBUG(...)
#endif

int pm_model;
uint32_t l1BufferSize, l2BufferSize;
persist_buffer m_L2PBuffer(false, NULL);
uint32_t useL2Buffer, forceWarpWait, deprioritise, deep_flush, eagerFlush, adrEnabled, outstandingFlushes;
std::map<int, int> num_coalesced;
uint64_t pBufferReordered = 0;
uint64_t pBufferNotReordered = 0;
uint32_t pmScaleDownRead;
uint32_t pmScaleDownWrite;

std::map<int, std::set<int> > delayed_warps;
std::map<int, std::set<int> > durable_warps;
std::map<int, std::set<int> > stalled_warps;

addr_t norm_addr(addr_t addr) { return addr - (addr % 128); }

void pm_reg_options(class OptionParser* opp) {
  option_parser_register(opp, "-pm_model", OPT_INT32, &pm_model,
        "Type of persistent memory model to use (0=none,1=seq,2=epoch,3=unbuff_srp,4=buff_srp_integ,5=buff_srp_disc)", "0");
  option_parser_register(opp, "-pm_l1_size", OPT_UINT32, &l1BufferSize,
        "Size of l1 persist buffer", "512");
  option_parser_register(opp, "-pm_l2_size", OPT_UINT32, &l2BufferSize,
        "Size of l2 persist buffer", "0");
  option_parser_register(opp, "-pm_use_l2", OPT_UINT32, &useL2Buffer,
        "Switch to turn on L2 buffer", "0");
  option_parser_register(opp, "-pm_force_wait", OPT_UINT32, &forceWarpWait,
        "Switch to force warps to wait on fence", "0");
  option_parser_register(opp, "-pm_deprioritise", OPT_UINT32, &deprioritise,
        "Switch to deprioritise warps on fence", "1");
  option_parser_register(opp, "-pm_deep_flush", OPT_UINT32, &deep_flush,
        "Whether to deep flush on kernel end", "0");
  option_parser_register(opp, "-pm_eager_flush", OPT_UINT32, &eagerFlush,
        "Whether to eager flush on a persistent entry", "1");
  option_parser_register(opp, "-pm_enable_adr", OPT_UINT32, &adrEnabled,
        "Whether to assume persistence on reaching DRAM queue", "1");
  option_parser_register(opp, "-pm_outstanding_flushes", OPT_UINT32, &outstandingFlushes,
        "Maximum amount of outstanding flushes allowed", "16");
  option_parser_register(opp, "-pm_bw_scale_read", OPT_UINT32, &pmScaleDownRead,
        "Factor by which to scale down GDDR read BW for PM", "4");
  option_parser_register(opp, "-pm_bw_scale_write", OPT_UINT32, &pmScaleDownWrite,
        "Factor by which to scale down GDDR write BW for PM", "8");
  
}

// Helper to check whether an address lies in PM
bool is_pm(addr_t addr) {
  return (addr >= GLOBAL_PM_START);
}

bool pm_warp_stalled(int sid, int warpId) {
    return (stalled_warps[sid].find(warpId) != stalled_warps[sid].end());
}

void pm_warps_clear(int sid) {
    stalled_warps[sid].clear();
}

// Implement delay due to PM write here
bool shader_core_ctx::warp_waiting_for_persist(unsigned warp_id) {
    switch(pm_model) {
        case PM_NONE:
            break;
        case PM_SEQUENTIAL:
        case PM_EPOCH:
        case PM_UNBUFF_SRP:
            // All outstanding persists done. Clear for this warp
            if(m_warp[warp_id]->persists_done())
                m_warp[warp_id]->clear_waiting_persist();
            // Return whether waiting on persist
            return m_warp[warp_id]->get_waiting_persist();
            break;
        case PM_BUFF_SRP_INTEG:
        case PM_BUFF_SRP_DISC:
            // I don't remember what this was for
            if(delayed_warps[get_sid()].find(warp_id % WARPS_PER_SM) != delayed_warps[get_sid()].end())
                return true;
            if(durable_warps[get_sid()].find(warp_id % WARPS_PER_SM) != durable_warps[get_sid()].end())
                return true;
            // Check if waiting for release op
            return m_warp[warp_id]->get_waiting_persist() || pm_warp_stalled(get_sid(), warp_id);
            break;
    }
    return false;
}

bool persist_buffer::checkAddEntry(addr_t address, int warpId)
{
    // No issues with non-PM addresses
    if(!is_pm(address))
        return true;

    // Not using l2 buffer, so shouldn't reach here
    if(!useL2Buffer && !isL1Buffer)
        assert(false);

    // Address already exists in buffer
    if(hasEntry(address)) {
        //PRINT_DEBUG("Buffer already has this entry\n");
        // Ordering bit set after entry
        // Have to flush before inserting
        bool orderAfter  = hasOrderAfter(entryMap[address], warp_mask_t().set(warpId % WARPS_PER_SM));
        auto end = bufferEntries.end();
        --end;
        bool orderBefore = hasOrderBefore(end, warp_mask_t().set(warpId % WARPS_PER_SM), entryMap[address]);
        if(orderAfter || orderBefore) {
            ++pBufferNotReordered;
            PRINT_DEBUG("Cannot insert for warp %d, entry %llx exists and is ordered (before? %d, after? %d)\n", warpId, address, orderAfter, orderBefore);
            if(m_sid != -1) {
                delayed_warps[m_sid].insert(warpId % WARPS_PER_SM);
                //PRINT_DEBUG("Delaying warp %d (%d, %d)\n", warpId, m_sid, warpId % WARPS_PER_SM);
            }
            beginFlush(warp_mask_t().set(warpId % WARPS_PER_SM));
            if(m_sid != -1) {
                flushingWarps.set(warpId % WARPS_PER_SM);
            }
            return false;
        }
        ++pBufferReordered;
        // No ordering bit after. Can directly modify value.
        return true;
    }

    //PRINT_DEBUG("Checking if we can add entry %llx for warpId %d\n", address, warpId);
    // Buffer full 
    if(bufferSize != 0 && bufferEntries.size() >= bufferSize) {
        // Aready flushing, just wait until slots opens up
        if(flushingWarps.count() > 0)
            return false;
        // Otherwise flush entries to free up a slot
        beginFlush(warp_mask_t().set(warpId % WARPS_PER_SM), true);
        return bufferEntries.size() < bufferSize;
    }
    return true;
}

void persist_buffer::addEntry(addr_t address, data_cache *cache, int warp)
{
    if(!is_pm(address))
        return;
        
    if(hasEntry(address)) {
        entryMap.find(address)->second->warpIds.set(warp % WARPS_PER_SM);
        PRINT_DEBUG("Merged entry for address %llx for warpId %d\n", address, warp);
        return;
    }

    PRINT_DEBUG("Added entry at %llx for warpId %d\n", address, warp);
    // Create new entry
    persist_buffer::bufferEntry entry;
    entry.address = address;
    entry.orderBit = false;
    entry.releaseBit = false;
    entry.u.cache = cache;
    entry.warpIds.set(warp % WARPS_PER_SM);
    
    entryMap[address] = bufferEntries.insert(bufferEntries.end(), entry);
    stats_max_buffer_size = std::max((int)bufferEntries.size(), stats_max_buffer_size);
    if(eagerFlush) 
        beginFlush(warp_mask_t().set(warp % WARP_PER_BUFF_ENTRY));
}

void persist_buffer::setOrder(int warpId, bool delayWarp)
{
    bool complete = false;
    for(auto i = bufferEntries.rbegin(); i != bufferEntries.rend(); ++i) {
        // Merge order entry
        if(!i->releaseBit && i->orderBit && i->durable == delayWarp) {
            //PRINT_DEBUG("Warp %d, merged order bit\n", warpId);
            i->warpIds.set(warpId % WARPS_PER_SM);
            complete = true;
            break;
        }
        // Cant merge if intervening write exists
        else if(i->warpIds.test(warpId % WARPS_PER_SM))
            break;
    }
    if(!complete) {
        //PRINT_DEBUG("Warp %d, added order bit\n", warpId);
        persist_buffer::bufferEntry entry;
        entry.address = 0; // Dummy, doesnt really matter
        entry.orderBit = true;
        entry.releaseBit = false;
        entry.durable = delayWarp;
        entry.u.warp = NULL;
        entry.warpIds.set(warpId % WARPS_PER_SM);
        bufferEntries.insert(bufferEntries.end(), entry);
        stats_max_buffer_size = std::max((int)bufferEntries.size(), stats_max_buffer_size);
    }
    if(delayWarp || forceWarpWait) {
        durable_warps[m_sid].insert(warpId % WARPS_PER_SM);
        PRINT_DEBUG("Delaying warp %d (%d, %d)\n", warpId, m_sid, warpId % WARPS_PER_SM);
        beginFlush(warp_mask_t().set(warpId % WARPS_PER_SM));
    } else if(deprioritise) {
        stalled_warps[m_sid].insert(warpId % WARPS_PER_SM);
        PRINT_DEBUG("Stalling warp %d (%d, %d)\n", warpId, m_sid, warpId % WARPS_PER_SM);
        beginFlush(warp_mask_t().set(warpId % WARPS_PER_SM));
    }
}

void persist_buffer::deepFlush(shd_warp_t *warp, int sid, warp_mask_t warpIds)
{
    int id = warp->get_warp_id();
    if(!isL1Buffer) {
        if(!useL2Buffer)
            return;
        id = sid;
    }
    PRINT_DEBUG("Start deep flush for %d\n", id);

    bool complete = false;
    for(auto i = bufferEntries.rbegin(); i != bufferEntries.rend(); ++i) {
        // Merge order entry
        if(!i->releaseBit && i->orderBit && !i->acquireBit && i->durable) {
            if(isL1Buffer)
                if(warpIds.count() == 0)
                    for(auto warpNum = 0u; warpNum < WARPS_PER_SM; ++warpNum)
                        i->warpIds.set(warpNum % WARPS_PER_SM);
                else
                    i->warpIds = (i->warpIds | warpIds);
            else {
                i->warpIds.set(sid);
            }
            complete = true;
            break;
        }
        // Cant merge if intervening write exists
        else if(i->warpIds.test(id % WARPS_PER_SM))
            break;
    }

    if(!complete) {
        persist_buffer::bufferEntry entry;
        entry.address = 0; // Dummy, doesnt really matter
        entry.orderBit = true;
        entry.releaseBit = false;
        entry.acquireBit = false;
        entry.u.warp = warp;
        entry.durable = true;
        if(isL1Buffer)
            if(warpIds.count() == 0)
                for(auto i = 0u; i < WARPS_PER_SM; ++i)
                    entry.warpIds.set(i);
            else
                entry.warpIds = (entry.warpIds | warpIds);
        else {
            entry.warpIds.set(sid);
        }
        bufferEntries.insert(bufferEntries.end(), entry);
        stats_max_buffer_size = std::max((int)bufferEntries.size(), stats_max_buffer_size);
    }
    // Wait for release op to complete
    if(isL1Buffer) {
        durable_warps[m_sid].insert(id % WARPS_PER_SM);
        PRINT_DEBUG("Deepflush, Delaying warp %d (%d, %d, %llx)\n", id, m_sid, id % WARPS_PER_SM, warpIds.to_ullong());
    }
    // Start flushing until release op
    beginFlush(warp_mask_t().set(id % WARPS_PER_SM));
}


void persist_buffer::acquireOp(shd_warp_t *warp, bool haltWarp, int sid) 
{
    int id = warp->get_warp_id();
    if(!isL1Buffer) {
        if(!useL2Buffer)
            return;
        id = sid;
    }

    bool complete = false;
    for(auto i = bufferEntries.rbegin(); i != bufferEntries.rend(); ++i) {
        PRINT_DEBUG("Merged acquireOp for %d\n", id);
        // Merge order entry
        if(i->acquireBit && haltWarp == i->durable) {
            i->warpIds.set(id % WARPS_PER_SM);
            complete = true;
            break;
        }
        // Cant merge if intervening write exists
        else if(i->warpIds.test(id % WARPS_PER_SM))
            break;
    }

    if(!complete) {
        PRINT_DEBUG("Added releaseOp for %d\n", id);
        persist_buffer::bufferEntry entry;
        entry.address = 0; // Dummy, doesnt really matter
        entry.orderBit = false;
        entry.acquireBit = true;
        entry.durable = haltWarp;
        entry.u.warp = warp;
        entry.warpIds.set(id % WARPS_PER_SM);
        bufferEntries.insert(bufferEntries.end(), entry);
        stats_max_buffer_size = std::max((int)bufferEntries.size(), stats_max_buffer_size);
    }
    // Wait for release op to complete
    if(haltWarp || forceWarpWait) {
        durable_warps[m_sid].insert(id % WARPS_PER_SM);
        PRINT_DEBUG("Delaying warp %d (%d, %d)\n", warp->get_warp_id(), m_sid, id % WARPS_PER_SM);
    } else if(deprioritise) {
        stalled_warps[m_sid].insert(id % WARPS_PER_SM);
        PRINT_DEBUG("Stalling warp %d (%d, %d)\n", warp->get_warp_id(), m_sid, id % WARPS_PER_SM);
    }
    // Start flushing until release op
    beginFlush(warp_mask_t().set(id % WARPS_PER_SM));
}

// Untested. Check with some benchmarks
void persist_buffer::releaseOp(shd_warp_t *warp, bool haltWarp, int sid) 
{
    int id = warp->get_warp_id();
    if(!isL1Buffer) {
        if(!useL2Buffer)
            return;
        id = sid;
    }

    bool complete = false;
    for(auto i = bufferEntries.rbegin(); i != bufferEntries.rend(); ++i) {
        PRINT_DEBUG("Merged releaseOp for %d\n", id);
        // Merge order entry
        if(i->releaseBit && haltWarp == i->durable) {
            i->warpIds.set(id % WARPS_PER_SM);
            complete = true;
            break;
        }
        // Cant merge if intervening write exists
        else if(i->warpIds.test(id % WARPS_PER_SM))
            break;
    }

    if(!complete) {
        PRINT_DEBUG("Added releaseOp for %d\n", id);
        persist_buffer::bufferEntry entry;
        entry.address = 0; // Dummy, doesnt really matter
        entry.orderBit = false;
        entry.releaseBit = true;
        entry.durable = haltWarp;
        entry.u.warp = warp;
        entry.warpIds.set(id % WARPS_PER_SM);
        bufferEntries.insert(bufferEntries.end(), entry);
        stats_max_buffer_size = std::max((int)bufferEntries.size(), stats_max_buffer_size);
    }
    // Wait for release op to complete
    if(haltWarp || forceWarpWait) {
        durable_warps[m_sid].insert(id % WARPS_PER_SM);
        PRINT_DEBUG("Delaying warp %d (%d, %d)\n", warp->get_warp_id(), m_sid, id % WARPS_PER_SM);
    } else if(deprioritise) {
        stalled_warps[m_sid].insert(id % WARPS_PER_SM);
        PRINT_DEBUG("Stalling warp %d (%d, %d)\n", warp->get_warp_id(), m_sid, id % WARPS_PER_SM);
    }
    // Start flushing until release op
    beginFlush(warp_mask_t().set(id % WARPS_PER_SM));
}

bool persist_buffer::checkEvict(addr_t address) 
{
    // No issues with non-PM addresses
    if(!is_pm(address))
        return true;
    
    if(!isL1Buffer && !useL2Buffer)
        return true;

    PRINT_DEBUG("Checking evict %llx: ", address);
    // Address has probably already been written back
    if(!hasEntry(address)) {
        //printf("Can evict, address written back\n");
        QPRINT_DEBUG("Can evict, address already written back\n");
        return true;
    }

    // Flush in progress, can't evict before these finish
    if((entryMap[address]->warpIds & flushingWarps).any()) {
        //printf("Flush in progress, cannot evict\n");
        QPRINT_DEBUG("Flush in progress, cannot evict\n");
        return false;
    }

    // Order bit set before entry, wait for this to finish flushing
    if(hasOrderBefore(entryMap[address], entryMap[address]->warpIds, bufferEntries.begin())) {
        QPRINT_DEBUG("Has order before, cannot evict\n");
        beginFlush(warp_mask_t(), true);
        ++pBufferNotReordered;
        return false;
    }

    ++pBufferReordered;
    PRINT_DEBUG("Can evict %llx\n", address);
    return true;
}


void persist_buffer::evict(addr_t address, int num) 
{
    // No issues with non-PM addresses
    if(!is_pm(address) || num == 0)
        return;
    
    if(!isL1Buffer && !useL2Buffer)
        return;

    // Address has probably already been written back
    if(!hasEntry(address))
        return;

    flushedEntries[norm_addr(address)] += num;
    PRINT_DEBUG("Evicted %llx: %lu outstanding flushes now (%d for addr)\n", address, flushedEntries.size(), num);
    // Erase evicted line
    auto entry = entryMap[address];
    bufferEntries.erase(entry);
    entryMap.erase(address);
}

void persist_buffer::flushesComplete() {
    // Force ordering at L2 because of normal order bit
    if(isL1Buffer && orderAfterFlush && useL2Buffer)
        l2Buffer->setOrder(m_sid);

    if(invalidateAfterFlush)
        m_cache->invalidate(true);

    orderAfterFlush = false;
    invalidateAfterFlush = false;    

    warp_mask_t copy = flushingWarps;
    flushingWarps.reset();

    // Continue next round of flushes
    if(hasOrderAfter(bufferEntries.begin(), warp_mask_t().set()) || (eagerFlush && bufferEntries.size() > 0))
        beginFlush(copy);

    if(m_sid != -1)
        for(auto i = 0u; i < copy.size(); ++i)
            if(copy.test(i) && !flushingWarps.test(i)) {
                delayed_warps[m_sid].erase(i);
                PRINT_DEBUG("Clearing warp (%d, %d)\n", m_sid, i);
            }
}

void persist_buffer::ackFlush(addr_t addr, int numAcks)
{
    if(!isL1Buffer && !useL2Buffer)
        return;
    
    // Decrement ACK count
    int num = (flushedEntries[norm_addr(addr)] -= numAcks);
    if(num <= 0) {
        flushedEntries.erase(norm_addr(addr));
        if(eagerFlush)
            beginFlush(warp_mask_t());
    }
    PRINT_DEBUG("Acked %d %llx flush, %lu entries left (%d for addr)\n", numAcks, norm_addr(addr), flushedEntries.size(), num);
    // No more outstanding flushes
    if(flushedEntries.size() == 0)
        flushesComplete();
}

// Check for ordering/release bit after given entry
// Remember, a release is a one-way fence, so releases after enforce order
// But releases before don't enforce order
bool persist_buffer::hasOrderAfter(std::list<persist_buffer::bufferEntry>::iterator it, warp_mask_t warpIds)
{
    for(auto i = it; i != bufferEntries.end(); ++i) {
        if(((i->orderBit || i->releaseBit) && (i->warpIds & warpIds).any()))
            return true;
    }
    return false;
}

// Check for ordering bit at or before given entry
bool persist_buffer::hasOrderBefore(std::list<persist_buffer::bufferEntry>::iterator it, warp_mask_t warpIds, 
    std::list<persist_buffer::bufferEntry>::iterator end)
{
    for(auto i = it; ; --i) {
        if((i->orderBit || i->acquireBit) && (i->warpIds & warpIds).any())
            return true;
        if(i == end)
            break;
    }
    return false;
}

void persist_buffer::beginFlush(warp_mask_t warps, bool ignoreLimit)
{
    if(!isL1Buffer && !useL2Buffer)
        return;

    // Already waiting on max limit of flushes
    if(outstandingFlushes && flushedEntries.size() > outstandingFlushes && !ignoreLimit)
        return;

    // Already flush in progress
    if((flushingWarps & warps).any() && flushedEntries.size() > 0) {
        PRINT_DEBUG("Do not flush, already in progress, with %lu outstanding entries\n", flushedEntries.size());
        return;
    }

    // Don't flush on kernel end
    if(isL1Buffer && activeWarps == 0 && !ignoreLimit) {
        PRINT_DEBUG("Do not flush, kernel has ended\n");
        return;
    }
    
    //PRINT_DEBUG("Begin flushing\n");
    for(auto i = bufferEntries.begin(); i != bufferEntries.end();) {
        if((i->warpIds & flushingWarps).any() && !i->orderBit && !i->releaseBit && !i->acquireBit && !i->durable) {
            PRINT_DEBUG("Stop flush for %lx, orderbit=%d, release=%d, mask=%llx\n", i->address, i->orderBit, i->releaseBit, i->warpIds.to_ullong());
            break;
        }
        // Normal entry, flush from cache
        if(!i->orderBit && !i->releaseBit && !i->acquireBit && !i->durable) {
            int num = i->u.cache->writeback(i->address);
            if(num > 0) {
                flushedEntries[norm_addr(i->address)] += num;
                PRINT_DEBUG("Flushed address %llx (%d entries)\n", norm_addr(i->address), num);
            }
            entryMap.erase(i->address);
            if(outstandingFlushes && flushedEntries.size() > outstandingFlushes) {
                auto j = i;
                ++i;
                bufferEntries.erase(j);
                break;
            }
        }

        // Release entry, stop delaying warp waiting for release
        if(i->releaseBit) {
            PRINT_DEBUG("Found release entry for (%u, %u)\n", i->u.warp->get_dynamic_warp_id(), i->u.warp->get_warp_id());
            // If l1 buffer, insert release entry into L2 buffer
            if(isL1Buffer && useL2Buffer) {
                m_L2PBuffer.releaseOp(i->u.warp, false, m_sid);
            }
            flushingWarps = (flushingWarps | i->warpIds);
        }

        if(i->acquireBit) {
            PRINT_DEBUG("Found acquire entry for (%u, %u)\n", i->u.warp->get_dynamic_warp_id(), i->u.warp->get_warp_id());
            flushingWarps = (flushingWarps | i->warpIds);
            if(i->durable)
                invalidateAfterFlush = true;
        }

        if(i->orderBit) {
            PRINT_DEBUG("Order bit found for %llx\n", i->warpIds.to_ullong());
            // Order bit found. Propogate to lower caches
            orderAfterFlush = true;
            flushingWarps = (flushingWarps | i->warpIds);
        }

        if(i->durable) {
            if(m_sid != -1)
                for(auto j = 0u; j < i->warpIds.size(); ++j)
                    if(i->warpIds.test(j)) {
                        durable_warps[m_sid].erase(j);
                        delayed_warps[m_sid].insert(j);
                    }
        }
        // Delete entry from buffer and map
        auto j = i;
        ++i;
        bufferEntries.erase(j);
    }
    PRINT_DEBUG("Flushed cache. Waiting on %lu entries\n", flushedEntries.size());
    // No more outstanding flushes
    if(flushedEntries.size() == 0)
        flushesComplete();
}

void persist_buffer::warpStart(shd_warp_t *warp)
{ 
    warp->get_pm_flush_done() = false;
    ++activeWarps;
    curr_warps.insert(warp);
}

int persist_buffer::warpEnd(shd_warp_t *warp)
{
    if(warp->stores_created() == false)
        return true;
    warp->get_pm_flush_done() = true;
    --activeWarps;
    if(activeWarps == 0) {
        stats_flush_start_cycle = m_gpu->gpu_sim_cycle;
    }
    PRINT_DEBUG("Warp %u complete\n", warp->get_warp_id());
    curr_warps.erase(warp);
    switch(pm_model) {
        case PM_NONE:
        case PM_SEQUENTIAL:
            return false;
        case PM_EPOCH:
            warp->set_waiting_persist();
            return true;
        case PM_UNBUFF_SRP: // Handled directly at function call
            if(activeWarps == 0 && (flushedEntries.size() != 0 || bufferEntries.size() != 0) && deep_flush) {
                return true;
            }
            return false;
        case PM_BUFF_SRP_INTEG:
        case PM_BUFF_SRP_DISC:
            if(activeWarps == 0 && (flushedEntries.size() != 0 || bufferEntries.size() != 0) && deep_flush) {
                deepFlush(warp, true);
                return true;
            } else if(activeWarps == 0 && flushedEntries.size() != 0) {
                delayed_warps[m_sid].insert(warp->get_warp_id() % MAX_WARP_PER_SM);
                flushingWarps.set(warp->get_warp_id() % MAX_WARP_PER_SM);
                PRINT_DEBUG("Delay warp %u, waiting on %lu flushes\n", warp->get_warp_id(), flushedEntries.size());
                return true;
            }
            return false;
            break;
    }
    return false;
}

void persist_buffer::print_stats()
{
    if(stats_flush_start_cycle)
        printf("%s, Flush: %llu cycles, Max buffer size: %d\n", m_name, m_gpu->gpu_sim_cycle - stats_flush_start_cycle, stats_max_buffer_size);
    stats_max_buffer_size = 0;
    stats_flush_start_cycle = 0;
}

void persist_buffer::dump()
{
    if(bufferEntries.size() > 0) {
        pb_printf("In buffer: ");
        for(auto i = bufferEntries.begin(); i != bufferEntries.end(); ++i) {
            printf(" (A? %d, R? %d, O? %d, D? %d, Addr: %lx, Mask: %llx), ", i->acquireBit, i->releaseBit, i->orderBit, i->durable, i->address, i->warpIds.to_ullong());
        }
        printf("\n");
    }

    if(flushedEntries.size() > 0) {
        pb_printf("Remaining: ");
        for(auto i = flushedEntries.begin(); i != flushedEntries.end(); ++i) {
            printf(" address %llx (%d), ", i->first, i->second);
        }
        printf("\n");
    }
    
    if(durable_warps[m_sid].size() > 0) {
        pb_printf("Durable warps: ");
        for(auto i = durable_warps[m_sid].begin(); i != durable_warps[m_sid].end(); ++i) {
            printf(" %d, ", *i);
        }
        printf("\n");
    }

    if(delayed_warps[m_sid].size() > 0) {
        pb_printf("Delayed warps: ");
        for(auto i = delayed_warps[m_sid].begin(); i != delayed_warps[m_sid].end(); ++i) {
            printf(" %d, ", *i);
        }
        printf("\n");
    }
    
    if(stalled_warps[m_sid].size() > 0) {
        pb_printf("Stalled warps: ");
        for(auto i = stalled_warps[m_sid].begin(); i != stalled_warps[m_sid].end(); ++i) {
            printf(" %d, ", *i);
        }
        printf("\n");
    }

    if(curr_warps.size() > 0) {
        pb_printf("Delayed warps (not PM): ");
        for(auto i = curr_warps.begin(); i != curr_warps.end(); ++i) {
            printf("%u (W: %u, P: %u, SC: %u, SD: %u), ", (*i)->get_warp_id(), (*i)->waiting(), (*i)->get_waiting_persist(), (*i)->stores_created(), (*i)->stores_done());
        }
        printf("\n");
    }
}
