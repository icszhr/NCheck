#ifndef TENSORFLOW_CORE_FRAMEWORK_NVM_ALLOCATOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_NVM_ALLOCATOR_H_

#include <mutex>
#include <unordered_map>
#include <string>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/types.h"

class NVMAllocator : public tensorflow::Allocator {
public:
    static NVMAllocator* GetInstance();
    std::string Name() override;
    void* AllocateRaw(size_t alignment, size_t num_bytes) override;
    void DeallocateRaw(void* ptr) override;
    size_t GetAllocatedSize(const void* ptr) const;
    void ResetOffsetForIteration();
    void* GetBaseAddress();

private:
    static NVMAllocator* instance_;
    static char* pmem_base_;
    static size_t mapped_len_;
    static size_t offset_;
    static size_t reserved_meta_offset_;
    static std::mutex mu_;
    static tensorflow::AllocatorStats stats_;
    static std::unordered_map<void*, size_t> allocation_map_;  // Map to track allocation sizes

    NVMAllocator();
    ~NVMAllocator() override;
    void Init();
};

#endif // TENSORFLOW_CORE_FRAMEWORK_NVM_ALLOCATOR_H_
