
#ifndef TENSORFLOW_CORE_NCHECK_NVM_ALLOCATOR_ADAPTER_H_
#define TENSORFLOW_CORE_NCHECK_NVM_ALLOCATOR_ADAPTER_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/NCheck/invm_allocator.h"  // Assuming INVMAllocator is the interface

namespace tensorflow {

class NVMAllocatorAdapter : public Allocator {
public:
    NVMAllocatorAdapter(INVMAllocator* base_allocator)
        : base_allocator_(base_allocator) {}

    Allocator* GetNVMAllocatorAdapter() {}

    std::string Name() override {
        return base_allocator_->Name();
    }

    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
        return base_allocator_->AllocateRaw(alignment, num_bytes);
    }

    void DeallocateRaw(void* ptr) override {
        base_allocator_->DeallocateRaw(ptr);
    }

private:
    INVMAllocator* base_allocator_;  // The actual NVM allocator implementation
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_NCHECK_NVM_ALLOCATOR_ADAPTER_H_