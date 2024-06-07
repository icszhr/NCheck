#include "tensorflow/core/NCheck/nvm_allocator_adapter.h"
#include "tensorflow/core/NCheck/nvm_allocator.h"

namespace tensorflow {

// Assume NVMAllocator::GetInstance() returns an INVMAllocator instance
Allocator* GetNVMAllocatorAdapter() {
    static NVMAllocatorAdapter* adapter = new NVMAllocatorAdapter(NVMAllocator::GetInstance());
    return adapter;
}

}  // namespace tensorflow