#ifndef TENSORFLOW_CORE_NCHECK_INVM_ALLOCATOR_H_
#define TENSORFLOW_CORE_NCHECK_INVM_ALLOCATOR_H_

#include <cstddef>
#include <string>

namespace tensorflow {

class INVMAllocator {
public:
    virtual ~INVMAllocator() {}

    virtual void* AllocateRaw(size_t alignment, size_t num_bytes) = 0;
    virtual void DeallocateRaw(void* ptr) = 0;
    virtual size_t GetAllocatedSize(const void* ptr) const = 0;
    virtual void* GetBaseAddress() = 0;
    virtual std::string Name() const = 0;
    virtual void ResetOffsetForIteration() = 0;

    // Factory method to get the singleton instance
    static INVMAllocator* GetInstance();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_NCHECK_INVM_ALLOCATOR_H_