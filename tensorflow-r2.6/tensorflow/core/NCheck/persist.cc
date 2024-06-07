#include <libpmem.h>
#include <cstring>
#include <iostream>
#include <string>
#include <mutex>
#include "tensorflow/core/NCheck/persist.h"  // Assuming the header file is named PersistenceManager.h

std::mutex PersistenceManager::metadata_mutex;
size_t PersistenceManager::next_metadata_offset = 0;

void PersistenceManager::InitializePersistentMemory(void* base) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);
    header->tensor_metadata_offset = sizeof(PersistenceHeader);
    header->model_state_offset = header->tensor_metadata_offset + 1024 * 1024; // 1GB for tensor metadata
    header->tensor_data_offset = header->model_state_offset + 1024 * 1024; // Another 1MB reserved for model state
    header->num_tensors = 0;

    size_t max_tensors = 10000;
    header->tensor_metadata_offsets = (size_t*)(base_char + sizeof(PersistenceHeader));
    header->tensor_metadata_sizes = (size_t*)(base_char + sizeof(PersistenceHeader) + sizeof(size_t) * max_tensors);

    std::memcpy(base_char, &header, sizeof(header));
    pmem_persist(base_char, sizeof(header));
}

void PersistenceManager::SaveTensorMetadata(void* base, const void* metadata, size_t metadata_size) {
    std::lock_guard<std::mutex> lock(metadata_mutex);
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    if (next_metadata_offset == 0) {  // Initialize if first call
        InitializePersistentMemory(base);
        next_metadata_offset = header->tensor_metadata_offset;
    }

    void* metadata_addr = base_char + next_metadata_offset;
    std::memcpy(metadata_addr, metadata, metadata_size);
    pmem_persist(metadata_addr, metadata_size);

    // Record the offset and size of the current metadata block
    header->tensor_metadata_offsets[header->num_tensors] = next_metadata_offset;
    header->tensor_metadata_sizes[header->num_tensors] = metadata_size;
    header->num_tensors++;
    next_metadata_offset += metadata_size;  // Update offset for next entry
    header->tensor_metadata_size += metadata_size;  // Update total size of metadata
    pmem_persist(&(header->tensor_metadata_size), sizeof(header->tensor_metadata_size));
}


void PersistenceManager::LoadTensorMetadata(void* base, size_t tensor_index, void** metadata, size_t* metadata_size) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    if (tensor_index >= header->num_tensors) {
        std::cerr << "Tensor index out of range." << std::endl;
        return;
    }

    size_t tensor_offset = header->tensor_metadata_offsets[tensor_index];
    *metadata = base_char + tensor_offset;
    *metadata_size = header->tensor_metadata_sizes[tensor_index];
}


void PersistenceManager::SaveModelState(void* base, const void* state, size_t size) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    void* state_addr = base_char + header->model_state_offset;
    std::memcpy(state_addr, state, size);
    pmem_persist(state_addr, size);

    header->model_state_size = size;
    pmem_persist(&(header->model_state_size), sizeof(size));
}

void PersistenceManager::LoadModelState(void* base, void** state, size_t* state_size) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    *state = base_char + header->model_state_offset;
    *state_size = header->model_state_size;
}
