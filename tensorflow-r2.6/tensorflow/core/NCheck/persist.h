// PersistenceManager.h
#ifndef PERSISTENCE_MANAGER_H_
#define PERSISTENCE_MANAGER_H_

#include <libpmem.h>
#include <cstring>
#include <iostream>
#include <string>
#include <mutex>

struct PersistenceHeader {
    size_t tensor_metadata_offset;
    size_t model_state_offset;
    size_t tensor_metadata_size;
    size_t model_state_size;
    size_t tensor_data_offset;  // Offset for the actual tensor data
    size_t num_tensors;   
    size_t *tensor_metadata_offsets;
    size_t *tensor_metadata_sizes;
};

class PersistenceManager {
public:
    static std::mutex metadata_mutex;
    static size_t next_metadata_offset;

    // Initializes the area for metadata and provides offsets for tensor data
    static void InitializePersistentMemory(void* base);

    // Saves tensor metadata into the persistent memory using preallocated space
    static void SaveTensorMetadata(void* base, const void* metadata, size_t metadata_size);

    // Loads metadata for tensor based on the persisted memory location
    static void LoadTensorMetadata(void* base, size_t tensor_index, void** metadata, size_t* metadata_size);

    // Saves model state into the persistent memory using preallocated space
    static void SaveModelState(void* base, const void* state, size_t size);

    // Loads metadata for model state based on the persisted memory location
    static void LoadModelState(void* base, void** state, size_t* state_size);
};

#endif  // PERSISTENCE_MANAGER_H_
