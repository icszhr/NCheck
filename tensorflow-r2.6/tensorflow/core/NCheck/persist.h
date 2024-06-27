// PersistenceManager.h
#ifndef PERSISTENCE_MANAGER_H_
#define PERSISTENCE_MANAGER_H_

#include <libpmem.h>
#include <cstring>
#include <iostream>
#include <string>
#include <mutex>
#include <cstdint>
#include "tensorflow/core/platform/logging.h"

struct TensorMetadata {
    size_t offset;
    size_t size;
    char name[256];
    char dtype[50];
    char shape[100];
    int t_id;
};

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

struct ModelStateLog {
    uint8_t epoch;
    uint8_t batch;
    uint16_t op;
    uint8_t learningRate; 
    uint8_t loss;    
    uint16_t state;         
};

class PersistenceManager {
public:
    static std::mutex metadata_mutex;
    static size_t next_metadata_offset;
    static std::vector<ModelStateLog> model_state_buffer;
    static size_t buffer_size_limit;
    int t_id;

    // Initializes the area for metadata and provides offsets for tensor data
    static void InitializePersistentMemory(void* base);

    // Saves tensor metadata into the persistent memory using preallocated space
    static void SaveTensorMetadata(void* base, void* data_ptr, const std::string& dtype, const std::string& shape, const std::string& name, int count);

    // Loads metadata for tensor based on the persisted memory location
    static void* LoadTensorData(void* base, const std::string& name, int count);

    // Saves model state into the persistent memory using preallocated space
    static void SaveModelState(void* base, uint8_t epoch, uint8_t batch, uint16_t op, uint8_t learningRate, uint8_t loss, uint16_t state);

    // Loads metadata for model state based on the persisted memory location
    static void LoadModelState(void* base, std::vector<ModelStateLog>& logs);

    static void FlushModelStateLogs(void* base);
};

#endif  // PERSISTENCE_MANAGER_H_
