#include <libpmem.h>
#include <cstring>
#include <iostream>
#include <string>
#include <mutex>
#include "tensorflow/core/NCheck/persist.h" 
#include <vector>
#include "tensorflow/core/platform/logging.h"

std::mutex PersistenceManager::metadata_mutex;
size_t PersistenceManager::next_metadata_offset = 0;
std::vector<TensorMetadata> metadata_registry;
std::vector<ModelStateLog> PersistenceManager::model_state_buffer;
size_t PersistenceManager::buffer_size_limit = 1024;


void PersistenceManager::InitializePersistentMemory(void* base) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = new(base_char) PersistenceHeader;
    header->tensor_metadata_offset = sizeof(PersistenceHeader);
    header->model_state_offset = header->tensor_metadata_offset + 1024 * 1024 * 1024;  // 1GB reserved for tensor metadata
    header->tensor_data_offset = header->model_state_offset + 1024 * 1024 * 1024;  // 1GB reserved for model state
    header->num_tensors = 0;
    pmem_persist(base_char, sizeof(PersistenceHeader));
}

void PersistenceManager::SaveTensorMetadata(void* base, void* data_ptr, const std::string& dtype, const std::string& shape, const std::string& name, int count) {
    std::lock_guard<std::mutex> lock(metadata_mutex);
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    LOG(INFO) << "DEBUG 9: Current number of tensors before initialization = " << header->num_tensors;

    if (header->num_tensors == 0) {  // If first call, initialize
        InitializePersistentMemory(base);
        LOG(INFO) << "DEBUG 10: Initialized persistent memory because it was the first call.";
    }

    // Calculate the current offset for the new metadata
    size_t current_offset = header->tensor_metadata_offset + header->num_tensors * sizeof(TensorMetadata);
    TensorMetadata* metadata_ptr = new(base_char + current_offset) TensorMetadata;

    // Copy data pointer offset
    
    
    // Copy strings into the char arrays
    std::strncpy(metadata_ptr->name, name.c_str(), sizeof(metadata_ptr->name));
    metadata_ptr->name[sizeof(metadata_ptr->name) - 1] = '\0';  // Ensure null-termination
    std::strncpy(metadata_ptr->dtype, dtype.c_str(), sizeof(metadata_ptr->dtype));
    metadata_ptr->dtype[sizeof(metadata_ptr->dtype) - 1] = '\0';  // Ensure null-termination
    std::strncpy(metadata_ptr->shape, shape.c_str(), sizeof(metadata_ptr->shape));
    metadata_ptr->shape[sizeof(metadata_ptr->shape) - 1] = '\0';  // Ensure null-termination

    size_t data_offset = static_cast<char*>(data_ptr) - base_char;
    metadata_ptr->offset = data_offset;
    metadata_ptr->t_id = count;
    LOG(INFO) << "DEBUG 11: Tensor Metadata saved. Name = " << metadata_ptr->name <<", Count = "<<metadata_ptr->t_id<<
     ", Dtype = " << metadata_ptr->dtype << ", Shape = " << metadata_ptr->shape<<", base: "<<size_t(base_char)<<", offset = "<<metadata_ptr->offset;

    pmem_persist(metadata_ptr, sizeof(TensorMetadata));

    metadata_registry.push_back(*metadata_ptr);  // Add to in-memory registry for quick access
    header->num_tensors++;
    pmem_persist(&(header->num_tensors), sizeof(header->num_tensors));

    LOG(INFO) << "DEBUG 12: Total number of tensors after addition = " << header->num_tensors;
}






void* PersistenceManager::LoadTensorData(void* base, const std::string& name, int count) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);
    LOG(INFO) << "DEBUG 4: " << header->num_tensors;

    for (size_t i = 0; i < header->num_tensors; ++i) {
        //LOG(INFO) << "DEBUG 6: Processing Tensor " << i;
        size_t metadata_offset = header->tensor_metadata_offset + i * sizeof(TensorMetadata);
        TensorMetadata* meta_ptr = reinterpret_cast<TensorMetadata*>(base_char + metadata_offset);
        std::string tensorName(meta_ptr->name, strnlen(meta_ptr->name, sizeof(meta_ptr->name)));
        //LOG(INFO) << "DEBUG 7: Current Tensor Name = " << tensorName;
        
        if (std::strcmp(meta_ptr->name, name.c_str()) == 0 && meta_ptr->t_id == count) {
               LOG(INFO) << "DEBUG 8: Match found.";
                LOG(INFO) << "Tensor Name: " << meta_ptr->name;
                LOG(INFO) << "Count: " << meta_ptr->t_id;
                LOG(INFO) << "Data Type: " << meta_ptr->dtype;
                LOG(INFO) << "Shape: " << meta_ptr->shape;
                LOG(INFO) << "Offset in NVM: " << meta_ptr->offset;
                LOG(INFO) << "Size of Tensor: " << meta_ptr->size;
                LOG(INFO) << "Base address: " << size_t(base_char);
            return base_char + meta_ptr->offset;
        }
    }

    LOG(INFO) << "DEBUG 8: No matching tensor metadata found for name: " << name;
    std::cerr << "No matching tensor metadata found for name: " << name << std::endl;
    return nullptr;
}








void PersistenceManager::SaveModelState(void* base, uint8_t epoch, uint8_t batch, uint16_t op, uint8_t learningRate, uint8_t loss, uint16_t state) {
    std::lock_guard<std::mutex> lock(metadata_mutex);
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    if (header->model_state_offset == 0) {
        InitializePersistentMemory(base);
    }

    // Add the new log to the buffer
    ModelStateLog log = {epoch, batch, op, learningRate, loss, state};
    model_state_buffer.push_back(log);

    // If the buffer is full, flush the logs to persistent memory
    if (model_state_buffer.size() >= buffer_size_limit) {
        FlushModelStateLogs(base);
    }
}

void PersistenceManager::LoadModelState(void* base, std::vector<ModelStateLog>& logs) {
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    size_t logCount = header->model_state_size / sizeof(ModelStateLog);
    logs.reserve(logCount);

    for (size_t i = 0; i < logCount; ++i) {
        ModelStateLog* log = reinterpret_cast<ModelStateLog*>(base_char + header->model_state_offset + i * sizeof(ModelStateLog));
        logs.push_back(*log);
    }
}

void PersistenceManager::FlushModelStateLogs(void* base) {
    std::lock_guard<std::mutex> lock(metadata_mutex);
    char* base_char = static_cast<char*>(base);
    PersistenceHeader* header = reinterpret_cast<PersistenceHeader*>(base_char);

    // Calculate the position to save the new logs
    size_t logPosition = header->model_state_offset + header->model_state_size;

    // Save each log in the buffer to persistent memory
    for (const auto& log : model_state_buffer) {
        ModelStateLog* logPtr = new (base_char + logPosition) ModelStateLog(log);
        pmem_persist(logPtr, sizeof(ModelStateLog));
        logPosition += sizeof(ModelStateLog);
    }

    // Update header information
    header->model_state_size += model_state_buffer.size() * sizeof(ModelStateLog);
    pmem_persist(&(header->model_state_size), sizeof(header->model_state_size));

    // Clear the buffer
    model_state_buffer.clear();
}
