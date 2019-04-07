/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <condition_variable>
#include <utility>
#include <mutex>
#include <cuda_runtime.h>
#include <functional>
#include <fstream>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/common_runtime/device.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

class TensorParams;
class TensorBuffer;
class Device;
class DeviceContext;

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public BFCAllocator {
 public:
  // 'cuda_gpu_id' refers to the ID of the GPU device within
  // the process and must reference a valid ID in the process.
  GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                  const string& name);
  GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                  const GPUOptions& gpu_options, const string& name);
  virtual ~GPUBFCAllocator() {
    const std::string invalid_swap_filename = "/tmp/invalid_swap.txt";
    std::fstream fout(invalid_swap_filename, fout.out);
    if (!fout.is_open()) {
      LOG(ERROR) << "Fail to open invalid swap file";
      return;
    }
    for (auto& name : invalid_swap_) {
      fout << name << "\n";
    }
  }

  void RecordSwapContext(const TensorParams& params, TensorBuffer* tensor_buf);

  void RecordTensorAccess(const string& tensor_name, const uint64 _time);

  void Notify(TensorBuffer* tensor_buf);

 private:

  enum SwapStatus {
    IN,
    OUT,
    SWAPPING_IN,
    SWAPPING_OUT
  };


  inline void MergeChunks(BFCAllocator::ChunkHandle h1, BFCAllocator::ChunkHandle h2) {
    CHECK(h1 != kInvalidChunkHandle && h2 != kInvalidChunkHandle);
    BFCAllocator::Chunk* c1 = ChunkFromHandle(h1);
    BFCAllocator::Chunk* c2 = ChunkFromHandle(h2);
    //CHECK(!c2->in_use());

    BFCAllocator::ChunkHandle h3 = c2->next;
    c1->next = h3;
    CHECK(c2->prev == h1);
    if (h3 != kInvalidChunkHandle) {
      BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
      c3->prev = h1;
    }

    // Set new size
    c1->size += c2->size;

    DeleteChunk(h2);
  }

  inline void MergeBuffers(const void* ptr1, const void* ptr2) {
    BFCAllocator::ChunkHandle h1 = region_manager_.get_handle(ptr1);
    BFCAllocator::ChunkHandle h2 = region_manager_.get_handle(ptr2);
    CHECK(h1 != kInvalidChunkHandle && h2 != kInvalidChunkHandle);
    MergeChunks(h1, h2);
  }

  inline void SplitBuffer(const void* ptr, size_t num_bytes) {
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    CHECK(h != kInvalidChunkHandle)
        << "Asked for SplitChunk of pointer we never allocated: " << ptr;
    BFCAllocator::Chunk* c = ChunkFromHandle(h);
    size_t rounded_bytes = RoundedBytes(num_bytes);
    CHECK(c->requested_size > num_bytes)
        << "Rounded bytes of the split number of bytes is largger than requested size of pointer: " << ptr;
    SplitChunkInUse(h, rounded_bytes);
  }

  void SplitChunkInUse(BFCAllocator::ChunkHandle h, size_t num_bytes) {
    ChunkHandle h_new_chunk = AllocateChunk();
    Chunk* c = ChunkFromHandle(h);
    CHECK(c->in_use() && (c->bin_num == kInvalidBinNum));

    // Create a new chunk starting num_bytes after c
    BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
    new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
    region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

    // Set the new sizes of the chunks.
    new_chunk->size = c->size - num_bytes;
    c->size = num_bytes;

    // The new chunk is not in use.
    new_chunk->allocation_id = -1;

    // Maintain the pointers.
    // c <-> c_neighbor becomes
    // c <-> new_chunk <-> c_neighbor
    BFCAllocator::ChunkHandle h_neighbor = c->next;
    new_chunk->prev = h;
    new_chunk->next = h_neighbor;
    c->next = h_new_chunk;
    if (h_neighbor != kInvalidChunkHandle) {
      Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
      c_neighbor->prev = h_new_chunk;
    }

    // Add the newly free chunk to the free bin.
    InsertFreeChunkIntoBin(h_new_chunk);
  }

  void SwapIn(const string& tensor_name);

  inline void SwapOut(const string& tensor_name) {
    SwapOut(tensor_name, 0);
  }

  void SwapOut(const string& tensor_name, const int64 retain_size);

  void LoadSwapPolicy();

  mutable std::mutex lock_;

  std::mutex mu_;

  typedef std::pair<std::shared_ptr<std::condition_variable>, std::shared_ptr<std::mutex> > condition_variable_and_mutex;

  struct TensorSwapParams {
    string tensor_name;
    Device* device;
    DeviceContext* device_context;
    TensorBuffer* tensor_buffer;
    std::pair<void*, int64> swapped_cpu_buffer; // set if buffer swapped out
    std::pair<void*, int64> swapped_gpu_buffer; // set if buffer swapped out
    condition_variable_and_mutex cv_mu;
    volatile int data_ready; // false if buffer swapped out
    bool can_deallocate_after_swap_out;
    bool then_deallocate;
    float out_fraction;
  };

  struct TriggerInfo {
    string tensor_name;
    int access_count;
    int out_trigger_count;  // 0 if tensor will not be swapped out
    int in_trigger_count;   // 0 if tensor is not a trigger node of any swap tensor
    int total_access_count; // total access count of this tensor in one step
    string in_tensor; // in_tensor will be swapped in if the tensor is accessed in_trigger_count times,
                      // do nothing if in_trigger equals 0
    string in_trigger;
    TensorSwapParams* out_params;  // swap params of in_tensor
    TensorSwapParams* in_params;   // swap params of this tensor
  };

  std::unordered_map<std::string, TriggerInfo> swap_triggers_;

  std::unordered_map<std::string, TensorSwapParams> tensor_swap_params_map_;

  std::unordered_map<const TensorBuffer*, std::string> buffer_tensor_map_;

  std::unordered_map<std::string, std::vector<uint64> > tensor_access_times_ GUARDED_BY(lock_);

  std::unordered_set<std::string> invalid_swap_;
  
  static cudaStream_t device_to_device_stream_;

  static cudaStream_t host_to_device_stream_;

  static cudaStream_t device_to_host_stream_;

  static cudaEvent_t cuda_event_;

  static void CUDART_CB CudaCallback(cudaStream_t stream, cudaError_t status, void* done) {
    auto func = static_cast<std::function<void()>* >(done);
    (*func)();
    delete func;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
};

// Suballocator for GPU memory.
class GPUMemAllocator : public SubAllocator {
 public:
  // Note: stream_exec cannot be null.
  explicit GPUMemAllocator(perftools::gputools::StreamExecutor* stream_exec)
      : stream_exec_(stream_exec) {
    CHECK(stream_exec_ != nullptr);
  }
  ~GPUMemAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      ptr = stream_exec_->AllocateArray<char>(num_bytes).opaque();
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      gpu::DeviceMemoryBase gpu_ptr(ptr);
      stream_exec_->Deallocate(&gpu_ptr);
    }
  }

 private:
  perftools::gputools::StreamExecutor* stream_exec_;  // not owned, non-null

  TF_DISALLOW_COPY_AND_ASSIGN(GPUMemAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
