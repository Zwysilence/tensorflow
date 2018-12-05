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
  virtual ~GPUBFCAllocator() {}

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
    int data_ready; // false if buffer swapped out
    bool can_deallocate_after_swap_out;
    bool then_deallocate;
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

  static cudaStream_t device_to_device_stream_;

  static cudaStream_t host_to_device_stream_;

  static cudaStream_t device_to_host_stream_;

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
