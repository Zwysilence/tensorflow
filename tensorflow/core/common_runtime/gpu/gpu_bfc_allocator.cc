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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <cstdlib>

using perftools::gputools::DeviceMemoryBase;
using perftools::gputools::Stream;

#define cudaCheckError(cudaCall) {                                                  \
    cudaError_t err = cudaCall;                                                       \
    if(err!=cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
      exit(0);                                                                        \
    }                                                                                 \
}


namespace tensorflow {

std::string GetEnv(const string& env_name) {
  const char* env_p = std::getenv(env_name.c_str());
  if (env_p == nullptr) return "";
  return env_p;
}

const int64 kCopyThreshold = 2 << 20;    // 2M

cudaStream_t GPUBFCAllocator::device_to_device_stream_;
cudaStream_t GPUBFCAllocator::host_to_device_stream_;
cudaStream_t GPUBFCAllocator::device_to_host_stream_;
cudaEvent_t GPUBFCAllocator::cuda_event_;

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const string& name)
    : GPUBFCAllocator(cuda_gpu_id, total_memory, GPUOptions(), name) {}

GPUBFCAllocator::GPUBFCAllocator(CudaGpuId cuda_gpu_id, size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name)
    : BFCAllocator(
          new GPUMemAllocator(
              GpuIdUtil::ExecutorForCudaGpuId(cuda_gpu_id).ValueOrDie()),
          total_memory, gpu_options.allow_growth(), name) {
      LoadSwapPolicy();
      cudaStreamCreate(&device_to_device_stream_);
      cudaStreamCreate(&host_to_device_stream_);
      cudaStreamCreate(&device_to_host_stream_);
      cudaEventCreate(&cuda_event_);
    }

void GPUBFCAllocator::RecordTensorAccess(const string& tensor_name, const uint64 _time) {
  if (tensor_swap_params_map_.count(tensor_name)) {
    SwapIn(tensor_name);
    auto &swap_params = tensor_swap_params_map_[tensor_name];
    auto &cv_mu = swap_params.cv_mu;
    std::unique_lock<std::mutex> l(*(cv_mu.second));
    volatile int* ready = &(swap_params.data_ready);
    static bool partial_swap = (GetEnv("PARTIAL_SWAP") == "true" ? true : false);
    if (partial_swap && *ready != SwapStatus::IN) {
      swap_params.out_fraction = 0.5;
    }
    cv_mu.first->wait(l, [ready]() { return *ready == SwapStatus::IN; });
  }

  if (swap_triggers_.count(tensor_name) == 0) {
    return;
  }

  auto& swap_trigger = swap_triggers_[tensor_name];
  int cnt;
  {
    std::lock_guard<std::mutex> l(mu_);
    swap_trigger.access_count++;
    cnt = swap_trigger.access_count;
    if (swap_trigger.access_count == swap_trigger.total_access_count) {
      swap_trigger.access_count = 0;
    }
  }
  if (swap_trigger.out_trigger_count != 0 && cnt == swap_trigger.out_trigger_count) {
    SwapOut(tensor_name);
  }

  if (swap_trigger.in_trigger_count != 0 && cnt == swap_trigger.in_trigger_count) {
    SwapIn(swap_trigger.in_tensor);
  }
}

void GPUBFCAllocator::RecordSwapContext(const TensorParams& params, TensorBuffer* tensor_buf) {
  std::lock_guard<std::mutex> l(lock_);
  if (tensor_swap_params_map_.count(params.name) == 0) return;
  const string &tensor_name = params.name;
  TensorSwapParams& swap_params = tensor_swap_params_map_[tensor_name];
  swap_params.device = params.device;
  swap_params.device_context = params.device_context;
  swap_params.tensor_buffer = tensor_buf;
  swap_params.data_ready = SwapStatus::IN;
  swap_params.can_deallocate_after_swap_out = true;
  swap_params.then_deallocate = false;
  buffer_tensor_map_[tensor_buf] = tensor_name;
}

void GPUBFCAllocator::Notify(TensorBuffer* tensor_buffer) {
  if (buffer_tensor_map_.count(tensor_buffer) == 0) return;
  const string& tensor_name = buffer_tensor_map_[tensor_buffer];
  auto& swap_params = tensor_swap_params_map_[tensor_name];
  auto& cv_mu = swap_params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  int64 gpu_part_size = swap_params.swapped_gpu_buffer.second;
  if (swap_params.can_deallocate_after_swap_out && swap_params.then_deallocate) {
    if (gpu_part_size <= 0)
      DeallocateRaw(tensor_buffer->data());
    else
      SplitBuffer(tensor_buffer->data(), gpu_part_size);
    tensor_buffer->set_data(nullptr);
    swap_params.data_ready = SwapStatus::OUT;
    swap_params.then_deallocate = false;
  }

  if (!swap_params.can_deallocate_after_swap_out && swap_params.then_deallocate) {
    swap_params.data_ready = SwapStatus::IN;
    cv_mu.first->notify_all();
  }
}

void GPUBFCAllocator::LoadSwapPolicy() {
  std::string swap_policy_file = "/tmp/swap_policy.txt";
  std::fstream fin(swap_policy_file, fin.in);
  if (!fin.is_open()) {
    LOG(INFO) << "open " << swap_policy_file << " failed.";
    return;
  }
  string out_tensor_name, in_trigger_name;
  int out_trigger_count, in_trigger_count;
  int out_tensor_total_access, in_trigger_total_access;
  while(fin >> out_tensor_name >> out_tensor_total_access >> out_trigger_count
            >> in_trigger_name >> in_trigger_total_access >> in_trigger_count) {
    if (out_tensor_name[0] == '#') {
      continue;
    }
    auto& swap_params = tensor_swap_params_map_[out_tensor_name];
    swap_params.tensor_name = out_tensor_name;
    swap_params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());
    swap_params.out_fraction = 1.0f;

    auto& swap_out_trigger = swap_triggers_[out_tensor_name];
    swap_out_trigger.tensor_name = out_tensor_name;
    swap_out_trigger.out_trigger_count = out_trigger_count;
    swap_out_trigger.out_params = &swap_params;
    swap_out_trigger.in_trigger = in_trigger_name;
    swap_out_trigger.access_count = 0;
    swap_out_trigger.total_access_count = out_tensor_total_access;

    auto& swap_in_trigger = swap_triggers_[in_trigger_name];
    swap_in_trigger.tensor_name = in_trigger_name;
    swap_in_trigger.in_trigger_count = in_trigger_count;
    swap_in_trigger.in_tensor = out_tensor_name;
    swap_in_trigger.in_params = &swap_params;
    swap_in_trigger.access_count = 0;
    swap_in_trigger.total_access_count = in_trigger_total_access;
  }
}

Status PrepareCopy(Device* device, const DeviceContext* ctx,
    const DeviceBase::GpuDeviceInfo** dev_info, gpu::Stream** stream) {
  if (device == nullptr) {
    return errors::Internal("Unexpected null device.");
  }
  auto di = device->tensorflow_gpu_device_info();
  if (di == nullptr) {
    return errors::Internal("Unexpected null device info.");
  }
  *dev_info = di;
  if (ctx == nullptr) {
    return errors::Internal("Unexpected null device context.");
  }
  auto gs = static_cast<const GPUDeviceContext*>(ctx)->stream();
  if (gs == nullptr) {
    return errors::Internal("No gpu stream is available.");
  }
  *stream = gs;
  return Status::OK();
}

void GPUBFCAllocator::SwapOut(const string& tensor_name, const int64 retain_size) {
  if (invalid_swap_.count(tensor_name) != 0) {
    return;
  }
  CHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];
  auto &cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    int ready = swap_params.data_ready;
    if (ready != SwapStatus::IN)
      return;
    swap_params.data_ready = SwapStatus::SWAPPING_OUT;
  }

  TensorBuffer* tensor_buffer = swap_params.tensor_buffer;
  float out_fraction = swap_params.out_fraction;
  void* src_ptr = (void*)(tensor_buffer->data());
  int64 total_bytes = RequestedSize(src_ptr);
  int64 gpu_part_size, cpu_part_size;
  if (fabs(out_fraction) < 1e-6) {
    gpu_part_size = total_bytes;
    cpu_part_size = 0;
  } else if (fabs(out_fraction - 1.0f) < 1e-6) {
    gpu_part_size = 0;
    cpu_part_size = total_bytes;
  } else {
    cpu_part_size = int(total_bytes * out_fraction) / 4 * 4;
    gpu_part_size = total_bytes - cpu_part_size;
  }

  if (cpu_part_size < kCopyThreshold) {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::IN;
    return;
  }

  LOG(INFO) << "Start to swap out " << tensor_name;

  swap_params.swapped_gpu_buffer = std::make_pair(src_ptr, gpu_part_size);

  static Allocator* cuda_host_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);
  void* cpu_part_dst_ptr = cuda_host_allocator->AllocateRaw(0, cpu_part_size);
  if (cpu_part_dst_ptr == nullptr) {
    LOG(FATAL) << "Allocate host memory failed.";
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    swap_params.data_ready = SwapStatus::IN;
    return;
  }

  cudaCheckError(cudaMemcpyAsync(cpu_part_dst_ptr, (void*)((uintptr_t)src_ptr + gpu_part_size), cpu_part_size, cudaMemcpyDeviceToHost, device_to_host_stream_));
  // Use of the input may outlive stack scope, so keep a ref.
  tensor_buffer->Ref();
  std::function<void()>* doneD2H = new std::function<void()>(
      [this, tensor_buffer, gpu_part_size, cpu_part_dst_ptr, &swap_params] {
        auto &cv_mu = swap_params.cv_mu;
        std::unique_lock<std::mutex> lk(*(cv_mu.second));
        if (cpu_part_dst_ptr != swap_params.swapped_cpu_buffer.first)
          return;
        // NOTE: assume gpu->gpu part is completed first than gpu->cpu part.
        if (swap_params.can_deallocate_after_swap_out) {
          if (tensor_buffer->UsingCount() == 0) {
            swap_params.data_ready = SwapStatus::OUT;
            if (gpu_part_size <= 0)
              DeallocateRaw(tensor_buffer->data());
            else
              SplitBuffer(tensor_buffer->data(), gpu_part_size);
            tensor_buffer->set_data(nullptr);
          } else {
            swap_params.then_deallocate = true;
          }
        } else {
          cuda_host_allocator->DeallocateRaw(cpu_part_dst_ptr);
          swap_params.data_ready = SwapStatus::IN;
          swap_params.can_deallocate_after_swap_out = true;
        }
        cv_mu.first->notify_all();
        tensor_buffer->Unref();
      });
  cudaCheckError(cudaStreamAddCallback(device_to_host_stream_, CudaCallback, (void*)doneD2H, 0));
  swap_params.swapped_cpu_buffer = std::make_pair(cpu_part_dst_ptr, cpu_part_size);
}

void GPUBFCAllocator::SwapIn(const string& tensor_name) {
  std::lock_guard<std::mutex> l(lock_);
  if (invalid_swap_.count(tensor_name) != 0) {
    return;
  }
  CHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];
  auto &cv_mu = swap_params.cv_mu;
  {
    std::unique_lock<std::mutex> l(*(cv_mu.second));
    int ready = swap_params.data_ready;
    if (ready != SwapStatus::OUT) {
      if (ready == SwapStatus::SWAPPING_OUT) {
        //swap_params.data_ready = SwapStatus::IN;  // this can lead to larger memory pressure
        LOG(WARNING) << "Swap in when swapping out not finish: " << tensor_name;
        if (invalid_swap_.insert(tensor_name).second) {
          LOG(INFO) << "Push " << tensor_name << " into invalid swap success";
        } else {
          LOG(ERROR) << "Push " << tensor_name << " into invalid swap failed";
        }
        swap_params.can_deallocate_after_swap_out = false;
      }
      return;
    }
    swap_params.data_ready = SwapStatus::SWAPPING_IN;
  }

  void* gpu_part_src_ptr = swap_params.swapped_gpu_buffer.first;
  void* cpu_part_src_ptr = swap_params.swapped_cpu_buffer.first;
  int64 gpu_part_size = swap_params.swapped_gpu_buffer.second;
  int64 cpu_part_size = swap_params.swapped_cpu_buffer.second;

  static Allocator* cuda_host_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);

  if (gpu_part_size > 0) {
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(gpu_part_src_ptr);
    CHECK(h != kInvalidChunkHandle);
    BFCAllocator::Chunk* c = ChunkFromHandle(h);
    BFCAllocator::Chunk* c_next = nullptr;
    if (c->next != kInvalidChunkHandle) {
      c_next = ChunkFromHandle(c->next);
    }
    mutex_lock ll(BFCAllocator::lock_);
    if (c_next && c_next->size >= cpu_part_size && !c_next->in_use()) {
      RemoveFreeChunkFromBin(c->next);
      c_next->allocation_id = next_allocation_id_++;
      void* gpu_part2_ptr = c_next->ptr;
      std::function<void()>* done = new std::function<void()>(
          [this, gpu_part_src_ptr, gpu_part2_ptr, cpu_part_src_ptr, &swap_params]() {
            auto& cv_mu = swap_params.cv_mu;
            std::lock_guard<std::mutex> l(*(cv_mu.second));
            swap_params.data_ready = SwapStatus::IN;
            MergeBuffers(gpu_part_src_ptr, gpu_part2_ptr);
            swap_params.tensor_buffer->set_data(gpu_part_src_ptr);
            cuda_host_allocator->DeallocateRaw(cpu_part_src_ptr);
            cv_mu.first->notify_all();
          });
      cudaCheckError(cudaMemcpyAsync(c_next->ptr, cpu_part_src_ptr, cpu_part_size, cudaMemcpyHostToDevice, host_to_device_stream_));
      cudaCheckError(cudaStreamAddCallback(host_to_device_stream_, CudaCallback, (void*)done , 0));
      return;
    }
  }

  void* dst_ptr = AllocateRaw(0, gpu_part_size + cpu_part_size);

  if (gpu_part_size > 0) {
    cudaCheckError(cudaMemcpyAsync(dst_ptr, gpu_part_src_ptr, gpu_part_size, cudaMemcpyDeviceToDevice, device_to_device_stream_));
    std::function<void()>* doneD2D = new std::function<void()>(
        [this, gpu_part_src_ptr, &swap_params]() {
          DeallocateRaw(gpu_part_src_ptr);
          // NOTE: assume gpu->gpu part is completed first than gpu->cpu part.
        });
    cudaCheckError(cudaStreamAddCallback(device_to_device_stream_, CudaCallback, (void*)doneD2D , 0));
  }

  cudaCheckError(cudaMemcpyAsync((void*)((uintptr_t)dst_ptr + gpu_part_size), cpu_part_src_ptr, cpu_part_size, cudaMemcpyHostToDevice, host_to_device_stream_));
  // Use of the input may outlive stack scope, so keep a ref.
  std::function<void()>* doneH2D = new std::function<void()>(
      [dst_ptr, cpu_part_src_ptr, &swap_params]() {
        auto &cv_mu = swap_params.cv_mu;
        std::lock_guard<std::mutex> l(*(cv_mu.second));
        swap_params.data_ready = SwapStatus::IN;
        swap_params.tensor_buffer->set_data(dst_ptr);
        cuda_host_allocator->DeallocateRaw(cpu_part_src_ptr);
        cv_mu.first->notify_all();
      });
  cudaCheckError(cudaStreamAddCallback(host_to_device_stream_, CudaCallback, (void*)doneH2D, 0));
}

}  // namespace tensorflow
