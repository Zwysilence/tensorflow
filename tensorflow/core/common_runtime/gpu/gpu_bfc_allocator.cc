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
#include <fstream>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>

using perftools::gputools::DeviceMemoryBase;
using perftools::gputools::Stream;


namespace tensorflow {

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
    }

void GPUBFCAllocator::RecordTensorAccess(const string& tensor_name, const uint64 _time) {
  if (tensor_swap_params_map_.count(tensor_name)) {
    SwapIn(tensor_name);
    auto &swap_params = tensor_swap_params_map_[tensor_name];
    auto &cv_mu = swap_params.cv_mu;
    std::unique_lock<std::mutex> l(*(cv_mu.second));
    int* ready = &(swap_params.data_ready);
    cv_mu.first->wait(l, [ready]() { return *ready == SwapStatus::IN; });
    l.unlock();
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
  buffer_tensor_map_[tensor_buf] = tensor_name;
}

void GPUBFCAllocator::Notify(const TensorBuffer* tensor_buf) {
  if (buffer_tensor_map_.count(tensor_buf) == 0) return;
  const string& tensor_name = buffer_tensor_map_[tensor_buf];
  auto &cv_mu = tensor_swap_params_map_[tensor_name].cv_mu;
  cv_mu.first->notify_one();
}

void GPUBFCAllocator::LoadSwapPolicy() {
  std::string swap_policy_file = "/tmp/swap_policy.txt";
  std::fstream fin(swap_policy_file, fin.in);
  if (!fin.is_open()) {
    LOG(FATAL) << "open " << swap_policy_file << " failed.";
    return;
  }
  string out_tensor_name, in_trigger_name;
  int out_trigger_count, in_trigger_count;
  int out_tensor_total_access, in_trigger_total_access;
  while(fin >> out_tensor_name >> out_tensor_total_access >> out_trigger_count 
            >> in_trigger_name >> in_trigger_total_access >> in_trigger_count) {
    auto& swap_params = tensor_swap_params_map_[out_tensor_name];
    swap_params.tensor_name = out_tensor_name;
    swap_params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());

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

void GPUBFCAllocator::SwapOut(const string& tensor_name) {
  CHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];
  auto &cv_mu = swap_params.cv_mu;
  {
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    int *ready = &(swap_params.data_ready);
    if (*ready != SwapStatus::IN)
      return;
    *ready = SwapStatus::SWAPPING_OUT;
  }
  Device* device = swap_params.device;
  DeviceContext* device_context = swap_params.device_context;
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  gpu::Stream* send_stream = nullptr;
  Status s = PrepareCopy(device, device_context, &dev_info, &send_stream);
  if (!s.ok()) {
    LOG(FATAL) << "PrepareCopy failed.";
    return;
  }

  static Allocator* cuda_host_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);
  TensorBuffer* tensor_buffer = swap_params.tensor_buffer;
  void *src_ptr = (void*)(tensor_buffer->data());
  size_t size = RequestedSize(src_ptr);
  void* dst_ptr = cuda_host_allocator->AllocateRaw(0, size);
  if (dst_ptr == nullptr) {
    LOG(FATAL) << "Allocate host memory failed.";
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    LOG(FATAL) << "No send gpu copy-out-stream is available.";
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  send_device_to_host_stream->ThenWaitFor(send_stream);

  int64 total_bytes = size;
  if (total_bytes > 0) {
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    send_device_to_host_stream->ThenMemcpy(dst_ptr, gpu_src_ptr, total_bytes);
  }
  // Use of the input may outlive stack scope, so keep a ref.
  tensor_buffer->Ref();
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream,
      [send_device_to_host_stream, tensor_buffer, this, &swap_params, dst_ptr]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
          tensor_buffer->Unref();
          return;
        }
        tensor_buffer->Unref();
        auto &cv_mu = swap_params.cv_mu;
        std::unique_lock<std::mutex> lk(*(cv_mu.second));
        if (swap_params.can_deallocate_after_swap_out) {
          cv_mu.first->wait(lk, [tensor_buffer]() { 
            return tensor_buffer->UsingCount() == 0; 
          });
          swap_params.data_ready = SwapStatus::OUT;
          this->DeallocateRaw(tensor_buffer->data());
          tensor_buffer->set_data(nullptr);
        } else {
          cuda_host_allocator->DeallocateRaw(dst_ptr);
          swap_params.data_ready = SwapStatus::IN;
          swap_params.can_deallocate_after_swap_out = true;
        }
        cv_mu.first->notify_all();
        lk.unlock();
      });
  swap_params.cpu_buffer = std::make_pair(dst_ptr, total_bytes);
}

void GPUBFCAllocator::SwapIn(const string& tensor_name) {
  CHECK(tensor_swap_params_map_.count(tensor_name));
  auto &swap_params = tensor_swap_params_map_[tensor_name];
  auto &cv_mu = swap_params.cv_mu;
  {
    std::unique_lock<std::mutex> l(*(cv_mu.second));
    int *ready = &(swap_params.data_ready);
    if (*ready != SwapStatus::OUT) {
      if (*ready == SwapStatus::SWAPPING_OUT)
        swap_params.can_deallocate_after_swap_out = false;
      return;
    }
    *ready = SwapStatus::SWAPPING_IN;
  }
  auto& cpu_buffer = swap_params.cpu_buffer;
  void* src_ptr = cpu_buffer.first;
  int64 total_bytes = cpu_buffer.second;
  void* dst_ptr = AllocateRaw(0, total_bytes);

  Device * device = swap_params.device;
  DeviceContext * device_context = swap_params.device_context;
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  gpu::Stream* recv_stream = nullptr;
  Status s = PrepareCopy(device, device_context, &dev_info, &recv_stream);
  if (!s.ok()) {
    LOG(FATAL) << "PrepareCopy failed.";
    return;
  }

  auto recv_host_to_device_stream =
    static_cast<const GPUDeviceContext*>(device_context)->host_to_device_stream();
  if (recv_host_to_device_stream == nullptr) {
    LOG(FATAL) << "No send gpu copy-out-stream is available.";
    return;
  }
  // Wait for the recv-stream to make sure the buffer is truly available.
  recv_host_to_device_stream->ThenWaitFor(recv_stream);

  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
    recv_host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, src_ptr, total_bytes);
  }
  static Allocator* cuda_host_allocator = ProcessState::singleton()->GetCUDAHostAllocator(0);
  // Use of cpu_tensor may outlive stack scope, so keep a ref.
  dev_info->event_mgr->ThenExecute(
      recv_host_to_device_stream,
      [recv_host_to_device_stream, this, &swap_params, src_ptr, dst_ptr]() {
        if (!recv_host_to_device_stream->ok()) {
          LOG(FATAL) << "CPU->GPU Memcpy failed";
          return;
        }
        auto& cv_mu = swap_params.cv_mu;
        std::lock_guard<std::mutex> lk(*(cv_mu.second));
        swap_params.data_ready = SwapStatus::IN;
        swap_params.tensor_buffer->set_data(dst_ptr);
        cuda_host_allocator->DeallocateRaw(src_ptr);
        cv_mu.first->notify_one();
      });
}

}  // namespace tensorflow
