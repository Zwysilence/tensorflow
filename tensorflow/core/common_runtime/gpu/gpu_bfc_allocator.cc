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
          total_memory, gpu_options.allow_growth(), name) {}

void GPUBFCAllocator::SaveTensorTrace() {
  string outfile = "/tmp/TensorTrace.txt";
  std::fstream fout(outfile, fout.out);
  if (!fout.is_open()) {
    LOG(FATAL) << "Can't open " << outfile;
    return;
  }
  for (auto &tensor_times : tensor2times_) {
    fout << tensor_times.first;
    if (tensor_times.second.size() == 0) {
      fout << "\n";
      continue;
    };
    fout << " {\n";
    for (auto time_ : tensor_times.second) {
      fout << time_ << "\n";
    }
    fout << "}\n";
  }
  fout.close();
}

void GPUBFCAllocator::RecordTensorTrace(const string& tensor_name, const uint64 _time) {
  mutex_lock l(lock_);
  if (cv_mus_.count(tensor_name)) {
    auto &cv_mu = cv_mus_[tensor_name];
    std::unique_lock<std::mutex> lk(*(cv_mu.second));
    bool * pReady = &tensor_ready_[tensor_name];
    cv_mu.first->wait(lk, [pReady]() { return *pReady; });
    lk.unlock();
  }
  tensor2times_[tensor_name].push_back(_time);
}

void GPUBFCAllocator::MapTensorToBuffer(const TensorParams & params, TensorBuffer * tensor_buf) {
  const string &tensor_name = params.name;
  tensor_buffers_[tensor_name] = tensor_buf;
  tensor_devices_[tensor_name] = params.device;
  tensor_devcxts_[tensor_name] = params.device_context;
  if (swap_nodes_.count(tensor_name)) {
    if (cv_mus_.count(tensor_name) == 0) {
      cv_mus_[tensor_name] = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());
      tensor_ready_[tensor_name] = true;
    }
    //SwapOut(tensor_name);
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
  Device * device = tensor_devices_[tensor_name];
  DeviceContext * device_context = tensor_devcxts_[tensor_name];
  TensorBuffer * tensor_buffer = tensor_buffers_[tensor_name];
  void *src_ptr = (void*)(tensor_buffer->data());
  size_t size = RequestedSize(src_ptr);
  void *dst_ptr = malloc(size);
  if (dst_ptr == nullptr) {
    return;
  }

  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  gpu::Stream* send_stream = nullptr;
  Status s = PrepareCopy(device, device_context, &dev_info, &send_stream);
  if (!s.ok()) {
    //done(s);
    return;
  }

  auto send_device_to_host_stream =
      static_cast<const GPUDeviceContext*>(device_context)->device_to_host_stream();
  if (send_device_to_host_stream == nullptr) {
    //done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the sender's main stream to make sure the data are available.
  send_device_to_host_stream->ThenWaitFor(send_stream);

  const int64 total_bytes = size;
  if (total_bytes > 0) {
    DeviceMemoryBase gpu_src_ptr(src_ptr, total_bytes);
    send_device_to_host_stream->ThenMemcpy(dst_ptr, gpu_src_ptr, total_bytes);
  }
  // Use of the input may outlive stack scope, so keep a ref.
  tensor_buffer->Ref();
  dev_info->event_mgr->ThenExecute(
      send_device_to_host_stream,
      //[send_device_to_host_stream, done, tensor_buffer]() {
      [send_device_to_host_stream, tensor_buffer, tensor_name, this]() {
        if (!send_device_to_host_stream->ok()) {
          LOG(FATAL) << "GPU->CPU Memcpy failed";
          tensor_buffer->Unref();
          return;
        }
        tensor_buffer->Unref();
        std::lock_guard<std::mutex> lk(*(this->cv_mus_[tensor_name].second));
        this->tensor_ready_[tensor_name] = false;
        this->DeallocateRaw(tensor_buffer->data());
        //done(Status::OK());
      });
  swapped_tensors_[tensor_name] = std::make_pair(dst_ptr, total_bytes);
}

void GPUBFCAllocator::SwapIn(const string& tensor_name) {
  if (swapped_tensors_.count(tensor_name) == 0)
    return;
  auto& buffer = swapped_tensors_[tensor_name];
  void* src_ptr = buffer.first;
  const int64 total_bytes = buffer.second;
  void* dst_ptr = AllocateRaw(0, total_bytes);

  Device * device = tensor_devices_[tensor_name];
  DeviceContext * device_context = tensor_devcxts_[tensor_name];
  const DeviceBase::GpuDeviceInfo* dev_info = nullptr;
  gpu::Stream* recv_stream = nullptr;
  Status s = PrepareCopy(device, device_context, &dev_info, &recv_stream);
  if (!s.ok()) {
    //done(s);
    return;
  }

  auto recv_host_to_device_stream =
    static_cast<const GPUDeviceContext*>(device_context)->host_to_device_stream();
  if (recv_host_to_device_stream == nullptr) {
    //done(errors::Internal("No send gpu copy-out-stream is available."));
    return;
  }
  // Wait for the recv-stream to make sure the buffer is truly available.
  recv_host_to_device_stream->ThenWaitFor(recv_stream);

  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
    recv_host_to_device_stream->ThenMemcpy(&gpu_dst_ptr, src_ptr, total_bytes);
  }
  // Use of cpu_tensor may outlive stack scope, so keep a ref.
  dev_info->event_mgr->ThenExecute(
      recv_host_to_device_stream,
      [recv_host_to_device_stream, tensor_name, this]() {
        if (!recv_host_to_device_stream->ok()) {
          LOG(FATAL) << "CPU->GPU Memcpy failed";
          return;
        }
        auto& cv_mu = this->cv_mus_[tensor_name];
        std::lock_guard<std::mutex> lk(*(cv_mu.second));
        this->tensor_ready_[tensor_name] = true;
        cv_mu.first->notify_one();
        //done(Status::OK());
      });
   tensor_buffers_[tensor_name]->set_data(dst_ptr);
}

}  // namespace tensorflow
