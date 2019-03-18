#include "tensorflow/core/common_runtime/recompute.h"

#include <fstream>
#include <chrono>
#include <thread>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

void RecomputeHelper::RecordTensorAccess(const std::string& tensor_name, const uint64 time_) {
  if (tensor_recompute_params_.count(tensor_name)) {
    RecomputeTensor(tensor_name);
    auto& recompute_params = tensor_recompute_params_[tensor_name];
    auto& cv_mu = recompute_params.cv_mu;
    volatile int* ready = &(recompute_params.data_ready);
    std::unique_lock<std::mutex> l(*(cv_mu.second));
    cv_mu.first->wait(l, [ready]() { return *ready == DataStatus::IN; });
  }

  if (!triggers_.count(tensor_name)) {
    return;
  }

  auto& trigger = triggers_[tensor_name];
  int cnt;
  {
    std::lock_guard<std::mutex> l(mu_);
    cnt = ++trigger.access_count;
    if (trigger.access_count == trigger.total_access_count) {
      trigger.access_count = 0;
    }
  }
  if (trigger.delete_trigger_count != 0 && cnt == trigger.delete_trigger_count) {
    DeleteMemory(trigger.tensor_name);
  }
  if (cnt <= trigger.recompute_tensors.size()) {
    for (auto& t : trigger.recompute_tensors[cnt-1]) 
      RecomputeTensor(t);
  }
}

void RecomputeHelper::RecomputeTensor(const std::string& tensor_name) {
  if (!recompute_calls.count(tensor_name)) {
    LOG(FATAL) << "Don't have the recompute call for " << tensor_name;
  }

  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  volatile int* ready = &(params.data_ready);
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  if (*ready == DataStatus::OUT) {
    LOG(INFO) << "Recompute " << tensor_name;
    *ready = DataStatus::RECOMPUTING;
    recompute_calls[tensor_name](params.target_tensor, params.feed_tensors, [&tensor_name, &cv_mu, ready]() {
        std::unique_lock<std::mutex> l(*(cv_mu.second));
        *ready = DataStatus::IN;
        cv_mu.first->notify_all();
        LOG(INFO) << "Recompute done " << tensor_name;
      });
  }
}

void RecomputeHelper::RecordTensorBuffer(const std::string& tensor_name, Tensor* tensor) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  tensor_recompute_params_[tensor_name].buf = tensor->buffer();
  tensor_recompute_params_[tensor_name].data_ready = DataStatus::IN;
}

void RecomputeHelper::RecordRecomputeCall(const std::string& tensor_name, RecomputeCall call) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  recompute_calls[tensor_name] = std::move(call);
}

void RecomputeHelper::IncrementUsingCount(const std::string& tensor_name) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_[tensor_name];
  std::lock_guard<std::mutex> l(mu_);
  params.using_count++;
}

void RecomputeHelper::DecrementUsingCount(const std::string& tensor_name) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  params.using_count--;
  if (params.using_count == 0 && params.then_delete) {
    TensorBuffer* buf = params.buf;
    Allocator* alloc = buf->GetAllocator();
    alloc->DeallocateRaw(buf->data());
    buf->set_data(nullptr);
    params.data_ready = DataStatus::OUT;
    params.then_delete = false;
    LOG(INFO) << "Deleted " << tensor_name;
  } else if (params.using_count < 0) {
    LOG(FATAL) << "Using count of " << tensor_name << " is less than 0.";
    params.using_count = 0;
  }
}

void RecomputeHelper::DeleteMemory(const std::string& tensor_name) {
  LOG(INFO) << "Deleting memory of " << tensor_name;
  auto& params= tensor_recompute_params_[tensor_name];
  if (!params.buf) {
    LOG(FATAL) << "Tensor buffer used but not initialzed.";
    return;
  }
  TensorBuffer* buf = params.buf;
  Allocator* alloc = buf->GetAllocator();
  auto& cv_mu = params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  if (params.using_count == 0) {
    alloc->DeallocateRaw(buf->data());
    buf->set_data(nullptr);
    params.data_ready = DataStatus::OUT;
    LOG(INFO) << "Deleted " << tensor_name;
  } else {
    params.then_delete = true;
  }
  LOG(INFO) << "Out delete memory of " << tensor_name;
}

void RecomputeHelper::LoadRecomputePolicy() {
  std::string policy_file = "/tmp/recompute_policy.txt";
  std::fstream fin(policy_file, fin.in);
  if (!fin.is_open()) {
    LOG(INFO) << "open " << policy_file << " failed.";
    return;
  }
  std::string target_tensor, trigger_tensor, feed_tensor, line;
  int del_cnt, total1, compute_cnt, total2;
  while(std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    iss >> target_tensor >> total1 >> del_cnt >> trigger_tensor >> total2 >> compute_cnt;
    auto& params = tensor_recompute_params_[target_tensor];
    params.target_tensor = target_tensor;
    params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());
    while(iss >> feed_tensor) {
      params.feed_tensors.push_back(feed_tensor);
    }
    params.data_ready = DataStatus::OUT;
    params.buf = nullptr;
    params.using_count = 0;
    params.then_delete = false;

    auto& delete_trigger = triggers_[target_tensor];
    delete_trigger.tensor_name = target_tensor;
    delete_trigger.access_count = 0;
    delete_trigger.delete_trigger_count = del_cnt;
    delete_trigger.total_access_count = total1;

    if (compute_cnt > 0) {
      auto& compute_trigger = triggers_[trigger_tensor];
      compute_trigger.tensor_name = trigger_tensor;
      compute_trigger.access_count = 0;
      compute_trigger.total_access_count = total2;
      compute_trigger.recompute_tensors.resize(total2);
      compute_trigger.recompute_tensors[compute_cnt-1].push_back(target_tensor);
    }
  }
  LOG(INFO) << "Recompute policy file loaded.";
}

} // tensorflow
