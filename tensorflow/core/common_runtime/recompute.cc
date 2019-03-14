#include "tensorflow/core/common_runtime/recompute.h"

#include <fstream>
#include <chrono>
#include <thread>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

void RecomputeHelper::RecordTensorAccess(const std::string& tensor_name, const uint64 time_, RecomputeCall recompute) {
  if (tensor_recompute_params_map_.count(tensor_name)) {
    LOG(INFO) << "RecordTensorAccess " << tensor_name;
    auto& recompute_params = tensor_recompute_params_map_[tensor_name];
    auto& cv_mu = recompute_params.cv_mu;
    volatile int* ready = &(recompute_params.data_ready);
    if (recompute_params.data_ready == DataStatus::OUT) {
      LOG(INFO) << "Recompute " << tensor_name;
      *ready = DataStatus::RECOMPUTING;
      recompute(recompute_params.target_tensor, recompute_params.feed_tensors, [&tensor_name, &cv_mu, ready]() {
        std::unique_lock<std::mutex> l(*(cv_mu.second));
        *ready = DataStatus::IN;
        cv_mu.first->notify_all();
        LOG(INFO) << "Recompute done " << tensor_name;
      });
    }
    std::unique_lock<std::mutex> l(*(cv_mu.second));
    cv_mu.first->wait(l, [ready]() { return *ready == DataStatus::IN; });
  }

  if (!delete_triggers_.count(tensor_name) && !compute_triggers_.count(tensor_name)) {
    return;
  }

  auto& delete_trigger = delete_triggers_[tensor_name];
  int cnt;
  {
    std::lock_guard<std::mutex> l(mu_);
    cnt = ++delete_trigger.access_count;
    if (delete_trigger.access_count == delete_trigger.total_access_count) {
      delete_trigger.access_count = 0;
    }
  }
  if (delete_trigger.delete_trigger_count != 0 && cnt == delete_trigger.delete_trigger_count) {
    DeleteMemory(delete_trigger.tensor_name);
  }
  //LOG(INFO) << "Out RecordTensorAccess " << tensor_name;
  //if (trigger.recompute_trigger_count != 0 && cnt == trigger.recompute_trigger_count) {
  //  RecomputeTensor(trigger.recompute_tensor);
  //}
}

void RecomputeHelper::RecordTensorBuffer(const std::string& tensor_name, Tensor* tensor) {
  if (!tensor_recompute_params_map_.count(tensor_name)) return;
  tensor_recompute_params_map_[tensor_name].buf = tensor->buffer();
  tensor_recompute_params_map_[tensor_name].data_ready = DataStatus::IN;
}

void RecomputeHelper::IncrementUsingCount(const std::string& tensor_name) {
  if (!tensor_recompute_params_map_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_map_[tensor_name];
  std::lock_guard<std::mutex> l(mu_);
  params.using_count++;
}

void RecomputeHelper::DecrementUsingCount(const std::string& tensor_name) {
  if (!tensor_recompute_params_map_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_map_[tensor_name];
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
  auto& params= tensor_recompute_params_map_[tensor_name];
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
    std::istringstream iss(line);
    iss >> target_tensor >> total1 >> del_cnt >> trigger_tensor >> total2 >> compute_cnt;
    auto& params = tensor_recompute_params_map_[target_tensor];
    params.target_tensor = target_tensor;
    params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());
    while(iss >> feed_tensor) {
      params.feed_tensors.push_back(feed_tensor);
    }
    params.data_ready = DataStatus::OUT;
    params.buf = nullptr;
    params.using_count = 0;
    params.then_delete = false;

    auto& delete_trigger = delete_triggers_[target_tensor];
    delete_trigger.tensor_name = target_tensor;
    delete_trigger.access_count = 0;
    delete_trigger.delete_trigger_count = del_cnt;
    delete_trigger.total_access_count = total1;

    auto& compute_trigger = compute_triggers_[trigger_tensor];
    compute_trigger.tensor_name = trigger_tensor;
    compute_trigger.access_count = 0;
    compute_trigger.compute_trigger_count = compute_cnt;
    compute_trigger.total_access_count = total2;
    compute_trigger.params = &params;
  }
  LOG(INFO) << "Recompute policy file loaded.";
}

} // tensorflow
