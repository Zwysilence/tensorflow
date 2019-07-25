#include "tensorflow/core/common_runtime/recompute.h"

#include <fstream>
#include <chrono>
#include <thread>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/graph/graph.h"

// #define _DEBUG

namespace tensorflow {

const std::string recompute_policy_env = "RECOMPUTE_POLICY_FILE";

static std::string GetEnv(const std::string& env) {
  const char* val = std::getenv(env.c_str());
  return val ? val : "";
}

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

void RecomputeHelper::SetRecomputing(const std::string& target_tensor, const std::vector<std::string>& recompute_nodes) {
  int cnt = 0;
  auto& recompute_tensors = recompute_tensors_[target_tensor];
  for (auto& node_name : recompute_nodes) {
    for (auto& tensor_name : node_to_tensors_[node_name]) {
      if (!recompute_tensors.count(tensor_name)) continue;
      auto& params = tensor_recompute_params_[tensor_name];
      auto& cv_mu = params.cv_mu;
      std::lock_guard<std::mutex> l(*(cv_mu.second));
      if (params.data_ready == DataStatus::OUT) {
        params.data_ready = DataStatus::RECOMPUTING;
        cnt++;
      }
    }
  }
#ifdef _DEBUG
  LOG(INFO) << "Size of recompute nodes (within node that hasn't recompute tensor) " << recompute_nodes.size();
  LOG(INFO) << "Size of recompute nodes (without node that hasn't recompute tensor) " << cnt+1;
#endif
}

void RecomputeHelper::SaveRecomputedTensor(const std::string& target, bool is_ref, const std::pair<std::string, Tensor*>& recomputed) {
  if (!tensor_recompute_params_.count(target) || !tensor_recompute_params_.count(recomputed.first) || !recompute_tensors_[target].count(recomputed.first))
    return;
  saved_tensors_[target][recomputed.first] = *(recomputed.second);
  if (is_ref) {
    LOG(FATAL) << "entry is a reference, handle it now.";
  }
}

void RecomputeHelper::RecomputeTensor(const std::string& tensor_name) {
  if (!recompute_calls_.count(tensor_name)) {
    LOG(FATAL) << "Don't have the recompute call for " << tensor_name;
  }

  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  std::unique_lock<std::mutex> ul(*(cv_mu.second));
  if (params.data_ready == DataStatus::OUT) {
    /* {
      std::lock_guard<std::mutex> l(mu_);
      if (*ready != DataStatus::OUT) return;
    } */
  #ifdef _DEBUG
    LOG(INFO) << "Recompute " << tensor_name << " buffer=" << params.buf;
  #endif
    params.data_ready = DataStatus::RECOMPUTING;
    ul.unlock();
    recompute_calls_[tensor_name](params.target_tensor, params.feed_tensors, [&tensor_name, this]() {
        SetRecomputedTensors(tensor_name);
      });
  }
}

void RecomputeHelper::SetRecomputedTensors(const std::string& target) {
  //std::lock_guard<std::mutex> l(mu_);
  auto& tensors = saved_tensors_[target];
  for (auto& t : tensors) {
    auto& params = tensor_recompute_params_[t.first];
    auto& cv_mu = params.cv_mu;
    std::unique_lock<std::mutex> ul(*(cv_mu.second));
    if (params.data_ready != DataStatus::IN) {
      if (params.buf->data() != nullptr) {
        LOG(FATAL) << "Buffer data should be null! " << t.first << " buffer=" << params.buf;
      }
      params.buf->set_data(t.second.data());
      t.second.set_data(nullptr);
      params.data_ready = DataStatus::IN;
      params.node->SetTensorDeleted(t.first, false);
      cv_mu.first->notify_all();
    #ifdef _DEBUG
      LOG(INFO) << "Recompute " << t.first << " done. buffer=" << params.buf;
    #endif
    }
  }
  tensors.clear();
}

void RecomputeHelper::RecordTensorInfo(const std::string& tensor_name, Tensor* tensor, Node* node) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  params.buf = tensor->buffer();
  params.data_ready = DataStatus::IN;
  params.node = node;
#ifdef _DEBUG
  LOG(INFO) << "Record Tensor Info " << tensor_name << " buffer=" << params.buf;
#endif
}

void RecomputeHelper::RecordRecomputeCall(const std::string& tensor_name, RecomputeCall call) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  recompute_calls_[tensor_name] = std::move(call);
}

void RecomputeHelper::IncrementUsingCount(const std::string& tensor_name) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  // std::lock_guard<std::mutex> l(mu_);
  std::lock_guard<std::mutex> l(*(cv_mu.second));  
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
  #ifdef _DEBUG
    LOG(INFO) << "Deleted " << tensor_name << " buffer=" << buf;
  #endif
  } else if (params.using_count < 0) {
    LOG(FATAL) << "Using count of " << tensor_name << " is less than 0.";
    params.using_count = 0;
  }
}

void RecomputeHelper::DeleteMemory(const std::string& tensor_name) {
  auto& params= tensor_recompute_params_[tensor_name];
  if (!params.buf) {
    LOG(FATAL) << "Tensor buffer used but not initialzed.";
    return;
  }
  // LOG(INFO) << "Deleting memory of " << tensor_name << "(" << readable_names_[tensor_name] << ") Buffer " << params.buf;
  auto& cv_mu = params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  params.node->SetTensorDeleted(tensor_name, true);
  TensorBuffer* buf = params.buf;
  Allocator* alloc = buf->GetAllocator();
  if (params.using_count == 0) {
    alloc->DeallocateRaw(buf->data());
    buf->set_data(nullptr);
    params.data_ready = DataStatus::OUT;
  #ifdef _DEBUG
    LOG(INFO) << "Deleted " << tensor_name; // << "(" << readable_names_[tensor_name] << ") Buffer " << buf;
  #endif
  } else if (params.using_count > 0) {
    params.then_delete = true;
    // params.data_ready = DataStatus::OUT;
  #ifdef _DEBUG
    LOG(INFO) << "Then delete " << tensor_name;
  #endif
  } else {
    LOG(FATAL) << "Using count of " << tensor_name << " is less than 0.";
  }
}

void RecomputeHelper::LoadRecomputePolicy() {
  std::string policy_file = GetEnv(recompute_policy_env);
  if (policy_file.empty()) {
    LOG(INFO) << "No recompute policy specified";
    return;
  }
  std::fstream fin(policy_file, fin.in);
  if (!fin.is_open()) {
    LOG(INFO) << "open " << policy_file << " failed.";
    return;
  }
  std::string target_tensor, trigger_tensor, feed_tensor, line;
  int del_cnt, total1, compute_cnt, total2, num_recomp_tensors;
  while(std::getline(fin, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    iss >> target_tensor >> total1 >> del_cnt >> trigger_tensor >> total2 >> compute_cnt;
    auto& params = tensor_recompute_params_[target_tensor];
    params.target_tensor = target_tensor;
    params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());

    recompute_tensors_[target_tensor].insert(target_tensor);
    iss >> num_recomp_tensors;
    string tname;
    while(num_recomp_tensors--) {
      iss >> tname;
      recompute_tensors_[target_tensor].insert(tname);
    }

    while(iss >> feed_tensor) {
      params.feed_tensors.push_back(feed_tensor);
    }

    params.data_ready = DataStatus::OUT;
    params.buf = nullptr;
    params.using_count = 0;
    params.then_delete = false;
    string node_name = target_tensor.substr(0, target_tensor.find(':'));
    node_to_tensors_[node_name].push_back(target_tensor);

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
  fin.close();
  LOG(INFO) << "Recompute policy file loaded.";
}

} // tensorflow
