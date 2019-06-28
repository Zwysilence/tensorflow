#include "tensorflow/core/common_runtime/eager/recompute.h"

#include <fstream>
#include <chrono>
#include <thread>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"

#define _DEBUG

namespace tensorflow {

const std::string recompute_policy_env = "RECOMPUTE_POLICY_FILE";

static std::string GetEnv(const std::string& env) {
  const char* val = std::getenv(env.c_str());
  return val ? val : "";
}

void EagerRecomputeHelper::RecordTensorAccess(const std::string& tensor_name, const uint64 time_) {
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
  //if (cnt <= trigger.recompute_tensors.size()) {
  //  RecomputeTensor(trigger.recompute_tensors[cnt-1][0]);
  //}
}

void EagerRecomputeHelper::SetRecomputing(const std::vector<std::string>& recompute_nodes) {
  int cnt = 0;
  for (auto& node_name : recompute_nodes) {
    for (auto& tensor_name : node_to_tensors_[node_name]) {
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

void EagerRecomputeHelper::SetRecomputing(const std::string& target) {
  if (!recompute_tensors_.count(target)) return;
  auto& recompute_tensors = recompute_tensors_[target];
  for (auto& tensor_name : recompute_tensors) {
    auto& params = tensor_recompute_params_[tensor_name];
    auto& cv_mu = params.cv_mu;
    std::lock_guard<std::mutex> l(*(cv_mu.second));
    if (params.data_ready == DataStatus::OUT) {
      params.data_ready = DataStatus::RECOMPUTING;
    }
  }
}

void EagerRecomputeHelper::SaveRecomputedTensor(const std::string& target, bool is_ref, const std::pair<std::string, Tensor*>& recomputed) {
  if (!tensor_recompute_params_.count(target) || !tensor_recompute_params_.count(recomputed.first))
    return;
  #ifdef _DEBUG
  LOG(INFO) << "Save " << recomputed.first;
  #endif
  if (recomputed.second->data() == nullptr) LOG(INFO) << recomputed.second->Name() << " is null";
  saved_tensors_[target][recomputed.first] = *(recomputed.second);
  if (is_ref) {
    LOG(FATAL) << "entry is a reference, handle it now.";
  }
}

void EagerRecomputeHelper::RecomputeTensor(const std::string& tensor_name) {
  if (!recompute_calls_.count(tensor_name)) {
    LOG(FATAL) << "Don't have the recompute call for " << tensor_name;
  }

  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  volatile int* ready = &(params.data_ready);
  std::unique_lock<std::mutex> ul(*(cv_mu.second));
  if (*ready == DataStatus::OUT) {
    /* {
      std::lock_guard<std::mutex> l(mu_);
      if (*ready != DataStatus::OUT) return;
    } */
  #ifdef _DEBUG
    LOG(INFO) << "Recompute " << tensor_name;
  #endif
    *ready = DataStatus::RECOMPUTING;
    ul.unlock();
    recompute_calls_[tensor_name](params.target_tensor, recompute_tensors_[tensor_name], params.feed_tensors);
    SetRecomputedTensors(tensor_name);
  }
}

void EagerRecomputeHelper::SetRecomputedTensors(const std::string& target) {
  std::lock_guard<std::mutex> l(mu_);
  auto& tensors = saved_tensors_[target];
  for (auto& t : tensors) {
    auto& params = tensor_recompute_params_[t.first];
    auto& cv_mu = params.cv_mu;
    std::unique_lock<std::mutex> ul(*(cv_mu.second));
    if (params.data_ready != DataStatus::IN) {
      if (params.buf->data() != nullptr)
        LOG(FATAL) << "Buffer data should be null";
      params.buf->set_data(t.second.data());
      t.second.set_data(nullptr);
      params.data_ready = DataStatus::IN;
      cv_mu.first->notify_all();
    #ifdef _DEBUG
      LOG(INFO) << "Recompute " << t.first << " done";
    #endif
    }
  }
  tensors.clear();
}

void EagerRecomputeHelper::RecordTensorBuffer(const std::string& tensor_name, Tensor* tensor) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  tensor_recompute_params_[tensor_name].buf = tensor->buffer();
  tensor_recompute_params_[tensor_name].data_ready = DataStatus::IN;
}

void EagerRecomputeHelper::RecordRecomputeCall(const std::string& tensor_name, RecomputeCall call) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  recompute_calls_[tensor_name] = std::move(call);
}

void EagerRecomputeHelper::IncrementUsingCount(const std::string& tensor_name) {
  if (!tensor_recompute_params_.count(tensor_name)) return;
  auto& params = tensor_recompute_params_[tensor_name];
  auto& cv_mu = params.cv_mu;
  // std::lock_guard<std::mutex> l(mu_);
  std::lock_guard<std::mutex> l(*(cv_mu.second));  
  params.using_count++;
}

void EagerRecomputeHelper::DecrementUsingCount(const std::string& tensor_name) {
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
    LOG(INFO) << "Deleted " << tensor_name;
  #endif
  } else if (params.using_count < 0) {
    LOG(FATAL) << "Using count of " << tensor_name << " is less than 0.";
    params.using_count = 0;
  }
}

void EagerRecomputeHelper::DeleteMemory(const std::string& tensor_name) {
  auto& params= tensor_recompute_params_[tensor_name];
  if (!params.buf) {
    LOG(FATAL) << "Tensor buffer used but not initialzed.";
    return;
  }
  #ifdef _DEBUG
  LOG(INFO) << "Deleting memory of " << tensor_name << "(" << readable_names_[tensor_name] << ") Buffer " << params.buf;
  #endif
  TensorBuffer* buf = params.buf;
  Allocator* alloc = buf->GetAllocator();
  auto& cv_mu = params.cv_mu;
  std::lock_guard<std::mutex> l(*(cv_mu.second));
  if (params.using_count == 0) {
    alloc->DeallocateRaw(buf->data());
    buf->set_data(nullptr);
    params.data_ready = DataStatus::OUT;
  #ifdef _DEBUG
    LOG(INFO) << "Deleted " << tensor_name; // << "(" << readable_names_[tensor_name] << ") Buffer " << buf;
  #endif
  } else if (params.using_count > 0) {
    params.then_delete = true;
    params.data_ready = DataStatus::OUT;
  } else {
    LOG(FATAL) << "Using count of " << tensor_name << " is less than 0.";
  }
}

void EagerRecomputeHelper::LoadRecomputePolicy() {
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
  std::string tensor_name, op_name, line;
  int total_delete, total_recompute;
  std::getline(fin, line);
  std::istringstream iss(line);
  iss >> total_delete >> total_recompute;
  int total, count, num;
  while(total_delete-- && std::getline(fin, line)) {
    // tensor_name total_access_per_iter delete_trigger_count 
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    iss >> tensor_name >> total >> count;
    auto& params = tensor_recompute_params_[tensor_name];
    params.target_tensor = tensor_name;
    params.cv_mu = std::make_pair(std::make_shared<std::condition_variable>(), std::make_shared<std::mutex>());
    string node_name = tensor_name.substr(0, tensor_name.find(':'));
    node_to_tensors_[node_name].push_back(tensor_name);

    auto& delete_trigger = triggers_[tensor_name];
    delete_trigger.tensor_name = tensor_name;
    delete_trigger.delete_trigger_count = count;
    delete_trigger.total_access_count = total;
  }

  while(total_recompute-- && std::getline(fin, line)) {
    // trigger_tensor_name total_access_per_iter recompute_trigger_count num_recompute_tensors ... num_op_names ... num_feed_tensors ...
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    iss >> tensor_name >> total >> count;
    auto& compute_trigger = triggers_[tensor_name];
    compute_trigger.tensor_name = tensor_name;
    compute_trigger.total_access_count = total;
    compute_trigger.recompute_tensors.resize(total);

    std::string tname, opname;
    // consume recompute tensor names
    iss >> num;
    num--;
    std::string target_tensor;
    iss >> target_tensor;
    auto& recompute_tensors = recompute_tensors_[target_tensor];
    if (recompute_tensors.empty()) recompute_tensors.push_back(target_tensor);
    while(num--) {
      iss >> tname;
      recompute_tensors.push_back(tname);
      //compute_trigger.recompute_tensors[count-1].push_back(tname);
    }

    // consume op names
    iss >> num;
    while(num--) {
      iss >> opname;
      node_input_ref_count_[opname]++;
    }

    // comsume feed tensor names
    auto& params = tensor_recompute_params_[target_tensor];
    iss >> num;
    while(num--) {
      iss >> tname;
      params.feed_tensors.push_back(tname);
    }

  }
  fin.close();
  LOG(INFO) << "Recompute policy file loaded.";
}

} // tensorflow
